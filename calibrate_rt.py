"""
calibrate_rt.py — calibrate the RT model to your own LC method.

The RT model predicts retention on a relative 0-1 scale. This script fits a map from that
scale to your measured RT (minutes, seconds or iRT — whatever unit your file uses) and saves
a calibrated copy of the model that the app can load directly.

Input: a CSV / TSV / Excel file with one peptide column and one RT column, e.g. a PEAKS,
MaxQuant, DIA-NN or Skyline export of confidently identified peptides from one run/method.
Column names are detected automatically (Peptide / Sequence / seq ... and RT / tr / Retention time / iRT ...);
override with --pep-col / --rt-col.

Steps
  1. drop peptides the model cannot handle (modified, > 19 residues, non-standard residues)
  2. one RT per sequence (median over repeated identifications)
  3. robust fit: fit, drop outliers (> 4 x MAD from the fit, e.g. wrong IDs), refit
  4. honest error estimate: 5-fold cross-validation split by sequence
  5. final fit on all clean peptides -> saved into the model file

Curve: straight line by default for < 50 peptides, 2nd-order curve from 50 peptides on
(tested on the pea alcalase data: ~10% lower error with enough peptides). Force with --degree.

    python calibrate_rt.py my_ids.csv --model rt_DIA.pt --out rt_DIA_myLC.pt --name "C18 40 min" --unit min

The same calibration is used by the pepMS2 app when you upload a peptide/RT table there.
"""
import argparse
import io
import os
import re
import sys

import numpy as np
import pandas as pd

from rt_model import RTPredictor, check_peptide

PEP_COLS = ["peptide", "sequence", "stripped.sequence", "stripped sequence", "seq",
            "peptide sequence", "peptidesequence", "modified sequence"]
MOD_COLS = ["ptm", "modifications", "modification", "mod", "mods"]
RT_COLS = ["rt", "retention time", "retentiontime", "rt (min)", "rt [min]", "tr",
           "observed rt", "apex rt", "irt"]
MIN_PEPTIDES = 5
MIN_R = 0.8


class CalibrationError(ValueError):
    pass


def read_table(source, name=None):
    """source: path or bytes. name: file name (for the extension) when source is bytes."""
    name = name or (source if isinstance(source, str) else "")
    ext = os.path.splitext(name)[1].lower()
    buf = io.BytesIO(source) if isinstance(source, (bytes, bytearray)) else source
    if ext in (".xlsx", ".xls"):
        try:
            return pd.read_excel(buf)
        except ImportError:
            raise CalibrationError("Reading Excel needs the 'openpyxl' package; save the table as CSV instead.")
    try:
        return pd.read_csv(buf, sep=None, engine="python")  # sniffs comma / tab / semicolon
    except Exception as e:
        raise CalibrationError(f"Could not read the table: {e}")


def _pick(df, wanted, given, what, flag):
    if given:
        if given not in df.columns:
            raise CalibrationError(f"Column '{given}' not found. Columns are: {list(df.columns)}")
        return given
    low = {str(c).strip().lower(): c for c in df.columns}
    for w in wanted:
        if w in low:
            return low[w]
    raise CalibrationError(f"Could not find the {what} column (looked for e.g. {', '.join(wanted[:4])}). "
                           f"Columns are: {list(df.columns)}. Name it with {flag}.")


def guess_columns(df):
    """Auto-detected (peptide column, RT column); None where nothing matched."""
    out = []
    for wanted in (PEP_COLS, RT_COLS):
        try:
            out.append(_pick(df, wanted, None, "", ""))
        except CalibrationError:
            out.append(None)
    return tuple(out)


def clean_sequence(p):
    p = str(p).strip()
    m = re.fullmatch(r"[A-Z\-_]\.(.+)\.[A-Z\-_]", p)   # K.PEPTIDE.R  -> PEPTIDE
    if m:
        p = m.group(1)
    return p.strip("_")                               # _PEPTIDE_ (MaxQuant)  -> PEPTIDE


def robust_fit(x, y, degree, n_iter=3, k=4.0):
    keep = np.ones(len(x), bool)
    for _ in range(n_iter):
        coef = np.polyfit(x[keep], y[keep], degree)
        res = y - np.polyval(coef, x)
        mad = np.median(np.abs(res[keep] - np.median(res[keep]))) * 1.4826
        new_keep = np.abs(res) <= max(k * mad, 1e-9)
        if new_keep.sum() < degree + 2 or (new_keep == keep).all():
            break
        keep = new_keep
    return np.polyfit(x[keep], y[keep], degree), keep


def cross_validate(x, y, degree, folds=5, seed=0):
    """Out-of-fold error; folds split by unique sequence (one row per sequence here)."""
    n = len(x)
    if n < 2 * folds:
        return None
    idx = np.random.default_rng(seed).permutation(n)
    err = np.empty(n)
    for f in range(folds):
        test = idx[f::folds]
        train = np.setdiff1d(idx, test)
        coef, _ = robust_fit(x[train], y[train], degree)
        err[test] = np.polyval(coef, x[test]) - y[test]
    return err


def calibrate_table(raw_df, model, pep_col=None, rt_col=None, degree=None, keep_modified=False,
                    name="", unit="", force=False):
    """Fit a calibration. Sets model.calibration / model.calibration_info and returns
    (data, messages): per-peptide DataFrame and a list of human-readable notes.
    Raises CalibrationError if the table cannot be used."""
    if model.is_calibrated:
        raise CalibrationError("This RT model is already calibrated; start from the uncalibrated model (rt_DIA.pt).")
    msgs = []
    pc = _pick(raw_df, PEP_COLS, pep_col, "peptide", "--pep-col")
    rc = _pick(raw_df, RT_COLS, rt_col, "RT", "--rt-col")
    msgs.append(f"Read {len(raw_df)} rows (peptide column '{pc}', RT column '{rc}')")

    low = {str(c).strip().lower(): c for c in raw_df.columns}
    mc = next((low[m] for m in MOD_COLS if m in low), None)
    df = pd.DataFrame({"peptide": raw_df[pc].map(clean_sequence),
                       "rt": pd.to_numeric(raw_df[rc], errors="coerce")})
    n_bad_rt = int(df.rt.isna().sum())
    if n_bad_rt == len(df):
        raise CalibrationError(f"Column '{rc}' has no numeric RT values; pick the column with the retention times.")
    reason = df["peptide"].map(check_peptide)
    df["skip_reason"] = reason.map(lambda r: None if not isinstance(r, str) else
                                   ("modified / non-standard residues" if "non-standard" in r else "longer than 19 residues"))
    if mc is not None and not keep_modified:
        has_mod = raw_df[mc].fillna("").astype(str).str.strip().replace({"Unmodified": "", "nan": ""}) != ""
        df.loc[has_mod & df.skip_reason.isna(), "skip_reason"] = f"modified (flagged in '{mc}' column)"
    df = df.dropna(subset=["rt"])
    if n_bad_rt:
        msgs.append(f"Skipped {n_bad_rt} rows without a numeric RT")
    skipped = df[df.skip_reason.notna()]
    if len(skipped):
        msgs.append(f"Skipped {len(skipped)} rows the model can't handle: " +
                    "; ".join(f"{n} {r}" for r, n in skipped.skip_reason.value_counts().items()))
    df = df[df.skip_reason.isna()]

    grp = df.groupby("peptide").rt
    data = pd.DataFrame({"measured_rt": grp.median(), "n_ids": grp.size(),
                         "rt_spread": grp.max() - grp.min()}).reset_index()
    n = len(data)
    msgs.append(f"{n} unique peptides usable for calibration")
    if n < MIN_PEPTIDES:
        raise CalibrationError(f"Only {n} usable peptides; at least {MIN_PEPTIDES} are needed.")
    if n < 20:
        msgs.append("Fewer than 20 peptides: the calibration will be rough")

    x = model.predict_raw(list(data.peptide)).astype(float)
    y = data.measured_rt.values.astype(float)
    degree = int(degree or (1 if n < 50 else 2))
    cv = cross_validate(x, y, degree)
    coef, keep = robust_fit(x, y, degree)
    r = float(np.corrcoef(x[keep], y[keep])[0, 1])
    if not np.isfinite(r) or (r < MIN_R and not force):
        raise CalibrationError(
            f"Model output and your RTs barely agree (r = {r:.2f}). Usually the wrong RT or peptide column was "
            f"picked, the file mixes several LC methods, or the identifications are unreliable.")
    grid = np.linspace(x.min(), x.max(), 200)
    if degree > 1 and not np.all(np.diff(np.polyval(coef, grid)) > 0):
        msgs.append("The fitted curve bends back inside your RT range: use a straight line (degree 1)")

    data["model_output"] = x
    data["predicted_rt"] = np.polyval(coef, x)
    data["residual"] = data.predicted_rt - data.measured_rt
    data["used_in_fit"] = keep
    if cv is not None:
        data["cv_residual"] = cv

    info = {"name": name, "unit": unit, "n_peptides": int(n), "n_used": int(keep.sum()), "degree": degree,
            "rt_range": (float(y.min()), float(y.max())),
            "model_output_range": (float(x.min()), float(x.max())), "pearson_r": round(r, 4)}
    if cv is not None:
        info["cv_median_abs_error"] = round(float(np.median(np.abs(cv))), 3)
        info["cv_abs_error_95pct"] = round(float(np.percentile(np.abs(cv), 95)), 3)
    model.calibration = coef
    model.calibration_info = info
    return data, msgs


def report_text(model, model_path, out):
    info, coef = model.calibration_info, model.calibration
    u = f" {info['unit']}" if info.get("unit") else ""
    lines = [
        f"RT calibration — {info['name']}",
        f"model            : {model_path}  ->  {out}",
        f"peptides         : {info['n_peptides']} unique, {info['n_used']} used, "
        f"{info['n_peptides'] - info['n_used']} outliers removed",
        f"fit              : {'straight line' if info['degree'] == 1 else 'polynomial degree %d' % info['degree']}"
        f"   coefficients {np.array2string(np.asarray(coef), precision=4)}",
        f"RT range covered : {info['rt_range'][0]:.2f} - {info['rt_range'][1]:.2f}{u}",
        f"correlation (r)  : {info['pearson_r']:.3f}",
    ]
    if "cv_median_abs_error" in info:
        lines.append(f"expected error   : median {info['cv_median_abs_error']:.2f}{u}, 95% of peptides within "
                     f"{info['cv_abs_error_95pct']:.2f}{u} (5-fold cross-validation)")
    lines.append("Predictions outside the RT range above are extrapolations; treat them as rough.")
    return "\n".join(lines)


def save_plot(data, model, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x, y, keep = data.model_output.values, data.measured_rt.values, data.used_in_fit.values
    grid = np.linspace(x.min(), x.max(), 200)
    u = model.calibration_info.get("unit") or ""
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
    ax[0].scatter(x[keep], y[keep], s=6, alpha=.5, color="#1f77b4", label="used")
    if (~keep).any():
        ax[0].scatter(x[~keep], y[~keep], s=10, color="#d62728", marker="x", label="outlier (removed)")
    ax[0].plot(grid, np.polyval(model.calibration, grid), color="black", lw=1.5, label="calibration")
    ax[0].set_xlabel("model output (relative 0-1)"), ax[0].set_ylabel(f"measured RT {u}".strip())
    ax[0].legend(frameon=False), ax[0].set_title("Calibration fit")
    cv = "cv_residual" in data
    ax[1].scatter(y, data.cv_residual if cv else data.residual, s=6, alpha=.5, color="#1f77b4")
    ax[1].axhline(0, color="black", lw=1)
    ax[1].set_xlabel(f"measured RT {u}".strip()), ax[1].set_ylabel("predicted − measured")
    ax[1].set_title("Error " + ("(cross-validated)" if cv else "(in-sample)"))
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("table", help="CSV/TSV/XLSX with peptide sequences and measured RT")
    ap.add_argument("--model", default="rt_DIA.pt", help="uncalibrated model (default rt_DIA.pt)")
    ap.add_argument("--out", help="calibrated model to write (default <model>_calibrated.pt)")
    ap.add_argument("--pep-col"), ap.add_argument("--rt-col")
    ap.add_argument("--degree", type=int, choices=[1, 2, 3], help="1 = line, 2/3 = curve (default: auto)")
    ap.add_argument("--keep-modified", action="store_true",
                    help="also use rows flagged in a PTM/Modifications column (they are predicted as unmodified)")
    ap.add_argument("--force", action="store_true", help=f"save even if the fit looks wrong (r < {MIN_R})")
    ap.add_argument("--name", default="", help="label for this LC method, shown in the app")
    ap.add_argument("--unit", default="min", help="RT unit of your file: min, s, iRT ... (default min)")
    a = ap.parse_args()

    out = a.out or os.path.splitext(a.model)[0] + "_calibrated.pt"
    stem = os.path.splitext(out)[0]
    model = RTPredictor(a.model)
    try:
        data, msgs = calibrate_table(read_table(a.table), model, a.pep_col, a.rt_col, a.degree, a.keep_modified,
                                     a.name or os.path.basename(a.table), a.unit, a.force)
    except CalibrationError as e:
        for m in getattr(e, "msgs", []):
            print(m)
        sys.exit(f"Stopped: {e}")
    for m in msgs:
        print(m)
    model.calibration_info["source_table"] = os.path.basename(a.table)
    model.save(out)
    data.sort_values("measured_rt").to_csv(stem + "_calibration.csv", index=False)
    report = report_text(model, a.model, out)
    open(stem + "_calibration.txt", "w", encoding="utf-8").write(report + "\n")
    print("\n" + report)
    try:
        save_plot(data, model, stem + "_calibration.png")
        print(f"\nSaved {out}, {stem}_calibration.txt / .csv / .png")
    except ImportError:
        print(f"\nSaved {out}, {stem}_calibration.txt / .csv  (install matplotlib for the plot)")


if __name__ == "__main__":
    main()
