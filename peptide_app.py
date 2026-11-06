"""Peptide MS/MS spectrum predictor - local web app.

Start:  streamlit run peptide_app.py
Needs:  pip install streamlit plotly pandas numpy torch
Same folder: peptide_model.py and peptide_predictor.pt (written by the notebook's export cell).
Retention time (optional): rt_model.py, calibrate_rt.py and rt_DIA.pt in the same folder. Without them the
spectrum part works as before; with them the app shows a relative RT, or the RT in your own units once an
RT calibration (calibrated .pt or a peptide/RT table) is uploaded in the sidebar.
"""
import inspect
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

PROTON, WATER = 1.007276, 18.010565
RESIDUE = {  # monoisotopic residue masses
    'G': 57.02146, 'A': 71.03711, 'S': 87.03203, 'P': 97.05276, 'V': 99.06841, 'T': 101.04768,
    'C': 103.00919, 'L': 113.08406, 'I': 113.08406, 'N': 114.04293, 'D': 115.02694, 'Q': 128.05858,
    'K': 128.09496, 'E': 129.04259, 'M': 131.04049, 'H': 137.05891, 'F': 147.06841, 'R': 156.10111,
    'Y': 163.06333, 'W': 186.07931}
CYS_CARBAMIDOMETHYL = 160.03065


# ---------------------------------------------------------------- fragment maths (no torch needed)
def residue_masses(cys_alkylated=False):
    res = dict(RESIDUE)
    if cys_alkylated:
        res['C'] = CYS_CARBAMIDOMETHYL
    return res


def neutral_mass(seq, res):
    return sum(res[a] for a in seq) + WATER


def ion_table(seq, intensities, n_frag, charge, res):
    """b/y ions with m/z and intensity (also as % of the base peak).
    `intensities` is ordered y1..yN then b1..bN, like the training file."""
    r = np.array([res[a] for a in seq])
    b = (np.cumsum(r)[:-1] + charge * PROTON) / charge
    y = (np.cumsum(r[::-1])[:-1] + WATER + charge * PROTON) / charge
    n = min(len(seq) - 1, n_frag)
    k = np.arange(1, n + 1)
    tab = pd.DataFrame({
        'ion': [f'y{i}' for i in k] + [f'b{i}' for i in k],
        'type': ['y'] * n + ['b'] * n,
        'mz': np.concatenate([y[:n], b[:n]]),
        'intensity': np.concatenate([intensities[:n], intensities[n_frag:n_frag + n]])})
    top = tab['intensity'].max()
    tab['rel'] = 100 * tab['intensity'] / top if top > 0 else 0.0
    return tab.sort_values('mz').reset_index(drop=True)


def check_sequence(raw, vocab, max_len_trained):
    """Returns (clean sequence, error message or None, warning message or None)."""
    seq = ''.join(raw.split()).upper()
    if not seq:
        return seq, None, None
    if not seq.isalpha():
        return seq, ('Only plain one-letter amino-acid codes are allowed. Modification tags such as M(ox) or '
                     'C[160], digits and symbols cannot be entered.'), None
    bad = sorted({c for c in seq if c not in vocab})
    if bad:
        return seq, (f"Not supported by this model: {', '.join(bad)}. Use plain one-letter codes of residues the "
                     f"model was trained on; modifications such as M(ox) cannot be entered."), None
    if len(seq) < 2:
        return seq, 'A peptide needs at least 2 residues.', None
    warn = None
    if len(seq) > max_len_trained:
        warn = (f'This peptide ({len(seq)} residues) is longer than any peptide the model was trained on '
                f'({max_len_trained}); treat the prediction with caution.')
    return seq, None, warn


# ---------------------------------------------------------------- plot
C_LIGHT = {'b': '#2a78d6', 'y': '#eb6834'}        # colour-blind-safe pair (light / dark theme steps)
C_DARK = {'b': '#3987e5', 'y': '#d95926'}


def _rgba(hex_colour, alpha):
    h = hex_colour.lstrip('#')
    return f'rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{alpha:.2f})'


def build_figure(tab, n_labels=12, cutoff=0.0, dark=False):
    """Stick spectrum: b ions blue, y ions orange. Sticks have a fixed pixel width (independent of the m/z range),
    every stick ends in a dot, and ions are labelled directly (all of them if there are few)."""
    col = C_DARK if dark else C_LIGHT
    show = tab[tab['rel'] >= cutoff]
    n_label = len(show) if len(show) <= 16 else n_labels            # few ions: label every one
    labelled = set(show.nlargest(n_label, 'rel').index) if n_label > 0 else set()
    span = max(tab['mz'].max() - tab['mz'].min(), 1.0)
    fig = go.Figure()
    for t, name in (('y', 'y ions'), ('b', 'b ions')):
        d = show[show['type'] == t]
        if d.empty:
            continue
        xs, ys = [], []
        for mz, rel in zip(d['mz'], d['rel']):
            xs += [mz, mz, None]
            ys += [0, rel, None]
        fig.add_scatter(x=xs, y=ys, mode='lines', line=dict(color=col[t], width=4), name=name,
                        hoverinfo='skip', legendgroup=t)
        fig.add_scatter(
            x=d['mz'], y=d['rel'], mode='markers+text', showlegend=False, legendgroup=t,
            marker=dict(size=9, color=col[t]),
            text=[ion if i in labelled else '' for i, ion in zip(d.index, d['ion'])],
            textposition='top center', textfont=dict(size=13, color=col[t]), cliponaxis=False,
            customdata=d['ion'],
            hovertemplate='<b>%{customdata}</b><br>m/z %{x:.4f}<br>%{y:.1f}% of base peak<extra></extra>')
    fig.update_layout(
        height=440, hovermode='closest', hoverdistance=30,
        xaxis_title='m/z', yaxis_title='relative intensity (%)',
        xaxis_range=[tab['mz'].min() - span * 0.06, tab['mz'].max() + span * 0.06],
        yaxis_range=[0, 118],
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        margin=dict(l=10, r=10, t=40, b=10))
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(gridcolor='rgba(128,128,128,0.25)', zeroline=True, zerolinecolor='rgba(128,128,128,0.5)',
                     tickvals=[0, 25, 50, 75, 100])
    return fig


def ladder_figure(seq, tab, dark=False):
    """Fragment map: the sequence with, at every peptide bond, the b ion (above, N-terminal side) and the
    complementary y ion (below). Strength of colour and line thickness follow the predicted intensity."""
    col = C_DARK if dark else C_LIGHT
    L = len(seq)
    rel = {r['ion']: r['rel'] for _, r in tab.iterrows()}
    fig = go.Figure()
    fig.add_scatter(x=list(range(1, L + 1)), y=[0] * L, mode='text', text=list(seq), showlegend=False,
                    textfont=dict(size=26), hoverinfo='skip')
    for j in range(1, L):                                      # bond between residue j and j+1
        for t, k, ysign in (('b', j, 1), ('y', L - j, -1)):
            r = rel.get(f'{t}{k}')
            if r is None:
                continue
            strength = 0.12 + 0.88 * r / 100
            x = j + 0.5
            # corner mark: vertical bar through the bond, with a short horizontal arm towards the fragment
            arm = -0.28 if t == 'b' else 0.28
            fig.add_scatter(x=[x + arm, x, x], y=[ysign * 0.95, ysign * 0.95, ysign * 0.45], mode='lines',
                            line=dict(color=_rgba(col[t], strength), width=1.5 + 3.5 * r / 100),
                            showlegend=False, hoverinfo='skip')
            fig.add_scatter(x=[x], y=[ysign * 1.45], mode='text', text=[f'{t}{k}'], showlegend=False,
                            textfont=dict(size=11, color=_rgba(col[t], max(strength, 0.45))),
                            hovertemplate=f'{t}{k}: {r:.1f}% of base peak<extra></extra>')
    fig.update_layout(height=170, margin=dict(l=10, r=10, t=5, b=5), hovermode='closest',
                      xaxis=dict(visible=False, range=[0.3, L + 0.7]), yaxis=dict(visible=False, range=[-2.0, 2.0]))
    return fig


def show_plot(fig):
    # `use_container_width` is being replaced by `width` in newer Streamlit versions
    if 'width' in inspect.signature(st.plotly_chart).parameters:
        st.plotly_chart(fig, width='stretch')
    else:
        st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------- model loading
@st.cache_resource(show_spinner='Loading model...')
def get_model(path):
    from peptide_model import load_checkpoint
    return load_checkpoint(path)


# ---------------------------------------------------------------- retention time
RT_TABLE_TYPES = ('csv', 'tsv', 'txt', 'xlsx')


@st.cache_resource(show_spinner='Loading RT model...')
def get_rt_base(path, mtime):
    from rt_model import RTPredictor
    return RTPredictor(path)


@st.cache_resource(show_spinner='Loading RT calibration...')
def get_rt_from_pt(data):
    from rt_model import RTPredictor
    return RTPredictor(data)


@st.cache_data(show_spinner='Reading table...')
def read_rt_table(data, file_name):
    from calibrate_rt import read_table
    return read_table(data, file_name)


@st.cache_resource(show_spinner='Calibrating RT model on your peptides...')
def get_rt_from_table(data, file_name, base_path, unit, method_name, pep_col, rt_col):
    from rt_model import RTPredictor
    from calibrate_rt import calibrate_table
    model = RTPredictor(base_path)
    table, msgs = calibrate_table(read_rt_table(data, file_name), model, pep_col=pep_col, rt_col=rt_col,
                                  unit=unit, name=method_name)
    model.calibration_info['source_table'] = file_name
    return model, table, msgs


def rt_sidebar(here):
    """Sidebar block for retention time. Returns (rt_model or None, calibration table or None, notes)."""
    base_path = os.path.join(here, 'rt_DIA.pt')
    st.header('Retention time')
    if not os.path.exists(base_path):
        st.caption('RT prediction is off: put rt_DIA.pt and rt_model.py next to this app to switch it on.')
        return None, None, []
    try:
        base = get_rt_base(base_path, os.path.getmtime(base_path))
    except Exception as e:
        st.warning(f'Could not load the RT model ({e}). RT prediction is off.')
        return None, None, []

    up = st.file_uploader(
        'RT calibration (optional)', type=('pt',) + RT_TABLE_TYPES, key='rt_upload',
        help='Either a calibrated RT model (.pt made by calibrate_rt.py or downloaded from this app), or a table '
             'of identified peptides with their measured RT from one LC method (CSV/TSV/Excel, columns such as '
             'Peptide and RT). Without a calibration the app shows the relative RT (0-1).')
    if up is None:
        st.caption('No calibration: showing relative RT on a 0-1 scale (early to late elution).')
        return base, None, []

    data = up.getvalue()
    ext = os.path.splitext(up.name)[1].lower().lstrip('.')
    if ext == 'pt':
        try:
            model = get_rt_from_pt(data)
        except Exception as e:
            st.error(f'Could not use {up.name}: {e}')
            return base, None, []
        if not model.is_calibrated:
            st.info(f'{up.name} is an RT model without calibration: showing relative RT from this model.')
        return model, None, []

    try:
        from calibrate_rt import guess_columns, calibrate_table  # noqa: F401  (checks the file is up to date)
    except ImportError as e:
        st.error('calibrate_rt.py next to this app is missing or an older version '
                 f'({e}). Replace it with the latest calibrate_rt.py and restart the app. '
                 'Uploading a calibrated .pt still works.')
        return base, None, []
    try:
        df = read_rt_table(data, up.name)
    except Exception as e:
        st.error(f'Could not read {up.name}: {e}')
        return base, None, []
    cols = [str(c) for c in df.columns]
    g_pep, g_rt = guess_columns(df)
    pep_col = st.selectbox('Peptide column', cols, index=cols.index(str(g_pep)) if g_pep is not None else 0,
                           key=f'rt_pep_col_{up.name}_{len(data)}')
    rt_col = st.selectbox('RT column', cols, index=cols.index(str(g_rt)) if g_rt is not None else min(1, len(cols) - 1),
                          key=f'rt_rt_col_{up.name}_{len(data)}')
    unit = st.selectbox('RT unit in your table', ['min', 's', 'iRT'], key='rt_unit')
    method = st.text_input('LC method name', os.path.splitext(up.name)[0], key='rt_method')
    if g_pep is None or g_rt is None:
        st.caption('Could not recognise the column names: check the peptide and RT columns above.')
    try:
        model, table, msgs = get_rt_from_table(data, up.name, base_path, unit, method,
                                               df.columns[cols.index(pep_col)], df.columns[cols.index(rt_col)])
    except Exception as e:  # CalibrationError or unreadable file
        st.error(f'Calibration failed: {e}')
        return base, None, []
    st.download_button('Download calibrated RT model (.pt)', model.to_bytes(),
                       file_name=f"rt_DIA_{''.join(c if c.isalnum() else '_' for c in method)}.pt",
                       help='Upload this file next time instead of the table; it holds the same calibration.')
    return model, table, msgs


def rt_unit(info):
    return info.get('unit') or 'RT units'


def rt_metric(rt_model, seq):
    """Returns (label, value, help, notes) for the RT metric of one peptide."""
    from rt_model import check_peptide
    calibrated = rt_model.is_calibrated
    label = 'Predicted RT' if calibrated else 'Relative RT'
    problem = check_peptide(seq)
    if problem:
        return label, 'n/a', f'No RT prediction: the RT model only handles unmodified peptides of up to 19 residues ({problem}).', []
    raw = float(rt_model.predict_raw([seq])[0])
    if not calibrated:
        return (label, f'{raw:.3f}',
                'Relative retention on a 0-1 scale (early to late, scale of the DIA training data). '
                'Upload an RT calibration in the sidebar to get the RT of your LC method.', [])
    info = rt_model.calibration_info
    u = rt_unit(info)
    rt = float(rt_model.predict([seq])[0])
    helptext = f"Calibrated on {info.get('n_used', '?')} peptides ({info.get('name') or 'uploaded calibration'})."
    if 'cv_abs_error_95pct' in info:
        helptext += (f" Typical error {info['cv_median_abs_error']:.2f} {u}; 95% of peptides within "
                     f"±{info['cv_abs_error_95pct']:.2f} {u} (cross-validated).")
    notes = []
    lo, hi = info.get('model_output_range', (raw, raw))
    if raw < lo or raw > hi:
        r0, r1 = info.get('rt_range', (float('nan'), float('nan')))
        notes.append(f'This peptide elutes outside the RT range of the calibration peptides ({r0:.1f}-{r1:.1f} {u}); '
                     f'the predicted RT is an extrapolation, treat it as rough.')
    return label, f'{rt:.2f} {u}' if u != 'RT units' else f'{rt:.2f}', helptext, notes


def rt_calibration_details(rt_model, table, msgs, dark=False):
    info = rt_model.calibration_info
    u = rt_unit(info)
    with st.expander(f"RT calibration: {info.get('name') or 'uploaded'}", expanded=False):
        line = (f"{info.get('n_peptides', '?')} peptides, {info.get('n_used', '?')} used "
                f"(outliers removed), RT range {info['rt_range'][0]:.1f}-{info['rt_range'][1]:.1f} {u}, "
                f"r = {info.get('pearson_r', float('nan')):.3f}")
        if 'cv_median_abs_error' in info:
            line += (f"; typical error {info['cv_median_abs_error']:.2f} {u}, 95% within "
                     f"±{info['cv_abs_error_95pct']:.2f} {u} (cross-validated)")
        st.write(line + '.')
        for m in msgs:
            st.caption(m)
        if table is None:
            return
        col = C_DARK if dark else C_LIGHT
        used = table[table['used_in_fit']]
        out = table[~table['used_in_fit']]
        pred = 'cv_residual' in table
        y_pred = lambda d: d['measured_rt'] + (d['cv_residual'] if pred else d['residual'])
        fig = go.Figure()
        fig.add_scatter(x=used['measured_rt'], y=y_pred(used), mode='markers', name='used',
                        marker=dict(size=5, color=col['b'], opacity=0.5), text=used['peptide'],
                        hovertemplate='%{text}<br>measured %{x:.2f}<br>predicted %{y:.2f}<extra></extra>')
        if len(out):
            fig.add_scatter(x=out['measured_rt'], y=y_pred(out), mode='markers', name='outlier (removed)',
                            marker=dict(size=7, color=col['y'], symbol='x'), text=out['peptide'],
                            hovertemplate='%{text}<br>measured %{x:.2f}<br>predicted %{y:.2f}<extra></extra>')
        lo, hi = float(table['measured_rt'].min()), float(table['measured_rt'].max())
        fig.add_scatter(x=[lo, hi], y=[lo, hi], mode='lines', name='1:1',
                        line=dict(color='rgba(128,128,128,0.7)', width=1, dash='dash'), hoverinfo='skip')
        fig.update_layout(height=380, xaxis_title=f'measured RT ({u})',
                          yaxis_title=f"predicted RT ({u})" + (' — cross-validated' if pred else ''),
                          legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
                          margin=dict(l=10, r=10, t=40, b=10))
        show_plot(fig)
        st.download_button('Download calibration table (CSV)', table.to_csv(index=False).encode('utf-8'),
                           file_name='rt_calibration_peptides.csv', mime='text/csv')


# ---------------------------------------------------------------- app
def main():
    st.set_page_config(page_title='Peptide MS/MS predictor', layout='wide')
    st.title('Peptide MS/MS spectrum predictor')
    st.caption('Predicts the relative intensities of the y and b ions of one peptide with the trained network, '
               'placed at their theoretical m/z.')

    try:
        here = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        here = os.getcwd()
    with st.sidebar:
        st.header('Settings')
        ckpt_path = st.text_input('Model file', os.path.join(here, 'peptide_predictor.pt'), key='ckpt')
        cys = st.checkbox('Cysteine carbamidomethylated (+57.02146)', value=False, key='cys')
        n_labels = st.slider('Labelled peaks (peptides with more than 16 ions)', 0, 30, 12, key='labels')
        cutoff = st.slider('Hide peaks below (% of base peak)', 0.0, 20.0, 0.0, 0.5, key='cutoff')

    try:
        model, meta = get_model(ckpt_path)
    except FileNotFoundError:
        st.error(f'Model file not found: {ckpt_path}\n\nRun the export cell at the end of the notebook, '
                 f'then put peptide_predictor.pt next to this app (or fix the path in the sidebar).')
        st.stop()
    except ImportError as e:
        st.error(f'Could not import the model code ({e}). Install PyTorch in this environment and keep '
                 f'peptide_model.py next to peptide_app.py.')
        st.stop()
    except Exception as e:  # corrupt / incompatible checkpoint
        st.error(f'Could not load the model file: {e}')
        st.stop()
    from peptide_model import predict_intensities
    zs = meta.get('charges') or []
    if zs:
        with st.sidebar:
            pz = st.radio('Precursor charge (model input)', zs, index=zs.index(2) if 2 in zs else 0,
                          format_func=lambda z: f'{z}+', horizontal=True, key='pz')
    else:
        pz = None
    with st.sidebar:                                               # a 1+ precursor can only give 1+ fragments
        charge = st.radio('Fragment charge (m/z axis only)', [1] if pz == 1 else [1, 2], format_func=lambda z: f'{z}+',
                          horizontal=True, key='charge')
    es = meta.get('energies') or []
    if es:
        default = 28.0 if min(es) <= 28 <= max(es) else float(es[len(es) // 2])
        with st.sidebar:
            pe = st.slider('Collision energy (HCD, NCE)', float(min(es)), float(max(es)), default, 1.0, format='%g', key='pe')
            st.caption('Trained at NCE ' + ', '.join(f'{e:g}' for e in es) + '. Values in between are interpolated. '
                       'NCE depends on the instrument: pick the value that matches your method.')
    else:
        pe = None
    with st.sidebar:
        rt_model, rt_table, rt_msgs = rt_sidebar(here)

    raw = st.text_input('Peptide sequence (one-letter codes)', value='TIFPSQNGHDGSTDVQQR', key='seq')
    vocab = set(meta['aa']) & set(RESIDUE)
    seq, err, warn = check_sequence(raw, vocab, meta['max_len_trained'])
    if not seq:
        st.info('Enter a peptide sequence to predict its spectrum.')
        st.stop()
    if err:
        st.error(err)
        st.stop()
    if warn:
        st.warning(warn)
    if len(seq) > meta['n_frag'] + 1:
        st.warning(f"Only y1-y{meta['n_frag']} and b1-b{meta['n_frag']} are predicted; this peptide has more fragment positions.")

    res = residue_masses(cys)
    intensities = predict_intensities(model, meta, seq, pz, pe)
    tab = ion_table(seq, intensities, meta['n_frag'], charge, res)
    mass = neutral_mass(seq, res)

    b_share = tab.loc[tab['type'] == 'b', 'intensity'].sum() / max(tab['intensity'].sum(), 1e-12)
    top = tab.loc[tab['intensity'].idxmax(), 'ion'] if tab['intensity'].max() > 0 else '-'
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric('Length', f'{len(seq)} residues')
    c2.metric('Mass [M] (Da)', f'{mass:.4f}')
    zp = int(pz) if pz else 2
    c3.metric(f'[M+{zp}H]{zp}+ m/z', f'{(mass + zp * PROTON) / zp:.4f}')
    c4.metric('Strongest ion', top)
    c5.metric('Predicted b / y share', f'{100 * b_share:.0f}% / {100 * (1 - b_share):.0f}%')
    if rt_model is not None:
        try:
            rt_label, rt_value, rt_help, rt_notes = rt_metric(rt_model, seq)
        except Exception as e:  # never let RT break the spectrum prediction
            rt_label, rt_value, rt_help, rt_notes = 'RT', 'error', f'RT prediction failed: {e}', []
        try:   # vertical_alignment needs Streamlit >= 1.36
            r1, r2 = st.columns([1, 4], vertical_alignment='bottom')
        except TypeError:
            r1, r2 = st.columns([1, 4])
        r1.metric(rt_label, rt_value)
        with r2:
            st.caption(rt_help)
            for note in rt_notes:
                st.caption(f'⚠ {note}')

    if not tab['intensity'].max() > 0:
        st.warning('The model predicts no fragment signal for this peptide.')
        st.stop()

    try:
        dark = st.context.theme.type == 'dark'
    except Exception:
        dark = False
    st.subheader('Predicted spectrum')
    show_plot(build_figure(tab, n_labels, cutoff, dark))
    st.subheader('Fragment map')
    st.caption('Every peptide bond with its b ion (above, blue) and complementary y ion (below, orange). '
               'Darker and thicker = stronger predicted ion.')
    show_plot(ladder_figure(seq, tab, dark))

    out = tab.rename(columns={'mz': 'm/z', 'rel': 'relative intensity (%)', 'intensity': 'predicted intensity'})
    out['m/z'] = out['m/z'].round(4)
    out['relative intensity (%)'] = out['relative intensity (%)'].round(2)
    out = out[['ion', 'type', 'm/z', 'relative intensity (%)', 'predicted intensity']]
    with st.expander('Ion table', expanded=False):
        st.dataframe(out, hide_index=True)
    st.download_button('Download table (CSV)', out.to_csv(index=False).encode('utf-8'),
                       file_name=f'{seq}_predicted_spectrum.csv', mime='text/csv')
    if zs:
        st.caption(f'Prediction for a {pz}+ precursor' + (f' at NCE {pe:g}' if es else '') + '. '
                   f'Precursor charge (trained on {", ".join(f"{z}+" for z in zs)})' + (' and collision energy are model inputs' if es else ' is a model input; collision energy is not')
                   + '. The fragment-charge and cysteine settings change the m/z values only.')
    else:
        st.caption('This model sees only the sequence: precursor charge and collision energy are not inputs. '
                   'The charge and cysteine settings change the m/z values only, not the predicted intensities.')
    if rt_model is not None and rt_model.is_calibrated:
        rt_calibration_details(rt_model, rt_table, rt_msgs, dark)


if __name__ == '__main__':
    main()
