"""Model definition and prediction helper used by peptide_app.py.

Needs: torch, numpy.
The architecture below must stay identical to the SpecNet cell in peptide_msms_nn.ipynb.
"""
import numpy as np
import torch
import torch.nn as nn


class SpecNet(nn.Module):
    """Single BiLSTM layer, then 3 dense layers shared by every cleavage site.
    Site j lies between residue j and j+1; it gives b_j and (complementary) y_(L-j)."""

    def __init__(self, n_aa, n_frag, hid=64, drop=0.2, n_charge=0, heads=0):
        super().__init__()
        self.n_frag, self.hid, self.n_charge = n_frag, hid, n_charge
        self.rnn = nn.LSTM(n_aa, hid, num_layers=1, batch_first=True, bidirectional=True)
        if heads:                                                   # optional self-attention over the residues (0 = off)
            self.attn = nn.MultiheadAttention(2 * hid, heads, dropout=drop, batch_first=True)
            self.norm = nn.LayerNorm(2 * hid)
        self.heads = heads
        self.site = nn.Sequential(
            nn.Linear(2 * hid + n_charge, 128), nn.ReLU(), nn.Dropout(drop),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(drop),
            nn.Linear(64, 2))

    def forward(self, x, z=None):
        """x: (B, T, n_aa) one-hot matrix. z: (B,) precursor-charge index (0..n_charge-1), needed if n_charge > 0."""
        x = x.float()                                              # (B, T, n_aa) one-hot matrix
        B, T, _ = x.shape
        pad = x.sum(-1) == 0                                       # True at padded positions
        L = (~pad).sum(1)                                          # true peptide lengths
        h, _ = self.rnn(x)                                         # (B, T, 2*hid)
        if self.heads:                                             # every residue looks at all others
            a, _ = self.attn(h, h, h, key_padding_mask=pad, need_weights=False)
            h = self.norm(h + a)
        feat = torch.cat([h[:, :-1, :self.hid], h[:, 1:, self.hid:]], dim=-1)   # (B, T-1, 2*hid)
        if self.n_charge:
            zc = nn.functional.one_hot(z.long(), self.n_charge).to(feat.dtype)           # (B, n_charge)
            feat = torch.cat([feat, zc[:, None, :].expand(-1, feat.shape[1], -1)], dim=-1)   # charge seen at every site
        s = self.site(feat)                                        # (B, T-1, 2): [...,0] y-type, [...,1] b-type
        k = torch.arange(1, self.n_frag + 1, device=x.device)
        b = s[:, :self.n_frag, 1]                                  # b_k  <- site k-1 (0-based)
        yi = (L[:, None] - 1 - k[None, :]).clamp(0, T - 2)         # y_k  <- site L-k-1 (0-based)
        y = s[..., 0].gather(1, yi)
        return torch.sigmoid(torch.cat([y, b], dim=1))             # y1..yN, b1..bN in [0, 1]


def load_checkpoint(path):
    """Load the file written by the notebook's export cell. Returns (model, meta)."""
    ckpt = torch.load(path, map_location='cpu', weights_only=True)
    charges = list(ckpt.get('charges', []))                        # precursor charges the model was trained on ([] = none)
    model = SpecNet(len(ckpt['aa']), ckpt['n_frag'], ckpt['hid'], ckpt['drop'], n_charge=len(charges),
                    heads=ckpt.get('heads', 0))
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    meta = {k: ckpt[k] for k in ('aa', 'n_frag', 'maxlen', 'max_len_trained')}
    meta['charges'] = charges
    return model, meta


def _charge_index(meta, charges, n):
    """Map precursor charges to the model's charge indices (None if the model has no charge input)."""
    if not meta.get('charges'):
        return None
    if charges is None:
        raise ValueError(f"This model needs the precursor charge (trained on {meta['charges']}).")
    c2i = {c: i for i, c in enumerate(meta['charges'])}
    charges = np.broadcast_to(np.asarray(charges), (n,))
    bad = sorted({int(c) for c in charges if int(c) not in c2i})
    if bad:
        raise ValueError(f"Precursor charge {bad} was not in the training data {meta['charges']}.")
    return np.array([c2i[int(c)] for c in charges], dtype=np.int64)


def predict_intensities(model, meta, seq, charge=None):
    """Predicted relative intensities for one peptide, ordered y1..yN then b1..bN (N = n_frag).
    Fragments that cannot exist for this peptide length are 0."""
    aa2i = {a: i for i, a in enumerate(meta['aa'])}
    T = max(len(seq), meta['maxlen'])                              # same padding as during training
    x = np.zeros((1, T, len(aa2i)), dtype=np.float32)
    x[0, np.arange(len(seq)), [aa2i[c] for c in seq]] = 1
    zi = _charge_index(meta, charge, 1)
    with torch.no_grad():
        p = model(torch.from_numpy(x), None if zi is None else torch.from_numpy(zi)).numpy()[0]
    k = np.arange(1, meta['n_frag'] + 1)
    valid = (k <= len(seq) - 1).astype(np.float32)
    return p * np.concatenate([valid, valid])


def predict_batch(model, meta, seqs, charges=None, device=None, batch_size=4096):
    """Predicted intensities for many peptides at once (every residue must be in meta['aa']).

    Returns an array (N, 2*n_frag), ordered y1..yN then b1..bN; fragments that cannot exist are 0.
    Each peptide is padded exactly as in predict_intensities(), so the results are identical to
    predicting the peptides one at a time - just much faster (GPU if available).
    charges: precursor charge per peptide (one number or a list); required if the model has a charge input."""
    device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    model = model.to(device).eval()
    aa2i = {a: i for i, a in enumerate(meta['aa'])}
    n_frag, n_aa = meta['n_frag'], len(aa2i)
    lens = np.array([len(s) for s in seqs])
    zi = _charge_index(meta, charges, len(seqs))
    pad = np.maximum(lens, meta['maxlen'])                         # padded length per peptide
    out = np.zeros((len(seqs), 2 * n_frag), dtype=np.float32)
    k = np.arange(1, n_frag + 1)[None, :]
    for T in np.unique(pad):
        idx = np.where(pad == T)[0]
        for start in range(0, len(idx), batch_size):
            sel = idx[start:start + batch_size]
            x = np.zeros((len(sel), T, n_aa), dtype=np.float32)
            for row, i in enumerate(sel):
                x[row, np.arange(lens[i]), [aa2i[c] for c in seqs[i]]] = 1
            with torch.no_grad():
                zb = None if zi is None else torch.from_numpy(zi[sel]).to(device)
                p = model(torch.from_numpy(x).to(device), zb).cpu().numpy()
            valid = (k <= lens[sel][:, None] - 1).astype(np.float32)
            out[sel] = p * np.concatenate([valid, valid], axis=1)
    return out
