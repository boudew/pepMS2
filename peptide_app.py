"""Peptide MS/MS spectrum predictor - local web app.

Start:  streamlit run peptide_app.py
Needs:  pip install streamlit plotly pandas numpy torch
Same folder: peptide_model.py and peptide_predictor.pt (written by the notebook's export cell).
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
def build_figure(tab, n_labels=8, cutoff=0.0, dark=False):
    """Interactive stick spectrum: b ions blue, y ions orange (colour-blind-safe pair, light/dark steps)."""
    c_b, c_y = ('#3987e5', '#d95926') if dark else ('#2a78d6', '#eb6834')
    show = tab[tab['rel'] >= cutoff]
    labelled = set(show.nlargest(n_labels, 'rel').index) if n_labels > 0 else set()
    span = max(tab['mz'].max() - tab['mz'].min(), 1.0)
    fig = go.Figure()
    for t, name, colour in (('b', 'b ions', c_b), ('y', 'y ions', c_y)):
        d = show[show['type'] == t]
        fig.add_bar(
            x=d['mz'], y=d['rel'], name=name, marker_color=colour, width=span * 0.004,
            text=[ion if i in labelled else '' for i, ion in zip(d.index, d['ion'])],
            textposition='outside', textangle=-90, cliponaxis=False, hoverinfo='skip')
        # the sticks are only a few pixels wide, so hover is carried by invisible markers instead:
        # with hovermode='x' the nearest stick (by x only) shows its tooltip anywhere in the column
        fig.add_scatter(
            x=d['mz'], y=d['rel'], mode='markers', marker=dict(size=16, color='rgba(0,0,0,0)'),
            showlegend=False, customdata=d['ion'],
            hovertemplate='%{customdata}<br>m/z %{x:.4f}<br>%{y:.1f}% of base peak<extra></extra>')
    fig.update_layout(
        barmode='overlay', height=430, hovermode='x', hoverdistance=25,
        xaxis_title='m/z', yaxis_title='relative intensity (%)',
        xaxis_range=[tab['mz'].min() - span * 0.04, tab['mz'].max() + span * 0.04],
        yaxis_range=[0, 128],
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        margin=dict(l=10, r=10, t=40, b=10))
    fig.update_xaxes(showgrid=False)
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
        charge = st.radio('Fragment charge (m/z axis only)', [1, 2], format_func=lambda z: f'{z}+',
                          horizontal=True, key='charge')
        cys = st.checkbox('Cysteine carbamidomethylated (+57.02146)', value=False, key='cys')
        n_labels = st.slider('Labelled peaks', 0, 20, 8, key='labels')
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

    res = residue_masses(cys)
    intensities = predict_intensities(model, meta, seq, pz)
    tab = ion_table(seq, intensities, meta['n_frag'], charge, res)
    mass = neutral_mass(seq, res)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric('Length', f'{len(seq)} residues')
    c2.metric('Monoisotopic mass [M]', f'{mass:.4f} Da')
    c3.metric('[M+2H]2+ m/z', f'{(mass + 2 * PROTON) / 2:.4f}')
    c4.metric('[M+3H]3+ m/z', f'{(mass + 3 * PROTON) / 3:.4f}')

    if not tab['intensity'].max() > 0:
        st.warning('The model predicts no fragment signal for this peptide.')
        st.stop()

    try:
        dark = st.context.theme.type == 'dark'
    except Exception:
        dark = False
    show_plot(build_figure(tab, n_labels, cutoff, dark))

    out = tab.rename(columns={'mz': 'm/z', 'rel': 'relative intensity (%)', 'intensity': 'predicted intensity'})
    out['m/z'] = out['m/z'].round(4)
    out['relative intensity (%)'] = out['relative intensity (%)'].round(2)
    out = out[['ion', 'type', 'm/z', 'relative intensity (%)', 'predicted intensity']]
    st.dataframe(out, hide_index=True)
    st.download_button('Download table (CSV)', out.to_csv(index=False).encode('utf-8'),
                       file_name=f'{seq}_predicted_spectrum.csv', mime='text/csv')
    if zs:
        st.caption(f'Prediction for a {pz}+ precursor. The precursor charge is a model input (trained on {", ".join(f"{z}+" for z in zs)}); '
                   'collision energy is not. The fragment-charge and cysteine settings change the m/z values only.')
    else:
        st.caption('This model sees only the sequence: precursor charge and collision energy are not inputs. '
                   'The charge and cysteine settings change the m/z values only, not the predicted intensities.')


if __name__ == '__main__':
    main()
