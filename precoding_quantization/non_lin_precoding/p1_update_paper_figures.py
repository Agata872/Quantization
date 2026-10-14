"""Write the independent-test-set results (p1_independent_eval.py) into the paper's simulation figures.

Only numbers are replaced; styles, layout, legends and comments other than the data-source note are kept.
Every replaced data block is first identified by matching its OLD numbers against the paper's old data on
Htest[4096:6144] (plot_continue_figures.load(), ber_16qam_20dB.json, phase_offset_eval.json); a block that matches no
old series aborts the script. Exception: the GNN-GD column of Rsum_sweep_vs_bits.tex, whose old values had been
edited by hand in paper commit 8837c73 (2026-10-13) and match no stored result; it is replaced by position and the
old values are logged.

Figures (paper repo Figure/): rate_vs_snrK1.tex (Fig. 3), Rsum_sweep_vs_snr.tex (Fig. 4), Rsum_sweep_vs_bits.tex
(Fig. 5), ber_16qam_20dB.tex (Fig. 6), rsum_phase_offset.tex (Fig. 7).

  python p1_update_paper_figures.py [--dry-run]
"""
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
PAPER = '/home/user/Quantized-aware-training-and-deploy-study/Figure'
NEW = os.path.join(HERE, 'exp_results', 'independent_test')
SNR = [float(x) for x in range(-30, 31, 5)]
DRY = '--dry-run' in sys.argv
LOG = []
SRC_NOTE = ('% Data: independent test set Htest[6144:8192] (2048 channels never used for training, checkpoint selection\n'
            '% or tuning; checkpoints selected on Htest[4096:6144]), written by p1_update_paper_figures.py from\n'
            '% exp_results/independent_test/ (p1_independent_eval.py), 2026-10-14.\n')


def old_curves():
    import plot_continue_figures as P
    return P.load()


def new_curves():
    d = {}
    for K in (1, 2, 4, 6):
        for b in (1, 2, 3):
            z = np.load(os.path.join(NEW, f'rates_K{K}b{b}.npz'))
            d[(K, b)] = {k: z[k].mean(1) for k in ('gnn', 'lin_q', 'lin_unq', 'wmmse_q', 'wmmse_unq')}
            d[(K, b)]['gd20'] = float(np.load(os.path.join(NEW, f'gd_K{K}b{b}.npz'))['gnn_gd_20'].mean())
    return d


OLD, NEWD = old_curves(), new_curves()
KEYMAP = {'gnn': 'gnn', 'lin_q': 'lin_q', 'lin_unq': 'lin_unq', 'wmmse_q': 'wmmse_q', 'wmmse_unq': 'wmmse_unq'}


def match_series(xs, ys, tol, Ks=(1, 2, 4, 6)):
    """(K, b, key) whose old curve reproduces ys at SNRs xs (to the printed precision)."""
    hits = []
    for (K, b), d in OLD.items():
        if K not in Ks:
            continue
        for key in KEYMAP:
            v = [d[key][SNR.index(float(x))] for x in xs]
            if all(abs(a - c) <= tol for a, c in zip(ys, v)):
                hits.append((K, b, key))
    vals = {tuple(np.round([NEWD[(K, b)][k][SNR.index(float(x))] for x in xs], 6)) for K, b, k in hits}
    if not hits or len(vals) > 1:
        raise RuntimeError(f'no unique old series for {list(zip(xs, ys))[:3]}...: {hits}')
    return hits[0]


def write(name, text):
    path = os.path.join(PAPER, name)
    if DRY:
        print(f'[dry-run] {name}: {len(text)} chars')
        return
    open(path, 'w').write(text)


# ------------------------------------------------------------------------------- Fig. 3: rate_vs_snrK1.tex
def fig3():
    t = open(os.path.join(PAPER, 'rate_vs_snrK1.tex')).read()
    m = re.search(r'\\pgfplotstableread\{\n(.*?)\n\}\\rateSnrData', t, re.S)
    rows = m.group(1).split('\n')
    head = rows[0].split()
    assert head == ['snr', 'gnnA', 'cdA', 'mrtA', 'gnnB', 'cdB', 'mrtB', 'gnnC', 'cdC', 'mrtC', 'reference'], head
    col = {'gnnA': (1, 'gnn'), 'mrtA': (1, 'lin_q'), 'gnnB': (2, 'gnn'), 'mrtB': (2, 'lin_q'),
           'gnnC': (3, 'gnn'), 'mrtC': (3, 'lin_q'), 'reference': (1, 'lin_unq')}
    out = ['snr gnnA mrtA gnnB mrtB gnnC mrtC reference']      # coordinate-descent columns (not plotted) dropped
    for r in rows[1:]:
        v = r.split()
        x = float(v[0])
        i = SNR.index(x)
        for c, (b, key) in col.items():
            assert abs(float(v[head.index(c)]) - OLD[(1, b)][key][i]) < 2e-6, (c, x)
        out.append(' '.join([v[0]] + [f'{NEWD[(1, b)][key][i]:.6f}' for c, (b, key) in col.items()]))
    t = t[:m.start(1)] + '\n'.join(out) + t[m.end(1):]
    t = re.sub(r'(% All curves: the SAME 2048 held-out channels \(Htest\[4096:6144\]\), 125 symbols each,)',
               '% All curves: the SAME 2048 independent test channels (Htest[6144:8192]), 125 symbols each,', t)
    t = SRC_NOTE + t
    LOG.append('Fig. 3: 7 columns x 13 SNRs replaced (old values verified to 2e-6); coordinate-descent columns dropped')
    write('rate_vs_snrK1.tex', t)


# ------------------------------------------------------------------------------- Fig. 4: Rsum_sweep_vs_snr.tex
def fig4():
    t = open(os.path.join(PAPER, 'Rsum_sweep_vs_snr.tex')).read()
    pat = re.compile(r'(\\addplot\[[^\]]*\] table \{%\n)((?:-?\d+ -?\d+\.\d+\n)+)(\};)')
    n = 0

    def rep(m):
        nonlocal n
        rows = [r.split() for r in m.group(2).strip().split('\n')]
        xs, ys = [float(a) for a, _ in rows], [float(b) for _, b in rows]
        K, b, key = match_series(xs, ys, 6e-5, Ks=(2, 4, 6))
        n += 1
        new = ''.join(f'{a} {NEWD[(K, b)][key][SNR.index(float(a))]:.4f}\n' for a, _ in rows)
        return m.group(1) + new + m.group(3)
    t = pat.sub(rep, t)
    t = SRC_NOTE + t
    LOG.append(f'Fig. 4: {n} curves replaced (each identified by its old values to 6e-5)')
    write('Rsum_sweep_vs_snr.tex', t)


# ------------------------------------------------------------------------------- Fig. 5: Rsum_sweep_vs_bits.tex
def fig5():
    t = open(os.path.join(PAPER, 'Rsum_sweep_vs_bits.tex')).read()
    i20 = SNR.index(20.0)
    for K, name in ((1, 'One'), (2, 'Two'), (4, 'Four'), (6, 'Six')):
        m = re.search(r'\\pgfplotstableread\{\n([^{}]*?)\n\}\\rsumBits' + name, t)
        rows = m.group(1).split('\n')
        head = rows[0].split()
        out = [rows[0]]
        for r in rows[1:]:
            v = r.split()
            b = int(v[0])
            o, nw = OLD[(K, b)], NEWD[(K, b)]
            new = {'b': v[0], 'gnn': nw['gnn'][i20], 'gd': nw['gd20'], 'linear': nw['lin_q'][i20],
                   'wmmse': nw['wmmse_q'][i20]}
            for c, key in (('gnn', 'gnn'), ('linear', 'lin_q'), ('wmmse', 'wmmse_q')):
                if c in head:
                    assert abs(float(v[head.index(c)]) - o[key][i20]) < 6e-5, (K, b, c)
            gd_old = float(v[head.index('gd')])
            LOG.append(f'Fig. 5 K={K} b={b}: GNN-GD in file {gd_old:.4f} (stored old-slice result {o["gd"][i20]:.4f}) '
                       f'-> test {nw["gd20"]:.4f}')
            out.append(' '.join([v[0]] + [f'{new[c]:.4f}' for c in head[1:]]))
        t = t[:m.start(1)] + '\n'.join(out) + t[m.end(1):]
    for K, x, key in ((1, '4.05', 'lin_unq'), (2, '4.05', 'lin_unq'), (2, '4.35', 'wmmse_unq'), (4, '4.05', 'lin_unq'),
                      (4, '4.35', 'wmmse_unq'), (6, '4.05', 'lin_unq'), (6, '4.35', 'wmmse_unq')):
        kname = {1: 'one', 2: 'two', 4: 'four', 6: 'six'}[K]
        style = 'unquantized wmmse' if key == 'wmmse_unq' else 'unquantized'
        p = re.compile(r'(\\addplot\[rsumK' + kname + r', rsum bits/' + style + r'\] coordinates \{\(' + re.escape(x)
                       + r',)(\d+\.\d+)(\)\};)')
        m = p.search(t)
        assert m and abs(float(m.group(2)) - OLD[(K, 1)][key][i20]) < 6e-5, (K, key, m and m.group(2))
        t = t[:m.start(2)] + f'{NEWD[(K, 1)][key][i20]:.4f}' + t[m.end(2):]
    t = t.replace('% GNN-GD: each continued GNN', '% GNN-GD (test set, 20 dB, p1_independent_eval.py --stage gd): each GNN')
    t = SRC_NOTE + t
    LOG.append('Fig. 5: 4 tables (GNN, GNN-GD, ZF/MRT + DAC, WMMSE + DAC) and 7 unquantized markers replaced')
    write('Rsum_sweep_vs_bits.tex', t)


# ------------------------------------------------------------------------------- Fig. 6: ber_16qam_20dB.tex
def fig6():
    t = open(os.path.join(PAPER, 'ber_16qam_20dB.tex')).read()
    old = json.load(open(os.path.join(HERE, 'exp_results', 'ber_16qam_20dB.json')))['results']
    new = {(K, b): np.load(os.path.join(NEW, f'ber_K{K}b{b}.npz')) for K in (1, 2, 4, 6) for b in (1, 2, 3)}
    meth = {'ideblk': 'ide_block', 'ide': 'ide_cal', 'squid': 'squid', 'gnn': 'gnn'}
    pat = re.compile(r'(\\addplot\[ber/(\w+), ber/b(\d)(?:, forget plot)?\] coordinates \{)([^}]*)(\};)')
    n = 0

    def rep(m):
        nonlocal n
        key, b = meth[m.group(2)], int(m.group(3))
        pts = re.findall(r'\((\d+),([0-9.e+-]+)\)', m.group(4))
        for K, v in pts:
            ref = old[f'K{K}b{b}'][key]
            assert abs(float(v) / ref - 1) < 1e-4, (key, b, K, v, ref)
        n += 1
        return m.group(1) + ' '.join(f'({K},{new[(int(K), b)][key].mean():.4e})' for K, _ in pts) + m.group(5)
    t = pat.sub(rep, t)
    t = SRC_NOTE + t
    LOG.append(f'Fig. 6: {n} BER curves replaced (old values verified to 1e-4 relative)')
    write('ber_16qam_20dB.tex', t)


# ------------------------------------------------------------------------------- Fig. 7: rsum_phase_offset.tex
def fig7():
    t = open(os.path.join(PAPER, 'rsum_phase_offset.tex')).read()
    old = json.load(open(os.path.join(HERE, 'exp_results', 'phase_offset_eval.json')))
    new = json.load(open(os.path.join(NEW, 'phase.json')))
    j20 = old['snr_db'].index(20.0)
    so, sn = old['sigma_deg'], new['sigma_deg']
    pat = re.compile(r'(% (\w+), b = (\d)\n\\addplot\[[^\n]*\{%\n)((?:[0-9.]+ [0-9.]+ [0-9.]+\n)+)(\};)')
    ref_pat = re.compile(r'(\\addplot\[fill=white, draw=poRef, ref hatch, po/bar\][^\n]*\{%\n)((?:[0-9.]+ [0-9.]+ [0-9.]+\n)+)(\};)')
    n = 0

    def rows_new(rows, method, key, offs):
        out = ''
        for (x, y, e), s in zip(rows, offs):
            om, oe = old['results'][key]['methods'][method]['mean'][so.index(s)][j20], \
                old['results'][key]['methods'][method]['std_offset'][so.index(s)][j20]
            assert abs(float(y) - om) < 6e-5 and abs(float(e) - oe) < 6e-5, (method, key, s, y, om, e, oe)
            nm = new['results'][key]['methods'][method]
            out += f"{x} {nm['mean'][sn.index(s)]:.4f} {nm['std_offset'][sn.index(s)]:.4f}\n"
        return out

    def rep(m):
        nonlocal n
        method, b = m.group(2), int(m.group(3))
        rows = [r.split() for r in m.group(4).strip().split('\n')]
        n += 1
        return m.group(1) + rows_new(rows, method, f'K2b{b}', (0.0, 10.0, 20.0, 30.0)) + m.group(5)
    t = pat.sub(rep, t)

    def rep_ref(m):
        nonlocal n
        rows = [r.split() for r in m.group(2).strip().split('\n')]
        n += 1
        return m.group(1) + rows_new(rows, 'wmmse_unq', 'K2b1', (0.0, 10.0, 20.0, 30.0)) + m.group(3)
    t = ref_pat.sub(rep_ref, t)
    assert n == 7, n
    t = t.replace('% Data: exp_results/phase_offset_eval.json (phase_offset_eval.py): 2048 held-out channels (Htest[4096:6144]),',
                  '% Data: exp_results/independent_test/phase.json (p1_independent_eval.py --stage phase): 2048 test channels\n'
                  '% (Htest[6144:8192]),')
    LOG.append(f'Fig. 7: {n} bar series replaced (means and offset std verified to 6e-5)')
    write('rsum_phase_offset.tex', t)


if __name__ == '__main__':
    for f in (fig3, fig4, fig5, fig6, fig7):
        f()
    print('\n'.join(LOG))
