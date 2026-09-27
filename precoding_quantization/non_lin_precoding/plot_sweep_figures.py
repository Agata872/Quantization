"""Two paper figures for the K x bits sweep, as standalone pgfplots files plus PNG previews.

Fig. A  rate vs SNR   -- one panel per K; GNN against GNN-GD (the GNN output refined per channel
                         by gradient descent, gd_refine_sweep.py) and the quantized ZF baseline.
                         Colour encodes the method, the same in both figures; DAC resolution b is
                         the marker shape (circle / square / triangle). Filled markers: the GNN;
                         open markers: references. Line style separates the methods in grayscale.
Fig. B  rate vs bits  -- one panel per K at the training SNR; GNN, GNN-GD, the coordinate-descent
                         reference and quantized ZF at b = 1, 2, 3, with the ZF curve continued
                         to b = infinity, i.e. unquantized ZF (the style of Feys et al., JSTSP
                         2025, Fig. 11). The ceiling is thus the natural end point of the
                         baseline rather than a free-floating line; the 3 -> inf segment is
                         dotted because nothing is measured in between.

Four panels (K = 1, 2, 4, 6) are laid out 2 x 2, three or fewer in one row. Panels share the x-axis
but NOT the y-axis: sum rates at K=6 are ~5x those at K=1, and the comparison the reader makes is
within a panel (GNN vs references at equal b). For K=1, ZF (h*/||h||^2) is MRT after power
normalization.

Colours: categorical slots 1-3 of the dataviz skill palette, checked with its validate_palette.py
against a white surface: #2a78d6,#eb6834,#1baf7a --pairs all -> all PASS (worst CVD dE 9.2);
contrast WARN for #1baf7a (2.74:1), relieved by markers, the legend and the solid-vs-dashed split
from the gray ZF curves (aqua and #898781 are close under deuteranopia, dE 5.1).
Grays (#898781 muted, #52514e secondary) carry the de-emphasised references.

Input JSON:
  {"snr_db": [...], "snr_ref_db": 20, "panels_K": [...], "panels_bits": [...],
   "configs": [{"K": 2, "bits": 1, "gnn": [...], "zf_q": [...], "zf_unq": [...],
                "cd_ref": 5.3, "gnn_gd": [...], "recipe": "train_sweep.py"}, ...]}
  gnn / zf_q / zf_unq / gnn_gd are aligned with snr_db; cd_ref is the rate at snr_ref_db.
  gnn_gd and recipe are optional.

  python plot_sweep_figures.py results.json OUT_DIR [--xmin -10] [--preview-note TEXT]
"""
import argparse
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

GNN_ONE = ('gnnblue', '2a78d6')
CD_COL = ('cdorange', 'eb6834')
GD_COL = ('gdaqua', '1baf7a')
ZF_GRAY = ('zfgray', '898781')
UNQ_GRAY = ('unqgray', '52514e')
GRID = ('gridgray', 'e6e5e1')
INF = 4.2          # x position of the b = infinity (unquantized) point
MARK_FILLED = {1: '*', 2: 'square*', 3: 'triangle*'}
MARK_OPEN = {1: 'o', 2: 'square', 3: 'triangle'}
MPL_MARK = {1: 'o', 2: 's', 3: '^'}
BITS_TXT = {1: '1 bit', 2: '2 bits', 3: '3 bits'}
GD_LABEL = 'GNN-GD'

# per method: pgfplots style (with %s for the marker in fig A) and matplotlib kwargs
STY_GNN = r'gnnblue, line width=1.0pt, mark=%s, mark size=1.9pt, mark options={solid}'
STY_GD = r'gdaqua, line width=1.0pt, mark=%s, mark size=1.9pt, mark options={solid, fill=white}'
STY_ZF = r'zfgray, line width=0.8pt, dashed, mark=%s, mark size=1.9pt, mark options={solid, fill=white}'
STY_CD = r'cdorange, line width=0.8pt, dash dot, mark=diamond, mark size=2.3pt, mark options={solid, fill=white}'
MPL_GNN = dict(color=GNN_ONE[1], lw=1.6, ms=5)
MPL_GD = dict(color=GD_COL[1], lw=1.6, ms=5, mfc='white')
MPL_ZF = dict(color=ZF_GRAY[1], lw=1.2, ls='--', ms=5, mfc='white')
MPL_CD = dict(color=CD_COL[1], lw=1.2, ls='-.', marker='D', ms=5, mfc='white')


def _rgb(h):
    return tuple(int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))


def _mpl(kw):
    return {k: (_rgb(v) if k in ('color', 'mfc') and v != 'white' else v) for k, v in kw.items()}


def _colordefs(*cols):
    seen, out = set(), []
    for name, hx in cols:
        if name not in seen:
            seen.add(name)
            out.append(r'\definecolor{%s}{HTML}{%s}' % (name, hx.upper()))
    return '\n'.join(out)


def _table(xs, ys, xmin=None):
    rows = [f'{x:g} {y:.4f}' for x, y in zip(xs, ys) if xmin is None or x >= xmin]
    return 'table {%\n' + '\n'.join(rows) + '\n};'


def _header(comment_lines, colors):
    return '\n'.join(['% ' + c for c in comment_lines] + [
        r'\documentclass[tikz,border=4pt]{standalone}',
        r'\usepackage{pgfplots}',
        r'\pgfplotsset{compat=1.18}',
        r'\usepgfplotslibrary{groupplots}',
        r'\usetikzlibrary{calc}',
        r'\begin{document}',
        _colordefs(*colors)])


def _layout(n):
    """Columns and the groupplot options for n panels: 2 x 2 for four, else one row."""
    cols = 2 if n == 4 else n
    rows = -(-n // cols)
    gs = (f'group style={{group name=grp, group size={cols} by {rows}, horizontal sep=1.25cm, '
          f'vertical sep=1.1cm{", x descriptions at=edge bottom" if rows > 1 else ""}}},'
          '\n  width=5.6cm, height=4.9cm, tick align=outside, tick pos=left,'
          '\n  grid=major, grid style={line width=0.3pt, draw=gridgray},'
          '\n  every axis plot/.append style={line join=round},'
          r'  title style={font=\small, yshift=-3pt},'
          '\n  tick label style={font=\\footnotesize}, label style={font=\\small},')
    return cols, gs


def _legend_node(cols, name):
    return (r'\node[anchor=south] at ($(grp c1r1.north)!0.5!(grp c%dr1.north)+(0,0.55cm)$) '
            r'{\pgfplotslegendfromname{%s}};' % (cols, name))


def _recipe_notes(data):
    return [f'K={c["K"]}, b={c["bits"]}: from {c.get("run", "?")} ({c["recipe"]}; every other (K, b) '
            'is a train_sweep.py run)' for c in data['configs'] if c.get('recipe', 'train_sweep.py') != 'train_sweep.py']


def _gd_note(data):
    m = data.get('gnn_gd_method')
    if not m:
        return []
    return [f'{GD_LABEL}: {m["what"]};', f'  recipe {m["recipe"]}, config {m["config"]}.']


def fig_rate_vs_snr(data, out_tex, xmin, note):
    snr = data['snr_db']
    Ks = data.get('panels_K') or sorted({c['K'] for c in data['configs']})
    bits = data.get('panels_bits') or sorted({c['bits'] for c in data['configs']})
    get = {(c['K'], c['bits']): c for c in data['configs']}
    has_gd = any('gnn_gd' in c for c in data['configs'])
    cols, gs = _layout(len(Ks))
    L = [_header([
        'Sum rate vs SNR for every (K, b) of the sweep. One panel per K, y-axes not shared.',
        f'Blue solid + filled marker: GNN.  Aqua solid + open marker: {GD_LABEL}.  '
        'Gray dashed + open marker: ZF + b-bit DAC (MRT for K=1).',
        'b = 1/2/3 -> circle/square/triangle.'] + _gd_note(data) + _recipe_notes(data) + [
        'Rates: ' + data.get('estimator', 'unbiased least-squares Bussgang estimator (see eq:tp_G_hat)')] + (
        ['', note] if note else []),
        [GNN_ONE, GD_COL, ZF_GRAY, UNQ_GRAY, GRID])]
    L += [r'\begin{tikzpicture}', r'\begin{groupplot}[', '  ' + gs,
          f'  xmin={xmin - 2}, xmax={max(snr) + 2}, xtick={{{",".join(str(int(s)) for s in snr if s >= xmin and s % 10 == 0)}}},',
          r'  xlabel={SNR [dB]},', ']']
    for i, K in enumerate(Ks):
        opts = [f'title={{$K={K}$}}']
        if i % cols == 0:
            opts.append(r'ylabel={$R_{\mathrm{sum}}$ [bits/channel use]}')
        if i == 0:
            opts += ['legend to name=legsnr', 'legend columns=3', 'legend cell align=left',
                     r'legend style={draw=none, font=\footnotesize, '
                     r'/tikz/every even column/.append style={column sep=0.35cm}}']
        if not any((K, b) in get for b in bits):
            opts += ['ymin=0', 'ymax=10']
        L.append(r'\nextgroupplot[' + ', '.join(opts) + ']')
        if i == 0:   # legend independent of which curves exist, so an empty first panel still has one
            rows = [(STY_GNN, MARK_FILLED, 'GNN')] + ([(STY_GD, MARK_OPEN, GD_LABEL)] if has_gd else []) + [
                (STY_ZF, MARK_OPEN, 'ZF + DAC')]
            for sty, mk, lab in rows:
                for b in bits:
                    L.append(r'\addlegendimage{%s}' % (sty % mk[b]))
                    L.append(r'\addlegendentry{%s, %s}' % (lab, BITS_TXT[b]))
        for b in bits:   # references first, so the GNN is drawn on top
            if (K, b) in get:
                L.append(r'\addplot[%s, forget plot] %s' % (STY_ZF % MARK_OPEN[b], _table(snr, get[(K, b)]['zf_q'], xmin)))
        for b in bits:
            if (K, b) in get and 'gnn_gd' in get[(K, b)]:
                L.append(r'\addplot[%s, forget plot] %s' % (STY_GD % MARK_OPEN[b], _table(snr, get[(K, b)]['gnn_gd'], xmin)))
        for b in bits:
            if (K, b) in get:
                L.append(r'\addplot[%s, forget plot] %s' % (STY_GNN % MARK_FILLED[b], _table(snr, get[(K, b)]['gnn'], xmin)))
        missing = [b for b in bits if (K, b) not in get]
        if missing:
            L.append(r'\node[font=\scriptsize, text=unqgray, anchor=north west, align=left] at '
                     r'(rel axis cs:0.03,0.97) {pending: $b=%s$};' % ','.join(map(str, missing)))
    L += [r'\end{groupplot}', _legend_node(cols, 'legsnr'), r'\end{tikzpicture}', r'\end{document}', '']
    open(out_tex, 'w').write('\n'.join(L))


def fig_rate_vs_bits(data, out_tex, note):
    ref = data.get('snr_ref_db', 20)
    idx = data['snr_db'].index(ref)
    Ks = data.get('panels_K') or sorted({c['K'] for c in data['configs']})
    bits = data.get('panels_bits') or sorted({c['bits'] for c in data['configs']})
    get = {(c['K'], c['bits']): c for c in data['configs']}
    has_gd = any('gnn_gd' in c for c in data['configs'])
    cols, gs = _layout(len(Ks))
    L = [_header([
        f'Sum rate at SNR = {ref} dB vs DAC resolution, one panel per K, y-axes not shared.',
        f'GNN (blue, solid, filled) | {GD_LABEL} (aqua, solid, open) | coordinate-descent reference (orange, dash-dot) |',
        'ZF + b-bit DAC (gray, dashed; MRT for K=1), continued (dotted) to b = infinity = unquantized ZF.',
        'Coordinate descent is a local-search reference implemented for this work, not a',
        'published baseline; its target scale and number of sweeps are tuned at this SNR on a',
        'disjoint channel slice.'] + _gd_note(data) + _recipe_notes(data) + [
        'Rates: ' + data.get('estimator', 'unbiased least-squares Bussgang estimator (see eq:tp_G_hat)')] + (
        ['', note] if note else []),
        [GNN_ONE, GD_COL, CD_COL, ZF_GRAY, UNQ_GRAY, GRID])]
    L += [r'\begin{tikzpicture}', r'\begin{groupplot}[', '  ' + gs,
          f'  xmin={min(bits) - 0.3}, xmax={INF + 0.3}, xtick={{{",".join(map(str, bits))},{INF}}},',
          '  xticklabels={' + ','.join(map(str, bits)) + r',$\infty$},',
          r'  xlabel={DAC resolution $b$ [bits]}, ymin=0,', ']']
    for i, K in enumerate(Ks):
        opts = [f'title={{$K={K}$}}']
        if i % cols == 0:
            opts.append(r'ylabel={$R_{\mathrm{sum}}$ [bits/channel use]}')
        if i == 0:
            opts += ['legend to name=legbits', f'legend columns={4 if has_gd else 3}', 'legend cell align=left',
                     r'legend style={draw=none, font=\footnotesize, '
                     r'/tikz/every even column/.append style={column sep=0.35cm}}']
        have = [b for b in bits if (K, b) in get]
        if not have:
            opts += ['ymin=0', 'ymax=10']
        L.append(r'\nextgroupplot[' + ', '.join(opts) + ']')
        if i == 0:   # legend independent of which curves exist
            L += [r'\addlegendimage{%s}' % (STY_GNN % '*'), r'\addlegendentry{GNN}']
            if has_gd:
                L += [r'\addlegendimage{%s}' % (STY_GD % 'o'), r'\addlegendentry{%s}' % GD_LABEL]
            L += [r'\addlegendimage{%s}' % STY_CD, r'\addlegendentry{coordinate descent}',
                  r'\addlegendimage{%s}' % (STY_ZF % 'square'), r'\addlegendentry{ZF + $b$-bit DAC}']
        if have:
            g = [get[(K, b)]['gnn'][idx] for b in have]
            cd = [get[(K, b)]['cd_ref'] for b in have]
            zq = [get[(K, b)]['zf_q'][idx] for b in have]
            unq = get[(K, have[0])]['zf_unq'][idx]
            L.append(r'\addplot[%s, forget plot] %s' % (STY_ZF % 'square', _table(have, zq)))
            if max(have) == max(bits):   # only bridge to infinity from the last resolution
                L.append(r'\addplot[zfgray, line width=0.8pt, densely dotted, forget plot] '
                         r'coordinates {(%g,%.4f) (%g,%.4f)};' % (max(bits), zq[-1], INF, unq))
            L.append(r'\addplot[unqgray, only marks, mark=square*, mark size=1.9pt, forget plot] '
                     r'coordinates {(%g,%.4f)};' % (INF, unq))
            L.append(r'\addplot[%s, forget plot] %s' % (STY_CD, _table(have, cd)))
            gd_have = [b for b in have if 'gnn_gd' in get[(K, b)]]
            if gd_have:
                L.append(r'\addplot[%s, forget plot] %s'
                         % (STY_GD % 'o', _table(gd_have, [get[(K, b)]['gnn_gd'][idx] for b in gd_have])))
            L.append(r'\addplot[%s, forget plot] %s' % (STY_GNN % '*', _table(have, g)))
        missing = [b for b in bits if b not in have]
        if missing:
            L.append(r'\node[font=\scriptsize, text=unqgray, anchor=north west, align=left] at '
                     r'(rel axis cs:0.03,0.97) {pending: $b=%s$};' % ','.join(map(str, missing)))
    L += [r'\end{groupplot}', _legend_node(cols, 'legbits'), r'\end{tikzpicture}', r'\end{document}', '']
    open(out_tex, 'w').write('\n'.join(L))


def preview(data, out_png, xmin, note):
    """matplotlib twin of both figures, one row each, to eyeball the layout without a TeX install."""
    snr = data['snr_db']; ref = data.get('snr_ref_db', 20); idx = snr.index(ref)
    Ks = data.get('panels_K') or sorted({c['K'] for c in data['configs']})
    bits = data.get('panels_bits') or sorted({c['bits'] for c in data['configs']})
    get = {(c['K'], c['bits']): c for c in data['configs']}
    has_gd = any('gnn_gd' in c for c in data['configs'])
    xs = [s for s in snr if s >= xmin]; cut = snr.index(xs[0])
    fig, ax = plt.subplots(2, len(Ks), figsize=(3.5 * len(Ks), 6.8), squeeze=False)
    from matplotlib.lines import Line2D
    for i, K in enumerate(Ks):
        a = ax[0, i]; have = [b for b in bits if (K, b) in get]; missing = [b for b in bits if b not in have]
        for b in have:
            a.plot(xs, get[(K, b)]['zf_q'][cut:], marker=MPL_MARK[b], **_mpl(MPL_ZF))
            if 'gnn_gd' in get[(K, b)]:
                a.plot(xs, get[(K, b)]['gnn_gd'][cut:], marker=MPL_MARK[b], **_mpl(MPL_GD))
            a.plot(xs, get[(K, b)]['gnn'][cut:], marker=MPL_MARK[b], **_mpl(MPL_GNN))
        a.set_title(f'$K={K}$', fontsize=10); a.set_xlabel('SNR [dB]', fontsize=9); a.set_xticks([x for x in xs if x % 10 == 0])
        a.set_xlim(xmin - 2, max(snr) + 2); a.grid(color=_rgb(GRID[1]), lw=0.6)
        if not have: a.set_ylim(0, 10)
        if missing:
            a.text(0.03, 0.97, 'pending: $b=%s$' % ','.join(map(str, missing)), transform=a.transAxes,
                   va='top', fontsize=7.5, color=_rgb(UNQ_GRAY[1]))
        bx = ax[1, i]
        if have:
            zq = [get[(K, b)]['zf_q'][idx] for b in have]
            bx.plot(have, zq, marker='s', **_mpl(MPL_ZF))
            u = get[(K, have[0])]['zf_unq'][idx]
            if max(have) == max(bits):
                bx.plot([max(bits), INF], [zq[-1], u], color=_rgb(ZF_GRAY[1]), lw=1.2, ls=':')
            bx.plot([INF], [u], color=_rgb(UNQ_GRAY[1]), marker='s', ms=5, ls='none')
            bx.plot(have, [get[(K, b)]['cd_ref'] for b in have], **_mpl(MPL_CD))
            gd_have = [b for b in have if 'gnn_gd' in get[(K, b)]]
            if gd_have:
                bx.plot(gd_have, [get[(K, b)]['gnn_gd'][idx] for b in gd_have], marker='o', **_mpl(MPL_GD))
            bx.plot(have, [get[(K, b)]['gnn'][idx] for b in have], marker='o', **_mpl(MPL_GNN))
        else:
            bx.set_ylim(0, 10)
        if missing:
            bx.text(0.03, 0.97, 'pending: $b=%s$' % ','.join(map(str, missing)), transform=bx.transAxes,
                    va='top', fontsize=7.5, color=_rgb(UNQ_GRAY[1]))
        bx.set_xlim(min(bits) - 0.3, INF + 0.3); bx.set_ylim(bottom=0)
        bx.set_xticks(bits + [INF]); bx.set_xticklabels([str(b) for b in bits] + [r'$\infty$'])
        bx.set_title(f'$K={K}$', fontsize=10); bx.set_xlabel('DAC resolution $b$ [bits]', fontsize=9)
        bx.grid(color=_rgb(GRID[1]), lw=0.6)
    for r in range(2): ax[r, 0].set_ylabel('$R_{sum}$ [bits/channel use]', fontsize=9)
    rowsA = [(MPL_GNN, 'GNN')] + ([(MPL_GD, GD_LABEL)] if has_gd else []) + [(MPL_ZF, 'ZF + DAC')]
    hA = [Line2D([], [], marker=MPL_MARK[b], **_mpl(kw)) for kw, _ in rowsA for b in bits]
    lA = [f'{lab}, {BITS_TXT[b]}' for _, lab in rowsA for b in bits]
    n, r = len(bits), len(rowsA); order = [k * n + j for j in range(n) for k in range(r)]   # row-major like pgfplots
    fig.legend([hA[j] for j in order], [lA[j] for j in order], loc='upper center', ncol=n, fontsize=8,
               frameon=False, bbox_to_anchor=(0.5, 1.0))
    hB = [Line2D([], [], marker='o', **_mpl(MPL_GNN))] + (
        [Line2D([], [], marker='o', **_mpl(MPL_GD))] if has_gd else []) + [
        Line2D([], [], **_mpl(MPL_CD)), Line2D([], [], marker='s', **_mpl(MPL_ZF))]
    lB = ['GNN'] + ([GD_LABEL] if has_gd else []) + ['coordinate descent', 'ZF + $b$-bit DAC']
    fig.legend(hB, lB, loc='upper center', ncol=len(lB), fontsize=8, frameon=False, bbox_to_anchor=(0.5, 0.49))
    if note:
        fig.text(0.5, 0.5, note, ha='center', va='center', fontsize=20, color=(0.85, 0.2, 0.2), alpha=0.22,
                 rotation=18, weight='bold')
    fig.tight_layout(rect=(0, 0, 1, 0.93), h_pad=5.0)
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


def fig_single_run(snr, curves, title, lin, bits, out_tex, out_pdf, comment_lines=(), labels=None):
    """Per-run figure (one training folder) in the colours of the sweep figures. curves holds any
    of gnn / gnn_before / gnn_gd / cd / lin_q / lin_unq, aligned with snr; only the keys present are
    drawn, in that order. labels overrides the legend text per key."""
    lq = f'{lin} + {BITS_TXT[bits].replace(" bits", "-bit").replace(" bit", "-bit")} DAC'
    series = [
        ('gnn', 'GNN', 'gnnblue, line width=1.0pt, mark=*, mark size=1.9pt',
         dict(color=_rgb(GNN_ONE[1]), lw=1.6, marker='o', ms=4.5)),
        ('gnn_before', 'GNN, before continued training', 'gnnblue, line width=0.8pt, densely dashed',
         dict(color=_rgb(GNN_ONE[1]), lw=1.2, ls='--', alpha=0.7)),
        ('gnn_gd', GD_LABEL,
         'gdaqua, line width=1.0pt, mark=o, mark size=1.9pt, mark options={solid, fill=white}',
         dict(color=_rgb(GD_COL[1]), lw=1.6, marker='o', ms=4.5, mfc='white')),
        ('cd', 'coordinate descent',
         'cdorange, line width=0.8pt, dash dot, mark=diamond, mark size=2.2pt, mark options={solid, fill=white}',
         dict(color=_rgb(CD_COL[1]), lw=1.2, ls='-.', marker='D', ms=4.5, mfc='white')),
        ('lin_q', lq,
         'zfgray, line width=0.8pt, dashed, mark=square, mark size=1.8pt, mark options={solid, fill=white}',
         dict(color=_rgb(ZF_GRAY[1]), lw=1.2, ls='--', marker='s', ms=4.5, mfc='white')),
        ('lin_unq', f'{lin}, unquantized', 'unqgray, line width=0.9pt, densely dotted',
         dict(color=_rgb(UNQ_GRAY[1]), lw=1.4, ls=':')),
    ]
    series = [(k, (labels or {}).get(k, lab), sty, kw) for k, lab, sty, kw in series if k in curves]
    L = [_header(list(comment_lines), [GNN_ONE, GD_COL, CD_COL, ZF_GRAY, UNQ_GRAY, GRID])]
    L += [r'\begin{tikzpicture}', r'\begin{axis}[',
          '  width=8cm, height=6.4cm, tick align=outside, tick pos=left,',
          '  grid=major, grid style={line width=0.3pt, draw=gridgray},',
          f'  xmin={min(snr) - 2}, xmax={max(snr) + 2}, xtick={{{",".join(str(int(s)) for s in snr if s % 10 == 0)}}}, ymin=0,',
          r'  xlabel={SNR [dB]}, ylabel={$R_{\mathrm{sum}}$ [bits/channel use]},',
          r'  title={%s}, title style={font=\small},' % title,
          r'  tick label style={font=\footnotesize}, label style={font=\small},',
          r'  legend pos=north west, legend cell align=left,',
          r'  legend style={draw=none, fill=white, fill opacity=0.85, text opacity=1, font=\footnotesize},',
          ']']
    for key, lab, sty, _ in series:
        L += [r'\addplot[%s] %s' % (sty, _table(snr, curves[key])), r'\addlegendentry{%s}' % lab]
    L += [r'\end{axis}', r'\end{tikzpicture}', r'\end{document}', '']
    open(out_tex, 'w').write('\n'.join(L))

    fig, ax = plt.subplots(figsize=(6, 4.4))
    for key, lab, _, kw in series:
        ax.plot(snr, curves[key], label=lab, **kw)
    ax.set_xlim(min(snr) - 2, max(snr) + 2); ax.set_ylim(bottom=0)
    ax.set_xticks([s for s in snr if s % 10 == 0])
    ax.set_xlabel('SNR [dB]'); ax.set_ylabel('$R_{sum}$ [bits/channel use]'); ax.set_title(title, fontsize=10)
    ax.grid(color=_rgb(GRID[1]), lw=0.6); ax.legend(loc='upper left', fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out_pdf)
    plt.close(fig)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('results'); p.add_argument('out_dir')
    p.add_argument('--xmin', type=float, default=-10)
    p.add_argument('--preview-note', default='')
    a = p.parse_args()
    data = json.load(open(a.results)); os.makedirs(a.out_dir, exist_ok=True)
    fig_rate_vs_snr(data, os.path.join(a.out_dir, 'Rsum_sweep_vs_snr.tex'), a.xmin, a.preview_note)
    fig_rate_vs_bits(data, os.path.join(a.out_dir, 'Rsum_sweep_vs_bits.tex'), a.preview_note)
    preview(data, os.path.join(a.out_dir, 'preview_both_figures.png'), a.xmin, a.preview_note)
    print('wrote', os.listdir(a.out_dir))
