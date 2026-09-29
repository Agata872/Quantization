"""Print the LaTeX rows of the GNN-gain table (tab:tp_gain in main.tex) from exp_results/rate_gain_table.json.

Layout (user's structure, compacted): rows (K, b) for K in {1, 2, 4, 6}; column groups GNN vs. ZF/MRT + DAC and
GNN vs. WMMSE + DAC, each at SNR = 0, 10, 20, 30 dB; every cell is d_mu/d_5 in percent, d = 100 (T_gnn / T_base - 1)
with T the mean (d_mu) or the 5th percentile (d_5) of the per-channel sum rate; 20 dB in bold. For K = 1, WMMSE
coincides with MRT, so the WMMSE group is a single note spanning the K = 1 rows.

  python make_gain_table_tex.py
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SNRS = ('0', '10', '20', '30')


def pct(v):
    r = int(abs(v) + 0.5)                          # round half away from zero (Python's round() is half-to-even)
    return f'{"+" if r and v > 0 else ("-" if r and v < 0 else "")}{r}'


def cell(v, base, x):
    mu = 100 * (v['gnn']['mean'] / v[base]['mean'] - 1)
    p5 = 100 * (v['gnn']['p5'] / v[base]['p5'] - 1)
    body = f'{{{pct(mu)}}}/{{{pct(p5)}}}'
    return f'$\\mathbf{{{body}}}$' if x == '20' else f'${body}$'


def main():
    res = json.load(open(os.path.join(HERE, 'exp_results', 'rate_gain_table.json')))['results']
    lines = []
    for K in (1, 2, 4, 6):
        for b in (1, 2, 3):
            r = res[f'K{K}b{b}']
            head = f'\\multirow{{3}}{{*}}{{${K}$}}' if b == 1 else ''
            left = ' & '.join(cell(r[x], 'zf_dac', x) for x in SNRS)
            if K == 1:
                right = (r'\multicolumn{4}{c}{\multirow{3}{*}{\emph{identical: WMMSE $\equiv$ MRT for $K=1$}}}'
                         if b == 1 else r'\multicolumn{4}{c}{}')
            else:
                right = ' & '.join(cell(r[x], 'wmmse_dac', x) for x in SNRS)
            lines.append(f'    {head} & ${b}$ & {left} & {right} \\\\')
        if K != 6:
            lines.append(r'    \addlinespace[3pt]')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
