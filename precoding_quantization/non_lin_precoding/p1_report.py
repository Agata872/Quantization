"""Prompt 1 report: paper numbers on the selection set Htest[4096:6144] (old) vs. the frozen independent test set
Htest[6144:8192] (new), from the outputs of p1_independent_eval.py. Writes exp_results/independent_test/report.md
and report.json. Only reads files; computes nothing that selects or tunes.

  python p1_report.py
"""
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
OUT = os.path.join(HERE, 'exp_results', 'independent_test')
SNR = [float(x) for x in range(-30, 31, 5)]
I = {x: SNR.index(x) for x in SNR}
CFG = [(K, b) for K in (1, 2, 4, 6) for b in (1, 2, 3)]


def old_curves():
    import plot_continue_figures as P
    return P.load()          # {(K, b): gnn, gd, lin_q, lin_unq, wmmse_q, wmmse_unq} on Htest[4096:6144]


def new_curves():
    d = {}
    for K, b in CFG:
        f = os.path.join(OUT, f'rates_K{K}b{b}.npz')
        if os.path.exists(f):
            z = np.load(f)
            d[(K, b)] = {k: z[k] for k in ('gnn', 'lin_q', 'lin_unq', 'wmmse_q', 'wmmse_unq')}
            g = os.path.join(OUT, f'gd_K{K}b{b}.npz')
            if os.path.exists(g):
                d[(K, b)]['gd20'] = np.load(g)['gnn_gd_20']
    return d


def fmt_ci(v, ci):
    return f'{v:.0f} [{ci[0]:.0f}, {ci[1]:.0f}]'


def main():
    old, new = old_curves(), new_curves()
    rep, md = {}, ['# Prompt 1: independent test-set re-evaluation', '',
                   'Old = Htest[4096:6144] (paper: checkpoint selection and reporting). '
                   'New = Htest[6144:8192] (never read before 2026-10-14). Same frozen checkpoints, same baselines, '
                   'same estimator (LS Bussgang), same Gaussian symbols per channel from the dataset.', '']
    ind = json.load(open(os.path.join(OUT, 'independence.json')))
    md += ['## Independence', '',
           f"* Slices referenced by code: {sorted({tuple(u['range']) for u in ind['code_slices']})}; "
           f"overlap with the test slice: {ind['any_code_slice_overlaps_test']}.",
           '* Exact duplicates of test rows in Htrain / Hval / rest of Htest: '
           + ', '.join(f"{k}: {sum(v['duplicates_of_test_rows_in'].values())}" for k, v in ind['data'].items()) + '.',
           '* Two-sample KS p-values (test vs. training, log10 ||H||_F^2 / strongest-AP share): '
           + ', '.join(f"{k}: {v['ks_pvalues']['log10_fro2']['vs_train100k']:.3f} / "
                       f"{v['ks_pvalues']['strongest_ap_share']['vs_train100k']:.3f}" for k, v in ind['data'].items())
           + ' (16 tests; Bonferroni 5% threshold 0.003).', '']

    # ---------------------------------------------------------------- Table I
    gains = json.load(open(os.path.join(OUT, 'gains.json')))['results'] if os.path.exists(
        os.path.join(OUT, 'gains.json')) else {}
    if gains:
        md += ['## Table I (gain over WMMSE + DAC at 20 dB, %)', '',
               '| K | b | Δμ old | Δμ new [95% CI] | Δ5 old | Δ5 new [95% CI] | GNN better on |',
               '|---|---|---|---|---|---|---|']
        for K, b in CFG:
            g = gains.get(f'K{K}b{b}', {}).get('20')
            if not g:
                continue
            e, o = g['vs_wmmse_dac'], g.get('old_eval', {})
            md.append(f"| {K} | {b} | {o.get('d_mu_pct_vs_wmmse', float('nan')):.0f} | {fmt_ci(e['d_mu_pct'], e['d_mu_ci95'])} | "
                      f"{o.get('d_5_pct_vs_wmmse', float('nan')):.0f} | {fmt_ci(e['d_5_pct'], e['d_5_ci95'])} | "
                      f"{100 * e['frac_channels_better']:.1f}% |")
        mus = [gains[k]['20']['vs_wmmse_dac']['d_mu_pct'] for k in gains]
        p5s = [gains[k]['20']['vs_wmmse_dac']['d_5_pct'] for k in gains]
        md += ['', f'Text range (paper: Δμ 25–153%, Δ5 22–100%): new Δμ {min(mus):.0f}–{max(mus):.0f}%, '
                   f'Δ5 {min(p5s):.0f}–{max(p5s):.0f}%.', '']
        md += ['30 dB check (Δμ / Δ5 vs WMMSE + DAC, new): '
               + ', '.join(f"K{k[1]}b{k[3]} {gains[k]['30']['vs_wmmse_dac']['d_mu_pct']:.0f}/"
                           f"{gains[k]['30']['vs_wmmse_dac']['d_5_pct']:.0f}" for k in gains), '']
        rep['table1'] = gains

    # ---------------------------------------------------------------- curves
    if new:
        md += ['## Mean sum rate [bit/channel use], old → new', '',
               '| K | b | SNR | GNN | ZF/MRT+DAC | WMMSE+DAC | WMMSE unq. | max abs Δ(GNN) over -30..30 dB |',
               '|---|---|---|---|---|---|---|---|']
        for (K, b), n in new.items():
            o = old[(K, b)]
            dmax = max(abs(o['gnn'][i] - n['gnn'][i].mean()) for i in range(len(SNR)))
            for x in (0.0, 20.0, 30.0):
                i = I[x]
                md.append(f"| {K} | {b} | {x:g} | {o['gnn'][i]:.3f} → {n['gnn'][i].mean():.3f} | "
                          f"{o['lin_q'][i]:.3f} → {n['lin_q'][i].mean():.3f} | {o['wmmse_q'][i]:.3f} → "
                          f"{n['wmmse_q'][i].mean():.3f} | {o['wmmse_unq'][i]:.3f} → {n['wmmse_unq'][i].mean():.3f} | "
                          + (f'{dmax:.3f} |' if x == 0.0 else ' |'))
        md.append('')
        md += ['## Text claims, old vs. new', '']
        claims = []

        def both(fn):
            try:
                return fn(old, lambda K, b, k, i: old[(K, b)][k][i]), fn(new, lambda K, b, k, i: float(np.mean(new[(K, b)][k][i])))
            except KeyError:
                return None, None

        if (1, 1) in new:
            o, n = both(lambda D, g: (g(1, 1, 'gnn', I[30.0]), g(1, 1, 'lin_q', I[30.0])))
            claims.append(f'K=1, b=1, 30 dB, GNN vs MRT + DAC (paper 6.3 vs 2.1): old {o[0]:.2f} vs {o[1]:.2f}; '
                          f'new {n[0]:.2f} vs {n[1]:.2f}')
        for bb in (1, 3):
            ks = [K for K in (2, 4, 6) if (K, bb) in new]
            if ks:
                f = lambda D, g: [g(K, bb, 'gnn', I[30.0]) / max(g(K, bb, 'lin_q', I[30.0]), g(K, bb, 'wmmse_q', I[30.0]))  # noqa: E731
                                  for K in ks]
                o, n = both(f)
                claims.append(f'30 dB, b={bb}, K={ks}: GNN / best quantized linear: old {min(o):.2f}–{max(o):.2f}, '
                              f'new {min(n):.2f}–{max(n):.2f} (paper: b=1 1.8–2.3, b=3 1.4–1.5)')
        # lowest SNR from which the GNN beats both quantized linear baselines in every configuration
        for name, D, g in (('old', old, lambda K, b, k, i: old[(K, b)][k][i]),
                           ('new', new, lambda K, b, k, i: float(np.mean(new[(K, b)][k][i])))):
            ok = [x for x in SNR if all(g(K, b, 'gnn', I[x]) > max(g(K, b, 'lin_q', I[x]), g(K, b, 'wmmse_q', I[x]))
                                        for (K, b) in new if K > 1)]
            first = next((x for x in SNR if all(y in ok for y in SNR if y >= x)), None)
            claims.append(f'{name}: GNN above both quantized linear baselines for every K>=2 configuration from '
                          f'{first} dB upward (paper: 5 dB)')
            k1 = [x for x in SNR if (1, 1) in new and g(1, 1, 'gnn', I[x]) < g(1, 1, 'lin_q', I[x])]
            claims.append(f'{name}: K=1, b=1 GNN below MRT + DAC at {k1} dB (paper: up to 0 dB)')
            lead = [(K, b, x) for (K, b) in new if K > 1 for x in SNR if x < 0
                    and g(K, b, 'wmmse_q', I[x]) > g(K, b, 'gnn', I[x]) and b > 1]
            claims.append(f'{name}: WMMSE + DAC ahead of the GNN below 0 dB for b>=2 at (K, b, SNR) = {lead} '
                          '(paper: b=2 with K>=4, b=3 with K=6)')
            # Fig. 5: b-bit GNN vs (b+1)-bit linear
            cmp_ = [(K, b) for K in (1, 2, 4, 6) for b in (1, 2) if (K, b) in new and (K, b + 1) in new]
            wins = [(K, b) for K, b in cmp_ if g(K, b, 'gnn', I[20.0]) > max(g(K, b + 1, 'lin_q', I[20.0]),
                                                                              g(K, b + 1, 'wmmse_q', I[20.0]))]
            claims.append(f'{name}: b-bit GNN above both (b+1)-bit quantized linear baselines at 20 dB in '
                          f'{len(wins)}/{len(cmp_)} comparisons; exceptions {sorted(set(cmp_) - set(wins))} '
                          '(paper: 6/8, exceptions b=1 with K>=4)')
            r80 = [g(K, 3, 'gnn', I[20.0]) / g(K, 3, 'wmmse_unq', I[20.0]) for K in (1, 2, 4, 6) if (K, 3) in new]
            if r80:
                claims.append(f'{name}: b=3 GNN / unquantized WMMSE (MRT for K=1) at 20 dB: '
                              + ', '.join(f'{v:.2f}' for v in r80) + ' (paper: about 80%)')
            absg = {K: [g(K, b, 'gnn', I[20.0]) - max(g(K, b, 'lin_q', I[20.0]), g(K, b, 'wmmse_q', I[20.0]))
                        for b in (1, 2, 3) if (K, b) in new] for K in (1, 2, 4, 6)}
            claims.append(f'{name}: absolute gain over the best quantized linear baseline at 20 dB, b=1/2/3: '
                          + '; '.join(f'K={K} ' + '/'.join(f'{v:.2f}' for v in vs) for K, vs in absg.items() if vs)
                          + ' (paper: peaks at b=2 for every K)')
        gd = [(K, b, float(new[(K, b)]['gnn'][I[20.0]].mean() / new[(K, b)]['gd20'].mean()))
              for (K, b) in new if 'gd20' in new[(K, b)]]
        if gd:
            claims.append('new: GNN / GNN-GD at 20 dB: ' + ', '.join(f'K{K}b{b} {r:.3f}' for K, b, r in gd)
                          + f' (range {min(r for *_, r in gd):.3f}–{max(r for *_, r in gd):.3f}; paper 93–99%)')
        ogd = [old[k]['gnn'][I[20.0]] / old[k]['gd'][I[20.0]] for k in old]
        claims.append(f'old: GNN / GNN-GD at 20 dB range {min(ogd):.3f}–{max(ogd):.3f}')
        md += [f'* {c}' for c in claims] + ['']
        rep['claims'] = claims

    # ---------------------------------------------------------------- BER
    berold = json.load(open(os.path.join(HERE, 'exp_results', 'ber_16qam_20dB.json')))['results']
    rows = []
    for K, b in CFG:
        f = os.path.join(OUT, f'ber_K{K}b{b}.npz')
        if os.path.exists(f):
            z = np.load(f)
            rows.append((K, b, {m: float(z[m].mean()) for m in ('gnn', 'ide_cal', 'ide_block', 'zf_dac', 'squid') if m in z},
                         berold[f'K{K}b{b}'], json.loads(str(z['extra']))))
    if rows:
        md += ['## 16-QAM BER at 20 dB (Fig. 6), old → new', '',
               '| K | b | GNN | IDE cal. | IDE block | SQUID | GNN lowest? |', '|---|---|---|---|---|---|---|']
        nlow = {'old': 0, 'new': 0}
        sq = {'old': [], 'new': []}
        for K, b, n, o, ex in rows:
            for tag, d in (('old', o), ('new', n)):
                if d['gnn'] < min(d['ide_cal'], d['ide_block']):
                    nlow[tag] += 1
                if 'squid' in d:
                    sq[tag].append(d['squid'] / d['gnn'])
            md.append(f"| {K} | {b} | {o['gnn']:.2e} → {n['gnn']:.2e} | {o['ide_cal']:.2e} → {n['ide_cal']:.2e} | "
                      f"{o['ide_block']:.2e} → {n['ide_block']:.2e} | "
                      + (f"{o['squid']:.2e} → {n['squid']:.2e} (gain {ex.get('squid_gain')}, paper {ex.get('squid_gain_paper')})"
                         if 'squid' in n else '') + f" | {o['gnn'] < min(o['ide_cal'], o['ide_block'])} → "
                      f"{n['gnn'] < min(n['ide_cal'], n['ide_block'])} |")
        md += ['', f"GNN below both IDE variants: old {nlow['old']}/{len(rows)}, new {nlow['new']}/{len(rows)} (paper 8/12). "
                   f"SQUID / GNN BER ratio (b=1): old {min(sq['old'], default=0):.1f}–{max(sq['old'], default=0):.1f}, "
                   f"new {min(sq['new'], default=0):.1f}–{max(sq['new'], default=0):.1f} (paper 2.5–24).", '']
        # paired bootstrap over channels of the BER ratios, and the tail share of the mean BER
        md += ['BER ratios on the test set with paired bootstrap 95% CI over channels (5000 resamples); '
               '"worst 1%": share of the mean BER contributed by the worst 1% of channels.', '',
               '| K | b | GNN/IDE cal. | GNN/IDE block | SQUID/GNN | worst 1% share GNN / IDE cal. / IDE block |',
               '|---|---|---|---|---|---|']
        g = np.random.default_rng(20261014)
        for K, b, n, o, ex in rows:
            z = np.load(os.path.join(OUT, f'ber_K{K}b{b}.npz'))
            idx = g.integers(0, len(z['gnn']), size=(5000, len(z['gnn'])))

            def rci(a, c):
                r = z[a][idx].mean(1) / z[c][idx].mean(1)
                return f"{z[a].mean() / z[c].mean():.2f} [{np.percentile(r, 2.5):.2f}, {np.percentile(r, 97.5):.2f}]"

            def tail(a):
                v = np.sort(z[a])[::-1]
                return 100 * v[:max(1, len(v) // 100)].sum() / v.sum()
            md.append(f"| {K} | {b} | {rci('gnn', 'ide_cal')} | {rci('gnn', 'ide_block')} | "
                      + (rci('squid', 'gnn') if 'squid' in z else '') + f" | {tail('gnn'):.0f}% / {tail('ide_cal'):.0f}% / "
                      f"{tail('ide_block'):.0f}% |")
        md.append('')
        rep['ber'] = [{'K': K, 'b': b, 'new': n, 'old': {k: v for k, v in o.items() if not isinstance(v, dict)}}
                      for K, b, n, o, _ in rows]

    # ---------------------------------------------------------------- IDE rates and BER/rate ranking
    idr = []
    iold = json.load(open(os.path.join(HERE, 'exp_results', 'ide_baseline.json')))['results']
    for K, b in CFG:
        f = os.path.join(OUT, f'idrates_K{K}b{b}.npz')
        if os.path.exists(f) and (K, b) in new:
            z = np.load(f)
            idr.append((K, b, float(new[(K, b)]['gnn'][I[20.0]].mean()), float(z['ide_block_20'].mean()),
                        float(z['ide_cal_20'].mean()), iold[f'K{K}b{b}']['ide_block']['20'], old[(K, b)]['gnn'][I[20.0]]))
    if idr:
        md += ['## Sum rate of IDE at 20 dB, and BER-vs-rate ranking against IDE block β', '',
               '| K | b | GNN new | IDE block new (old) | IDE cal. new | rate rank = BER rank (new) |', '|---|---|---|---|---|---|']
        agree = 0
        bmap = {(K, b): n for K, b, n, *_ in rows}
        for K, b, g, ib, ic, ibo, go in idr:
            same = None
            if (K, b) in bmap:
                same = (g > ib) == (bmap[(K, b)]['gnn'] < bmap[(K, b)]['ide_block'])
                agree += bool(same)
            md.append(f'| {K} | {b} | {g:.3f} (old {go:.3f}) | {ib:.3f} ({ibo:.3f}) | {ic:.3f} | {same} |')
        md += ['', f'Ranking agreement GNN vs IDE block β: {agree}/{len(idr)} (paper 12/12).', '']

    # ---------------------------------------------------------------- phase
    pf = os.path.join(OUT, 'phase.json')
    if os.path.exists(pf):
        P = json.load(open(pf))
        pold = json.load(open(os.path.join(HERE, 'exp_results', 'phase_offset_eval.json')))
        sd = P['sigma_deg']
        j20 = pold['snr_db'].index(20.0)
        md += ['## Phase offsets at 20 dB (Fig. 7), old → new', '',
               '| K | b | GNN loss 10°/20° [%] | IDE loss 10°/20° [%] | GNN−IDE at 0° | gain retained over IDE 20° [95% CI] | GNN std_offset/mean 20° |',
               '|---|---|---|---|---|---|---|']
        for key, r in P['results'].items():
            o = pold['results'][key]['methods']
            mg, mi = r['methods']['gnn']['mean'], r['methods']['ide_cal']['mean']
            og = [row[j20] for row in o['gnn']['mean']]
            oi = [row[j20] for row in o['ide_cal']['mean']]
            l = lambda m, i: 100 * (1 - m[i] / m[0])  # noqa: E731
            i10, i20 = sd.index(10.0), sd.index(20.0)
            ret = r['boot']['gain_retained_vs_ide_20deg']
            oret = (og[i20] - oi[i20]) / (og[0] - oi[0])
            so = r['methods']['gnn']['std_offset'][i20] / mg[i20]
            soo = o['gnn']['std_offset'][i20][j20] / og[i20]
            md.append(f'| {key[1]} | {key[3]} | {l(og, i10):.1f}/{l(og, i20):.1f} → {l(mg, i10):.1f}/{l(mg, i20):.1f} | '
                      f'{l(oi, i10):.1f}/{l(oi, i20):.1f} → {l(mi, i10):.1f}/{l(mi, i20):.1f} | '
                      f'{og[0] - oi[0]:+.2f} → {mg[0] - mi[0]:+.2f} | {oret:.2f} → {ret[0]:.2f} [{ret[1]:.2f}, {ret[2]:.2f}] | '
                      f'{100 * soo:.1f}% → {100 * so:.1f}% |')
            cross = next((s for s, a, c in zip(sd, mg, mi) if a > c), None)
            ocross = next((s for s, a, c in zip(sd, og, oi) if a > c), None)
            md.append(f'|  |  | first σθ with GNN > IDE: {ocross} → {cross} |  |  |  |  |')
        md.append('')
        rep['phase'] = P['results']
    open(os.path.join(OUT, 'report.md'), 'w').write('\n'.join(md) + '\n')
    json.dump(rep, open(os.path.join(OUT, 'report.json'), 'w'), indent=1, default=float)
    print('\n'.join(md))


if __name__ == '__main__':
    main()
