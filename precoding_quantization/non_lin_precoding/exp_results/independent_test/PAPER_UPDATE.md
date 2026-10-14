# Paper update to the independent test set (2026-10-14)

User request: compare the final models' independent-test results with the paper and update what disagrees.
Test set Htest[6144:8192] (2048 channels per K, never read before 2026-10-14); checkpoint selection stays on
Htest[4096:6144], now called the validation set in the paper. Pre-edit copies: session scratchpad
`paper_backup_2057/`.

## Figures (data only; styles unchanged) — `p1_update_paper_figures.py`

Every data block was first matched to the old stored series on Htest[4096:6144] before being replaced.

| Figure file | Content | Note |
|---|---|---|
| rate_vs_snrK1.tex (Fig. 3) | GNN, MRT + DAC, unquantized MRT, -30..30 dB | unplotted coordinate-descent columns dropped |
| Rsum_sweep_vs_snr.tex (Fig. 4) | 30 curves, K = 2/4/6 | |
| Rsum_sweep_vs_bits.tex (Fig. 5) | GNN, GNN-GD, ZF/MRT + DAC, WMMSE + DAC, unquantized markers | GNN-GD column: see below |
| ber_16qam_20dB.tex (Fig. 6) | 10 BER curves | 16-QAM test symbols seed 6144 |
| rsum_phase_offset.tex (Fig. 7) | 7 bar series, mean and offset std | |

**GNN-GD column of Fig. 5.** The values in the file had been lowered by hand in paper commit 8837c73
(2026-10-13; e.g. K=2: 10.0191 → 9.5191, K=6, b=3: 39.0350 → 36.3350), together with the text change
"88%–98%" → "93%–99%". No stored result has the lowered values. The stored old-slice GNN-GD
(continued runs, gnn_gd_curves.json) and the new test-set GNN-GD agree with each other (e.g. K=2, b=1:
10.019 vs 10.015). The figure now has the test-set values; the GNN / GNN-GD ratio is 87.8–97.9 %.

## main.tex

| Place | Old | New |
|---|---|---|
| Contribution (numerical evaluation) | 25%–153%; 93%–99% | 25%–152%; 88%–98% |
| IV-D LS check | 0.012 bit | 0.008 bit (test set; old slice reproduces 0.012) |
| IV-D, IV-F, Alg. | "held-out" rate / evaluation | validation rate / evaluation |
| V-A footnote | 30.4, 31.1, GNN 36.1 | 30.5, 31.1, GNN 36.2 |
| Table I (parameters) | Held-out channels 2048 | Validation / test channels 2048 / 2048 |
| VI-A setup | same set for reporting and selection (+ TODO) | 2048 test realizations, separate validation set |
| Table II (gains) | see gains.json | K1: 152/95, 79/57, 36/28; K2: 105/94, 61/58, 38/40; K4: 75/68, 44/40, 29/27; K6: 60/58, 36/32, 25/22 |
| VI-C text | 25%–153%, 22%–100% | 25%–152%, 22%–95% |
| VI-D GNN-GD | 93%–99%; gap grows with K for b=1, shrinks for b≥2 | 88%–98%; gap generally grows with K, smallest for b=3 |
| VI-E BER | factors 2.5–24; 8 of 12; ranking matches in all 12 | 2.5–21; 7 of 12; IDE (block β) ties for K=b=1 (ratio 1.04, 95% CI 0.95–1.14); otherwise ranking matches |
| VI-F phase | 2048 evaluation channels (+ AUTHOR CHECK) | 2048 test channels |

Unchanged because they still hold on the test set: 6.3 vs 2.1 at 30 dB (K=1); 1.8–2.3× / 1.4–1.5× at 30 dB;
from 5 dB upward; WMMSE + DAC lead below 0 dB; K=1 b=1 deficit up to 0 dB and larger in p5 (−37 % vs −19 %);
b≥2 above MRT + DAC everywhere; 6 of 8 cross-resolution comparisons and the K=4/6 exceptions; ~80 % of
unquantized WMMSE at b=3; absolute gain peaks at b=2; IDE calibrated median vs mean for K=b=1; phase:
~4 %, 11–15 %, ~80 % retained, b=1 crossover at 10°, std 6–8 % comparable to IDE and WMMSE.

Build: 0 errors. 14 pages before and after the update (the working tree was already at 14 pages); the update
changes the total by −5 words and leaves the last page unchanged.
Not touched: abstract, conclusions, Sec. VIII (OTA), Fig. 9/10. Pre-existing caption issue noticed: the Fig. 5
caption says "four panels", but the figure is a single axis.
