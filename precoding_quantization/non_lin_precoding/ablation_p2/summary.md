# Prompt 2 ablation summary

Frozen test set Htest[6144:8192], 2048 channels, 20 dB, LS estimator, argmax decisions. Mean +- std over seeds; checkpoint chosen on validation only.

## K=2, b=1

| variant | seeds done | val 20 dB | test mean 20 dB | test p5 20 dB | test mean 30 dB | best ep | h/run |
|---|---|---|---|---|---|---|---|
| A Original (lr 5e-3) | 1 | 7.752 ± 0.000 | 7.769 ± 0.000 | 5.835 ± 0.000 | 8.283 ± 0.000 | [15] | 5.2 |
| A Original (lr 1e-3) | 1 | 7.743 ± 0.000 | 7.769 ± 0.000 | 5.960 ± 0.000 | 8.320 ± 0.000 | [14] | 6.5 |
| F Full (retrained) | 1 | 8.841 ± 0.000 | 8.837 ± 0.000 | 7.149 ± 0.000 | 9.906 ± 0.000 | [15] | 5.3 |
| F paper ckpt, 15 ep | 1 | 8.847 ± 0.000 | 8.840 ± 0.000 | 7.151 ± 0.000 | 9.921 ± 0.000 | [13] | nan |
| F paper ckpt, 15+15 ep | 1 | 8.951 ± 0.000 | 8.946 ± 0.000 | 7.238 ± 0.000 | 10.074 ± 0.000 | [24] | nan |

| effect (test, 20 dB) | Δ mean [bit/cu] | 95% CI (channels) | relative | share of channels better |
|---|---|---|---|---|
| paper F (15 ep) - A* (15 ep), same budget | +1.071 | [+1.052, +1.091] | +13.8% | 0.99 |
| paper F (15+15 ep) - A* (15 ep) | +1.177 | [+1.158, +1.196] | +15.1% | 1.00 |
| total (F - A*) | +1.068 | [+1.049, +1.088] | +13.7% | 0.99 |

## K=2, b=3

| variant | seeds done | val 20 dB | test mean 20 dB | test p5 20 dB | test mean 30 dB | best ep | h/run |
|---|---|---|---|---|---|---|---|
| A Original (lr 5e-3) | 0 (running: val 0.022) | | | | | | |
| A Original (lr 1e-3) | 0 (running: val 10.282) | | | | | | |
| F paper ckpt, 15 ep | 1 | 15.504 ± 0.000 | 15.512 ± 0.000 | 12.678 ± 0.000 | 18.173 ± 0.000 | [14] | nan |
| F paper ckpt, 15+15 ep | 1 | 15.637 ± 0.000 | 15.647 ± 0.000 | 12.766 ± 0.000 | 18.436 ± 0.000 | [30] | nan |

## K=6, b=1

| variant | seeds done | val 20 dB | test mean 20 dB | test p5 20 dB | test mean 30 dB | best ep | h/run |
|---|---|---|---|---|---|---|---|
| A Original (lr 5e-3) | 0 (running: val 14.136) | | | | | | |
| A Original (lr 1e-3) | 0 (running: val ) | | | | | | |
| F paper ckpt, 15 ep | 1 | 20.023 ± 0.000 | 20.054 ± 0.000 | 17.205 ± 0.000 | 21.836 ± 0.000 | [15] | nan |
| F paper ckpt, 15+15 ep | 1 | 20.318 ± 0.000 | 20.349 ± 0.000 | 17.491 ± 0.000 | 22.242 ± 0.000 | [29] | nan |

## K=6, b=3

| variant | seeds done | val 20 dB | test mean 20 dB | test p5 20 dB | test mean 30 dB | best ep | h/run |
|---|---|---|---|---|---|---|---|
| A Original (lr 5e-3) | 0 (running: val ) | | | | | | |
| A Original (lr 1e-3) | 0 (running: val ) | | | | | | |
| F paper ckpt, 15 ep | 1 | 35.728 ± 0.000 | 35.833 ± 0.000 | 30.804 ± 0.000 | 41.770 ± 0.000 | [15] | nan |
| F paper ckpt, 15+15 ep | 1 | 36.054 ± 0.000 | 36.150 ± 0.000 | 31.063 ± 0.000 | 42.261 ± 0.000 | [29] | nan |

