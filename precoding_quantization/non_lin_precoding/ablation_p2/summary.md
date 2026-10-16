# Prompt 2 ablation summary

Frozen test set Htest[6144:8192], 2048 channels, 20 dB, LS estimator, argmax decisions. Mean +- std over seeds; checkpoint chosen on validation only.

## K=2, b=1

| variant | seeds done | val 20 dB | test mean 20 dB | test p5 20 dB | test mean 30 dB | best ep | h/run |
|---|---|---|---|---|---|---|---|
| A Original (lr 5e-3) | 3 | 5.191 ± 4.477 | 5.203 ± 4.488 | 3.904 ± 3.377 | 5.556 ± 4.792 | [15, 14, 1] | 5.7 |
| A Original (lr 1e-3) | 1 | 7.743 ± 0.000 | 7.769 ± 0.000 | 5.960 ± 0.000 | 8.320 ± 0.000 | [14] | 6.5 |
| F Full (retrained) | 1 | 8.841 ± 0.000 | 8.837 ± 0.000 | 7.149 ± 0.000 | 9.906 ± 0.000 | [15] | 5.3 |
| F paper ckpt, 15 ep | 1 | 8.847 ± 0.000 | 8.840 ± 0.000 | 7.151 ± 0.000 | 9.921 ± 0.000 | [13] | nan |
| F paper ckpt, 15+15 ep | 1 | 8.951 ± 0.000 | 8.946 ± 0.000 | 7.238 ± 0.000 | 10.074 ± 0.000 | [24] | nan |

| effect (test, 20 dB) | Δ mean [bit/cu] | 95% CI (channels) | relative | share of channels better |
|---|---|---|---|---|
| paper F (15 ep) - A* (15 ep), same budget | +3.637 | [+3.619, +3.655] | +69.9% | 1.00 |
| paper F (15+15 ep) - A* (15 ep) | +3.743 | [+3.725, +3.761] | +71.9% | 1.00 |
| total (F - A*) | +3.634 | [+3.617, +3.652] | +69.8% | 1.00 |

## K=2, b=3

| variant | seeds done | val 20 dB | test mean 20 dB | test p5 20 dB | test mean 30 dB | best ep | h/run |
|---|---|---|---|---|---|---|---|
| A Original (lr 5e-3) | 0 (running: val 0.022) | | | | | | |
| A Original (lr 1e-3) | 3 | 12.015 ± 0.160 | 12.054 ± 0.154 | 9.397 ± 0.148 | 12.858 ± 0.186 | [15, 15, 14] | 6.0 |
| F paper ckpt, 15 ep | 1 | 15.504 ± 0.000 | 15.512 ± 0.000 | 12.678 ± 0.000 | 18.173 ± 0.000 | [14] | nan |
| F paper ckpt, 15+15 ep | 1 | 15.637 ± 0.000 | 15.647 ± 0.000 | 12.766 ± 0.000 | 18.436 ± 0.000 | [30] | nan |

| effect (test, 20 dB) | Δ mean [bit/cu] | 95% CI (channels) | relative | share of channels better |
|---|---|---|---|---|
| paper F (15 ep) - A* (15 ep), same budget | +3.458 | [+3.401, +3.519] | +28.7% | 1.00 |
| paper F (15+15 ep) - A* (15 ep) | +3.593 | [+3.535, +3.655] | +29.8% | 1.00 |

## K=6, b=1

| variant | seeds done | val 20 dB | test mean 20 dB | test p5 20 dB | test mean 30 dB | best ep | h/run |
|---|---|---|---|---|---|---|---|
| A Original (lr 5e-3) | 2 | 18.754 ± 0.337 | 18.812 ± 0.323 | 16.235 ± 0.265 | 20.123 ± 0.419 | [15, 14] | 10.5 |
| A Original (lr 1e-3) | 1 | 17.957 ± 0.000 | 17.997 ± 0.000 | 15.604 ± 0.000 | 19.103 ± 0.000 | [15] | 12.7 |
| F paper ckpt, 15 ep | 1 | 20.023 ± 0.000 | 20.054 ± 0.000 | 17.205 ± 0.000 | 21.836 ± 0.000 | [15] | nan |
| F paper ckpt, 15+15 ep | 1 | 20.318 ± 0.000 | 20.349 ± 0.000 | 17.491 ± 0.000 | 22.242 ± 0.000 | [29] | nan |

| effect (test, 20 dB) | Δ mean [bit/cu] | 95% CI (channels) | relative | share of channels better |
|---|---|---|---|---|
| paper F (15 ep) - A* (15 ep), same budget | +1.242 | [+1.217, +1.268] | +6.6% | 0.97 |
| paper F (15+15 ep) - A* (15 ep) | +1.537 | [+1.511, +1.563] | +8.2% | 0.99 |

## K=6, b=3

| variant | seeds done | val 20 dB | test mean 20 dB | test p5 20 dB | test mean 30 dB | best ep | h/run |
|---|---|---|---|---|---|---|---|
| A Original (lr 5e-3) | 1 | 0.067 ± 0.000 | 0.066 ± 0.000 | 0.028 ± 0.000 | 0.072 ± 0.000 | [1] | 11.7 |
| A Original (lr 1e-3) | 3 | 30.763 ± 0.402 | 30.834 ± 0.405 | 27.134 ± 0.075 | 33.759 ± 0.545 | [15, 14, 14] | 9.2 |
| F paper ckpt, 15 ep | 1 | 35.728 ± 0.000 | 35.833 ± 0.000 | 30.804 ± 0.000 | 41.770 ± 0.000 | [15] | nan |
| F paper ckpt, 15+15 ep | 1 | 36.054 ± 0.000 | 36.150 ± 0.000 | 31.063 ± 0.000 | 42.261 ± 0.000 | [29] | nan |

| effect (test, 20 dB) | Δ mean [bit/cu] | 95% CI (channels) | relative | share of channels better |
|---|---|---|---|---|
| paper F (15 ep) - A* (15 ep), same budget | +4.999 | [+4.928, +5.069] | +16.2% | 1.00 |
| paper F (15+15 ep) - A* (15 ep) | +5.316 | [+5.244, +5.388] | +17.2% | 1.00 |

