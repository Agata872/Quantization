#!/usr/bin/env python3
"""Plot received 16-QAM constellations exclusively from saved simulation IQ.

Usage::

    python plot_constellation_16qam.py --input-dir exp_results/constellation_16qam

The shared files contain the transmitted symbols/bits and the split between
independent-channel evaluation blocks and the trailing fixed-channel blocks.
Each case file contains the receiver's saved, complex, gain-equalized samples.
No noise, jitter, or synthetic constellation samples are generated here.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap


LEVELS = np.asarray([-3.0, -1.0, 1.0, 3.0]) / np.sqrt(10.0)
GRAY_BITS = np.asarray([[0, 0], [0, 1], [1, 1], [1, 0]], dtype=np.uint8)
AXIS_LIMIT = 1.8
MAX_ENSEMBLE_POINTS = 12_000
# Runs made before the baseline became selectable have no "methods" entry in metadata.json.
DEFAULT_METHODS = ("gnn", "wmmse")
LABELS = {"gnn": "GNN", "wmmse": "WMMSE", "ide_wf": r"IDE, $\beta_{\mathrm{WF}}$", "ide_block": "IDE, block β",
          "ide_cal": "IDE, calibrated β"}


@dataclass
class SharedData:
    symbols: np.ndarray
    bits: np.ndarray
    n_eval: int
    snapshot_channel_index: int


@dataclass
class Panel:
    received: np.ndarray
    symbol_ids: np.ndarray
    metrics: dict


def hard_demap(symbols: np.ndarray) -> np.ndarray:
    """Nearest-neighbor Gray 16-QAM demapping; bits are [I0,I1,Q0,Q1]."""
    real_index = np.argmin(np.abs(symbols.real[..., None] - LEVELS), axis=-1)
    imag_index = np.argmin(np.abs(symbols.imag[..., None] - LEVELS), axis=-1)
    return np.concatenate((GRAY_BITS[real_index], GRAY_BITS[imag_index]), axis=-1)


def _symbol_ids(symbols: np.ndarray) -> np.ndarray:
    real_index = np.argmin(np.abs(symbols.real[..., None] - LEVELS), axis=-1)
    imag_index = np.argmin(np.abs(symbols.imag[..., None] - LEVELS), axis=-1)
    return 4 * real_index + imag_index


def _read_shared(input_dir: Path, users: int) -> SharedData:
    source = input_dir / f"shared_K{users}.npz"
    with np.load(source, allow_pickle=False) as saved:
        shared = SharedData(
            symbols=saved["tx_symbols"],
            bits=saved["tx_bits"],
            n_eval=int(saved["n_eval"].item()),
            snapshot_channel_index=int(saved["snapshot_channel_index"].item()),
        )
    if shared.symbols.ndim != 3 or shared.symbols.shape[1] != users:
        raise ValueError(f"{source}: tx_symbols must have shape [blocks,{users},symbols].")
    if shared.bits.shape != shared.symbols.shape + (4,):
        raise ValueError(f"{source}: tx_bits must append a four-bit dimension to tx_symbols.")
    if not 0 < shared.n_eval <= shared.symbols.shape[0]:
        raise ValueError(f"{source}: expected a nonempty evaluation prefix, optionally followed by snapshot blocks.")
    if not np.all(np.isfinite(shared.symbols)):
        raise ValueError(f"{source}: transmitted symbols contain non-finite values.")
    if not np.array_equal(hard_demap(shared.symbols), shared.bits):
        raise ValueError(f"{source}: tx_bits do not match Gray mapping [I0,I1,Q0,Q1].")
    return shared


def _make_panel(
    received: np.ndarray,
    transmitted: np.ndarray,
    bits: np.ndarray,
    max_points: int | None,
) -> Panel:
    rx = received.reshape(-1)
    tx = transmitted.reshape(-1)
    true_bits = bits.reshape(-1, 4)
    errors = rx.astype(np.complex128) - tx
    symbol_power = float(np.mean(np.abs(tx.astype(np.complex128)) ** 2))
    bit_errors = int(np.count_nonzero(hard_demap(rx) != true_bits))
    clipped = (np.abs(rx.real) > AXIS_LIMIT) | (np.abs(rx.imag) > AXIS_LIMIT)
    metrics = {
        "symbols": int(rx.size),
        "evm_rms_percent": float(100.0 * np.sqrt(np.mean(np.abs(errors) ** 2) / symbol_power)),
        "ber": float(bit_errors / true_bits.size),
        "bit_errors": bit_errors,
        "bits": int(true_bits.size),
        "outside_window_percent": float(100.0 * np.mean(clipped)),
        "axis_limit": AXIS_LIMIT,
    }
    if max_points is not None and rx.size > max_points:
        selection = np.linspace(0, rx.size - 1, max_points, dtype=np.int64)
        rx = rx[selection]
        tx = tx[selection]
    metrics["plotted_symbols"] = int(rx.size)
    metrics["plotted_outside_window_percent"] = float(
        100.0 * np.mean((np.abs(rx.real) > AXIS_LIMIT) | (np.abs(rx.imag) > AXIS_LIMIT))
    )
    return Panel(received=rx, symbol_ids=_symbol_ids(tx), metrics=metrics)


def _load_panels(input_dir: Path, shared: dict[int, SharedData], resolutions: tuple[int, ...],
                 methods: tuple[str, ...]) -> dict:
    panels = {"snapshot": {}, "ensemble": {}}
    for users, common in shared.items():
        for dac_bits in resolutions:
            for method in methods:
                source = input_dir / f"K{users}_b{dac_bits}_{method}.npz"
                with np.load(source, allow_pickle=False) as saved:
                    # NPZ members are loaded lazily: large per-antenna TX arrays
                    # and raw RX arrays are unnecessary for these figures.
                    received = saved["rx_equalized"]
                if received.shape != common.symbols.shape:
                    raise ValueError(f"{source}: rx_equalized does not match tx_symbols shape.")
                if not np.all(np.isfinite(received)):
                    raise ValueError(f"{source}: rx_equalized contains non-finite values.")
                for user in range(users):
                    key = (users, user, method, dac_bits)
                    for mode, selection, max_points in (
                        ("snapshot", slice(common.n_eval, None), None),
                        ("ensemble", slice(0, common.n_eval), MAX_ENSEMBLE_POINTS),
                    ):
                        if mode == "snapshot" and common.n_eval == common.symbols.shape[0]:
                            continue
                        panels[mode][key] = _make_panel(
                            received[selection, user, :],
                            common.symbols[selection, user, :],
                            common.bits[selection, user, :, :],
                            max_points,
                        )
    return panels


def _format_ber(ber: float) -> str:
    return "0" if ber == 0.0 else f"{ber:.2g}"


def _draw_figure(
    output_dir: Path,
    stem: str,
    rows: list[tuple[int, int, str]],
    panels: dict,
    shared: dict[int, SharedData],
    metadata: dict,
    mode: str,
) -> list[Path]:
    nrows = len(rows)
    resolutions = tuple(metadata["bits"])
    height = 2.55 * nrows + 1.65
    width = max(8.0, 3.0 * len(resolutions) + 1.5)
    fig, axes = plt.subplots(nrows, len(resolutions), figsize=(width, height), squeeze=False)
    fig.subplots_adjust(left=2.2 / width, right=0.985, bottom=0.90 / height,
                        top=1.0 - 1.03 / height, hspace=0.17, wspace=0.15)
    color_map = ListedColormap(plt.get_cmap("tab20")(np.linspace(0, 1, 16)))
    ideal = (LEVELS[:, None] + 1j * LEVELS[None, :]).reshape(-1)

    for row_index, (users, user, method) in enumerate(rows):
        for col_index, dac_bits in enumerate(resolutions):
            ax = axes[row_index, col_index]
            panel = panels[(users, user, method, dac_bits)]
            rx = panel.received
            ax.scatter(rx.real, rx.imag, c=panel.symbol_ids, cmap=color_map,
                       vmin=-0.5, vmax=15.5, s=3.3 if mode == "snapshot" else 2.0,
                       alpha=0.40 if mode == "snapshot" else 0.22,
                       edgecolors="none", rasterized=True, zorder=2)
            ax.scatter(ideal.real, ideal.imag, marker="x", c="black", s=20,
                       linewidths=0.85, zorder=4)
            ax.set(xlim=(-AXIS_LIMIT, AXIS_LIMIT), ylim=(-AXIS_LIMIT, AXIS_LIMIT))
            ax.set_aspect("equal", adjustable="box")
            ax.set_xticks([-1.5, -0.75, 0, 0.75, 1.5])
            ax.set_yticks([-1.5, -0.75, 0, 0.75, 1.5])
            ax.grid(color="#dadde1", linewidth=0.45, alpha=0.8, zorder=0)
            ax.tick_params(labelsize=7.8, length=2.5, pad=2)
            for spine in ax.spines.values():
                spine.set_color("#939aa3")
                spine.set_linewidth(0.65)
            if row_index == 0:
                ax.set_title(f"DAC: {dac_bits} bit{'s' if dac_bits > 1 else ''}",
                             fontsize=11, fontweight="semibold", pad=8)
            if row_index == nrows - 1:
                ax.set_xlabel("In-phase (I)", fontsize=9, labelpad=3)
            else:
                ax.tick_params(labelbottom=False)
            if col_index == 0:
                ax.set_ylabel("Quadrature (Q)", fontsize=9, labelpad=3)
                row_label = f"K = {users}, UE {user + 1}\n{LABELS.get(method, method.upper())}"
                ax.text(-0.40, 0.5, row_label, transform=ax.transAxes,
                        ha="right", va="center", fontsize=10,
                        fontweight="semibold", linespacing=1.6)
            else:
                ax.tick_params(labelleft=False)
            metrics = panel.metrics
            annotation = (
                f"EVM {metrics['evm_rms_percent']:.1f}%   "
                f"BER {_format_ber(metrics['ber'])}"
            )
            outside = metrics["outside_window_percent"]
            annotation += f"\nOutside window: {outside:.2f}%"
            ax.text(0.025, 0.98, annotation, transform=ax.transAxes,
                    ha="left", va="top", fontsize=7.5, linespacing=1.35,
                    bbox={"boxstyle": "round,pad=0.25", "facecolor": "white",
                          "edgecolor": "none", "alpha": 0.9}, zorder=5)

    users_set = sorted({row[0] for row in rows})
    user_description = (
        "single user" if users_set == [1] else
        "two users" if users_set == [2] else "single and two users"
    )
    snr_db = float(metadata.get("snr_db", 20.0))
    first_shared = shared[users_set[0]]
    if mode == "snapshot":
        snapshot_indices = {shared[k].snapshot_channel_index for k in users_set}
        if len(snapshot_indices) == 1:
            source_text = f"fixed channel H_test[{first_shared.snapshot_channel_index}]"
        else:
            source_text = "fixed channels " + ", ".join(
                f"K={k}: H_test[{shared[k].snapshot_channel_index}]" for k in users_set
            )
        sample_counts = {
            (shared[k].symbols.shape[0] - shared[k].n_eval) * shared[k].symbols.shape[2]
            for k in users_set
        }
        sample_text = (
            f"{next(iter(sample_counts)):,} symbols/user" if len(sample_counts) == 1
            else "all saved snapshot symbols"
        )
        context = f"{source_text}; {sample_text}"
    else:
        counts = {shared[k].n_eval for k in users_set}
        count_text = f"{next(iter(counts)):,}" if len(counts) == 1 else "saved"
        context = f"{count_text} evaluation channels; evenly sampled, up to {MAX_ENSEMBLE_POINTS:,} points/panel"
    fig.suptitle(f"Received 16-QAM: {user_description}\n"
                 f"M = {metadata.get('m', 40)} AP | Transmit SNR = {snr_db:g} dB | Oracle block-gain EQ\n"
                 f"{context}", fontsize=11.5, linespacing=1.45,
                 y=1.0 - 0.10 / height)
    footer = "Color: transmitted symbol. Black crosses: ideal 16-QAM. All panels use I, Q in [−1.8, 1.8]."
    if mode == "ensemble":
        footer += "\nEVM, BER and outside-window rates use every saved evaluation symbol, before display subsampling."
    else:
        footer += "\nMetrics use all snapshot samples, including those outside the window. BER = 0: no observed bit errors."
    footer = "\n".join(textwrap.fill(line, width=int(width * 15)) for line in footer.splitlines())
    fig.text(0.59, 0.13 / height, footer, ha="center", va="bottom", fontsize=7.6,
             linespacing=1.4, color="#414853")
    paths = []
    for extension in ("png", "pdf"):
        destination = output_dir / f"{stem}.{extension}"
        fig.savefig(destination, dpi=240, facecolor="white")
        paths.append(destination)
    plt.close(fig)
    return paths


def plot_all(input_dir: str | Path) -> list[Path]:
    """Render all comparisons from saved IQ and return the generated paths."""
    input_dir = Path(input_dir).expanduser().resolve()
    with (input_dir / "metadata.json").open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    output_dir = input_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    resolutions = tuple(int(b) for b in metadata["bits"])
    shared = {users: _read_shared(input_dir, users) for users in sorted(metadata["users"])}
    methods = tuple(metadata.get("methods", DEFAULT_METHODS))
    panels = _load_panels(input_dir, shared, resolutions, methods)
    rows_by_users = {users: [(users, user, method) for user in range(users) for method in methods]
                     for users in shared}
    all_rows = [row for group in rows_by_users.values() for row in group]
    specifications = []
    if all(common.n_eval < len(common.symbols) for common in shared.values()):
        specifications.extend((f"constellation_K{users}", rows, "snapshot")
                              for users, rows in rows_by_users.items())
        specifications.append(("constellation_overview", all_rows, "snapshot"))
    specifications.append(("constellation_ensemble", all_rows, "ensemble"))
    outputs = []
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 9,
                         "pdf.fonttype": 42, "ps.fonttype": 42}):
        for stem, rows, mode in specifications:
            outputs.extend(_draw_figure(output_dir, stem, rows, panels[mode],
                                        shared, metadata, mode))
    metrics_path = output_dir / "constellation_plot_metrics.json"
    metrics_export = {
        "source": "Saved tx_symbols, tx_bits and rx_equalized; no synthetic plot samples.",
        "evm_definition": "100 * sqrt(mean(abs(rx_equalized-tx_symbols)^2) / mean(abs(tx_symbols)^2))",
        "bit_mapping": "I then Q; levels [-3,-1,+1,+3]/sqrt(10) map to [00,01,11,10].",
        "display_window": [-AXIS_LIMIT, AXIS_LIMIT],
        "panels": {
            mode: {
                f"K{k}_user{user + 1}_{method}_b{bits}": panel.metrics
                for (k, user, method, bits), panel in mode_panels.items()
            }
            for mode, mode_panels in panels.items()
        },
    }
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics_export, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    outputs.append(metrics_path)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True,
                        help="Directory containing metadata.json, shared_K*.npz and case IQ files.")
    args = parser.parse_args()
    for path in plot_all(args.input_dir):
        print(path)


if __name__ == "__main__":
    main()
