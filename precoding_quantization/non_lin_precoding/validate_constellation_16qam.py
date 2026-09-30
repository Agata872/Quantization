"""Independently validate saved 16-QAM constellation experiments using NumPy.

Usage: python validate_constellation_16qam.py --input-dir PATH
No simulation module is imported. Exit status is nonzero if any check fails.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

PT = 40.0
NOISE_VARIANCE = 0.4
PAM = np.array([-3.0, -1.0, 1.0, 3.0]) / math.sqrt(10.0)
GRAY = np.array([[0, 0], [0, 1], [1, 1], [1, 0]], dtype=np.uint8)
ERFC = np.vectorize(math.erfc, otypes=[np.float64])


class Validator:
    def __init__(self):
        self.checks = []
        self.max_errors = {}

    def check(self, name, passed, detail=None):
        item = {"name": name, "passed": bool(passed)}
        if detail is not None:
            item["detail"] = detail
        self.checks.append(item)

    def close(self, name, actual, expected, atol=2e-5, rtol=2e-5):
        actual, expected = np.asarray(actual), np.asarray(expected)
        if actual.shape != expected.shape:
            self.check(name, False, {"actual_shape": list(actual.shape),
                                     "expected_shape": list(expected.shape)})
            return
        error = np.abs(actual - expected)
        bound = atol + rtol * np.abs(expected)
        maximum = float(np.max(error)) if error.size else 0.0
        ratios = np.divide(error, bound, out=np.zeros_like(error), where=bound > 0)
        ratios = np.where((bound == 0) & (error > 0), np.inf, ratios)
        ratio = float(np.max(ratios)) if error.size else 0.0
        self.max_errors[name] = {"max_abs": maximum if math.isfinite(maximum) else None,
                                 "max_tolerance_ratio": ratio if math.isfinite(ratio) else None}
        self.check(name, np.all(np.isfinite(error)) and np.all(error <= bound))

    def finite(self, prefix, arrays):
        for key, value in arrays.items():
            if np.issubdtype(value.dtype, np.number):
                self.check(f"{prefix}/{key}/finite", np.isfinite(value).all())


def load_npz(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def demap(z):
    """Nearest Gray-mapped PAM-4 decision per I/Q dimension."""
    i = np.argmin(np.abs(z.real[..., None] - PAM), axis=-1)
    q = np.argmin(np.abs(z.imag[..., None] - PAM), axis=-1)
    return np.concatenate((GRAY[i], GRAY[q]), axis=-1)


def analytical_ber(z_noiseless, s, gain, noise_variance=NOISE_VARIANCE):
    sd = np.sqrt(noise_variance / (2.0 * np.abs(gain) ** 2))[..., None]
    d = 2.0 / math.sqrt(10.0)

    def q(x):
        return 0.5 * ERFC(x / math.sqrt(2.0))

    def dim_ber(u, a):
        sign = q(np.sign(a) * u / sd)
        inner = q((d - u) / sd) + q((d + u) / sd)
        outer = q((-d - u) / sd) - q((d - u) / sd)
        return 0.5 * (sign + np.where(np.abs(a) < d, inner, outer))

    return (0.5 * (dim_ber(z_noiseless.real, s.real)
                   + dim_ber(z_noiseless.imag, s.imag))).mean(axis=-1)


def validate_shared(v, data, k, metadata):
    ns, m = int(metadata["ns"]), int(metadata["m"])
    noise_variance = float(metadata["noise_variance_complex"])
    start = int(metadata["start"])
    prefix = f"shared_K{k}"
    required = ("H", "tx_symbols", "tx_bits", "noise", "channel_indices",
                "n_eval", "snapshot_channel_index", "noise_variance")
    missing = sorted(set(required) - set(data))
    v.check(f"{prefix}/required_arrays", not missing, {"missing": missing})
    if missing:
        raise ValueError(f"missing shared arrays: {missing}")
    v.finite(prefix, data)
    h, s, bits, noise = (data[key] for key in ("H", "tx_symbols", "tx_bits", "noise"))
    count = h.shape[0]
    shapes = {"H": (count, m, k), "tx_symbols": (count, k, ns),
              "tx_bits": (count, k, ns, 4), "noise": (count, k, ns),
              "channel_indices": (count,)}
    shapes_ok = all(data[key].shape == shape for key, shape in shapes.items())
    for key, shape in shapes.items():
        v.check(f"{prefix}/{key}/shape", data[key].shape == shape,
                {"actual": list(data[key].shape), "expected": list(shape)})
    if not shapes_ok:
        raise ValueError("incompatible shared shapes")
    n_eval = int(data["n_eval"].item())
    snapshot_index = int(data["snapshot_channel_index"].item())
    v.check(f"{prefix}/n_eval", 0 < n_eval <= count and n_eval == int(metadata["n_eval"]),
            {"n_eval": n_eval, "n_snapshot_blocks": count - n_eval,
             "evaluation_symbols_per_user": n_eval * ns})
    v.check(f"{prefix}/snapshot_count", count - n_eval == int(metadata["snapshot_blocks"]))
    v.check(f"{prefix}/snapshot_index", snapshot_index == start == int(metadata["snapshot_channel_index"]))
    v.close(f"{prefix}/noise_variance_metadata", data["noise_variance"], np.asarray(noise_variance),
            atol=1e-12, rtol=1e-12)
    v.check(f"{prefix}/evaluation_indices",
            np.array_equal(data["channel_indices"][:n_eval], np.arange(start, start + n_eval)))
    v.check(f"{prefix}/snapshot_indices",
            np.all(data["channel_indices"][n_eval:] == snapshot_index))
    v.close(f"{prefix}/snapshot_channel", h[n_eval:],
            np.broadcast_to(h[0], h[n_eval:].shape), atol=0.0, rtol=0.0)
    v.check(f"{prefix}/tx_bits_binary", np.all((bits == 0) | (bits == 1)))
    # Inverse Gray map: 00 -> -3, 01 -> -1, 11 -> +1, 10 -> +3.
    lookup = np.array([-3.0, -1.0, 3.0, 1.0]) / math.sqrt(10.0)
    decoded = lookup[2 * bits[..., 0] + bits[..., 1]] + 1j * lookup[2 * bits[..., 2] + bits[..., 3]]
    v.close(f"{prefix}/gray_mapping", s, decoded, atol=1e-7, rtol=1e-7)
    v.check(f"{prefix}/gray_roundtrip", np.array_equal(demap(s), bits))
    eval_noise = noise[:n_eval].astype(np.complex128)
    energy = float(np.mean(np.abs(eval_noise) ** 2))
    relative_error = abs(energy / noise_variance - 1.0)
    tolerance = max(0.03, 5.0 / math.sqrt(eval_noise.size))
    v.check(f"{prefix}/noise_variance", relative_error <= tolerance,
            {"measured": energy, "target": noise_variance,
             "relative_error": relative_error, "relative_tolerance": tolerance,
             "n_complex_samples": int(eval_noise.size)})
    for component_name, component in (("real", eval_noise.real), ("imag", eval_noise.imag)):
        bound = 5.0 * math.sqrt(noise_variance / (2.0 * component.size))
        v.check(f"{prefix}/noise_{component_name}_mean", abs(float(component.mean())) <= bound,
                {"mean": float(component.mean()), "five_sigma_bound": bound})
        relative_component_error = abs(float(np.mean(component ** 2)) / (noise_variance / 2.0) - 1.0)
        component_tolerance = max(0.03, 5.0 * math.sqrt(2.0 / component.size))
        v.check(f"{prefix}/noise_{component_name}_variance", relative_component_error <= component_tolerance,
                {"relative_error": relative_component_error, "relative_tolerance": component_tolerance})
    normalized_cross = float(np.mean(eval_noise.real * eval_noise.imag)) / (noise_variance / 2.0)
    v.check(f"{prefix}/noise_iq_cross_moment", abs(normalized_cross) <= 5.0 / math.sqrt(eval_noise.size),
            {"normalized_cross_moment": normalized_cross})
    return n_eval


def validate_case(v, shared, data, k, bits, method, n_eval, metadata):
    ns, m = int(metadata["ns"]), int(metadata["m"])
    pt, noise_variance = float(metadata["pt"]), float(metadata["noise_variance_complex"])
    prefix = f"K{k}_b{bits}_{method}"
    required = ("tx_samples", "dac_indices", "levels", "power_scale", "rx_noiseless",
                "rx_raw", "rx_equalized", "gain", "bit_errors", "symbol_errors", "evm_rms", "ber_analytic")
    missing = sorted(set(required) - set(data))
    v.check(f"{prefix}/required_arrays", not missing, {"missing": missing})
    if missing:
        raise ValueError(f"missing case arrays: {missing}")
    v.finite(prefix, data)
    h = shared["H"].astype(np.complex128)
    s = shared["tx_symbols"].astype(np.complex128)
    count = h.shape[0]
    shapes = {"tx_samples": (count, m, ns), "dac_indices": (count, m, ns, 2),
              "levels": (2 ** bits,), "power_scale": (count,),
              "rx_noiseless": (count, k, ns), "rx_raw": (count, k, ns),
              "rx_equalized": (count, k, ns), "gain": (count, k),
              "bit_errors": (count, k), "symbol_errors": (count, k),
              "evm_rms": (count, k), "ber_analytic": (count, k)}
    shapes_ok = True
    for key, shape in shapes.items():
        ok = data[key].shape == shape
        v.check(f"{prefix}/{key}/shape", ok,
                {"actual": list(data[key].shape), "expected": list(shape)})
        shapes_ok = shapes_ok and ok
    if not shapes_ok:
        raise ValueError("incompatible case shapes")
    index, levels = data["dac_indices"], data["levels"]
    valid_index = (np.issubdtype(index.dtype, np.integer)
                   and np.all(index >= 0) and np.all(index < 2 ** bits))
    v.check(f"{prefix}/dac_indices_range", valid_index)
    v.check(f"{prefix}/dac_levels_sorted_distinct", np.all(np.diff(levels) > 0))
    if not valid_index:
        raise ValueError("invalid DAC indices")
    dac = levels[index[..., 0]].astype(np.float64) + 1j * levels[index[..., 1]].astype(np.float64)
    tx = data["tx_samples"].astype(np.complex128)
    scale = data["power_scale"].astype(np.float64)
    v.check(f"{prefix}/positive_power_scale", np.all(scale > 0))
    v.close(f"{prefix}/dac_reconstruction", tx, dac * scale[:, None, None], atol=2e-6, rtol=2e-6)
    if method == "wmmse" and "precoder_matrix" in data:
        matrix = data["precoder_matrix"].astype(np.complex128)
        v.check(f"{prefix}/precoder_matrix_shape", matrix.shape == (count, m, k))
        v.close(f"{prefix}/precoder_matrix_power", (np.abs(matrix) ** 2).sum(axis=(1, 2)),
                np.full(count, pt), atol=2e-5, rtol=2e-5)
        thresholds = (levels[:-1].astype(np.float64) + levels[1:]) / 2.0
        maximum_suboptimal_distance = 0.0
        for start in range(0, count, 128):
            stop = min(start + 128, count)
            vb = matrix[start:stop]
            row_norm = np.maximum(np.linalg.norm(vb, axis=-1, keepdims=True), 1e-12)
            linear = (vb @ s[start:stop]) / row_norm
            for dim, component in enumerate((linear.real, linear.imag)):
                nearest = levels[np.searchsorted(thresholds, component, side="left")]
                selected = levels[index[start:stop, ..., dim]]
                excess = np.abs(component - selected) - np.abs(component - nearest)
                maximum_suboptimal_distance = max(maximum_suboptimal_distance, float(np.max(excess)))
        v.check(f"{prefix}/wmmse_normalize_then_dac", maximum_suboptimal_distance <= 2e-5,
                {"max_excess_quantization_distance": maximum_suboptimal_distance})
    dac_power = (np.abs(dac) ** 2).sum(axis=1).mean(axis=-1)
    v.close(f"{prefix}/power_scale", scale, np.sqrt(pt / dac_power), atol=2e-6, rtol=2e-6)
    power = (np.abs(tx) ** 2).sum(axis=1).mean(axis=-1)
    v.close(f"{prefix}/transmit_power", power, np.full(count, pt), atol=2e-5, rtol=2e-6)
    propagated = np.einsum("cmk,cmt->ckt", h, tx, optimize=True)
    noiseless = data["rx_noiseless"].astype(np.complex128)
    raw = data["rx_raw"].astype(np.complex128)
    equalized = data["rx_equalized"].astype(np.complex128)
    v.close(f"{prefix}/channel_transpose_without_conjugation", noiseless, propagated)
    v.close(f"{prefix}/shared_awgn", raw, noiseless + shared["noise"])
    v.close(f"{prefix}/noise_disabled_reconstruction", raw - shared["noise"], propagated)
    gain = (propagated * s.conj()).sum(axis=-1) / (np.abs(s) ** 2).sum(axis=-1)
    v.check(f"{prefix}/nonzero_gain", np.all(np.abs(gain) > 1e-12))
    v.close(f"{prefix}/oracle_gain", data["gain"], gain)
    v.close(f"{prefix}/equalization", equalized, raw / gain[..., None])
    detected = demap(equalized)
    if "rx_bits" in data:
        v.check(f"{prefix}/rx_bits", np.array_equal(data["rx_bits"], detected))
    if "tx_power" in data:
        v.close(f"{prefix}/stored_tx_power", data["tx_power"], power)
    if bits == 1:
        v.close(f"{prefix}/one_bit_levels", levels, np.array([-1.0, 1.0]) * math.sqrt(pt / (2 * m)),
                atol=1e-7, rtol=1e-7)
    errors = detected != shared["tx_bits"]
    bit_errors = errors.sum(axis=(-1, -2))
    symbol_errors = np.any(errors, axis=-1).sum(axis=-1)
    v.check(f"{prefix}/bit_errors", np.array_equal(data["bit_errors"], bit_errors))
    v.check(f"{prefix}/symbol_errors", np.array_equal(data["symbol_errors"], symbol_errors))
    evm = np.sqrt((np.abs(equalized - s) ** 2).sum(axis=-1) / (np.abs(s) ** 2).sum(axis=-1))
    v.close(f"{prefix}/evm_rms", data["evm_rms"], evm, atol=2e-6, rtol=2e-5)
    theory = analytical_ber(propagated / gain[..., None], s, gain, noise_variance)
    v.close(f"{prefix}/analytic_ber", data["ber_analytic"], theory, atol=2e-6, rtol=2e-5)
    v.check(f"{prefix}/analytic_ber_probability", np.all((data["ber_analytic"] >= 0) & (data["ber_analytic"] <= 1)))
    # A fixed H and repeated joint symbol must have identical noiseless receive IQ
    # inside a block. Across blocks, power normalization can change the scale.
    max_repeat_error, comparisons = 0.0, 0
    for block in range(n_eval, count):
        joint = np.concatenate((s[block].real.T, s[block].imag.T), axis=1)
        _, inverse = np.unique(joint, axis=0, return_inverse=True)
        for label in np.unique(inverse):
            positions = np.flatnonzero(inverse == label)
            if len(positions) > 1:
                values = noiseless[block][:, positions]
                max_repeat_error = max(max_repeat_error, float(np.max(np.abs(values - values[:, :1]))))
                comparisons += len(positions) - 1
    v.check(f"{prefix}/deterministic_repeated_joint_symbols", max_repeat_error <= 2e-5,
            {"max_abs": max_repeat_error, "comparisons": comparisons})
    # In a K=1 one-bit linear precoder, positive radial rescaling of a symbol
    # leaves all DAC signs unchanged. Inner/outer diagonal QAM points therefore
    # collide before AWGN; this is a useful sanity check of actual quantized IQ.
    if k == 1 and bits == 1 and method == "wmmse":
        collision_error, collision_count = 0.0, 0
        for block in range(n_eval, count):
            sb = s[block, 0]
            diagonal = np.isclose(np.abs(sb.real), np.abs(sb.imag), atol=1e-7)
            for si in (-1, 1):
                for sq in (-1, 1):
                    positions = np.flatnonzero(diagonal & (np.sign(sb.real) == si) & (np.sign(sb.imag) == sq))
                    if len(positions) > 1:
                        values = noiseless[block, 0, positions]
                        collision_error = max(collision_error, float(np.max(np.abs(values - values[0]))))
                        collision_count += len(positions) - 1
        v.check(f"{prefix}/one_bit_diagonal_amplitude_collision", collision_error <= 2e-5,
                {"max_abs": collision_error, "comparisons": collision_count})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    args = parser.parse_args()
    v = Validator()
    metadata = json.loads((args.input_dir / "metadata.json").read_text())
    v.check("metadata/status", metadata.get("status") in ("simulated", "complete"))
    v.check("metadata/antennas", metadata["m"] == 40)
    v.check("metadata/transmit_power", metadata["pt"] == PT)
    v.check("metadata/configuration", int(metadata["ns"]) > 0 and int(metadata["n_eval"]) > 0
            and int(metadata["start"]) >= 0 and int(metadata["snapshot_blocks"]) >= 0
            and set(metadata["users"]).issubset({1, 2}) and len(metadata["users"]) > 0
            and set(metadata["bits"]).issubset({1, 2, 3}) and len(metadata["bits"]) > 0)
    v.close("metadata/snr_noise_variance", np.asarray(metadata["noise_variance_complex"]),
            np.asarray(metadata["pt"] / 10 ** (metadata["snr_db"] / 10)), atol=1e-12, rtol=1e-12)
    # Runs made before the baseline became selectable have no "methods" entry.
    methods = metadata.get("methods", ["gnn", "wmmse"])
    v.check("metadata/methods", methods[0] == "gnn" and len(methods) == 2
            and methods[1] in ("wmmse", "ide_wf", "ide_block", "ide_cal"), {"methods": methods})
    expected_files = {f"K{k}_b{b}_{method}.npz" for k in metadata["users"]
                      for b in metadata["bits"] for method in methods}
    v.check("metadata/case_files", set(metadata["case_files"]) == expected_files)
    for k in metadata["users"]:
        shared_path = args.input_dir / f"shared_K{k}.npz"
        try:
            shared = load_npz(shared_path)
            n_eval = validate_shared(v, shared, k, metadata)
        except Exception as error:
            v.check(f"shared_K{k}/read_or_validate", False, str(error))
            continue
        for bits in metadata["bits"]:
            for method in methods:
                path = args.input_dir / f"K{k}_b{bits}_{method}.npz"
                try:
                    validate_case(v, shared, load_npz(path), k, bits, method, n_eval, metadata)
                except Exception as error:
                    v.check(f"{path.stem}/read_or_validate", False, str(error))
    failed = [item for item in v.checks if not item["passed"]]
    report = {"passed": not failed, "n_checks": len(v.checks), "n_failed": len(failed),
              "independent_implementation": "NumPy + Python standard library; no simulator imports",
              "transmit_power": metadata["pt"], "noise_variance": metadata["noise_variance_complex"],
              "checks": v.checks, "max_errors": v.max_errors}
    output = args.input_dir / "validation.json"
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"{len(v.checks) - len(failed)}/{len(v.checks)} checks passed; report: {output}")
    for item in failed:
        print(f"FAIL {item['name']}: {item.get('detail', '')}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
