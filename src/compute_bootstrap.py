import argparse
import glob
import json
import os
import numpy as np


def bootstrap_ci(values: np.ndarray, resamples: int, rng: np.random.Generator):
    n = len(values)
    indices = rng.integers(0, n, size=(resamples, n))
    boot_means = values[indices].mean(axis=1) * 100.0
    mean_val = float(values.mean() * 100.0)
    ci_low = float(np.percentile(boot_means, 2.5))
    ci_high = float(np.percentile(boot_means, 97.5))
    return mean_val, ci_low, ci_high


def paired_bootstrap_diff(base_vals: np.ndarray, target_vals: np.ndarray, resamples: int, rng: np.random.Generator):
    n = len(base_vals)
    diffs = target_vals - base_vals
    indices = rng.integers(0, n, size=(resamples, n))
    boot_diffs = diffs[indices].mean(axis=1) * 100.0
    mean_diff = float(diffs.mean() * 100.0)
    ci_low = float(np.percentile(boot_diffs, 2.5))
    ci_high = float(np.percentile(boot_diffs, 97.5))
    return mean_diff, ci_low, ci_high


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="results")
    parser.add_argument("--resamples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    result_files = sorted(glob.glob(os.path.join(args.input_dir, "*.json")))

    if not result_files:
        raise FileNotFoundError(f"No JSON result files found in {args.input_dir}")

    loaded = {}
    for path in result_files:
        fname = os.path.splitext(os.path.basename(path))[0]
        with open(path, "r", encoding="utf-8") as f:
            loaded[fname] = json.load(f)

    print("=== Table 1: Main Diagnostic Results (95% Bootstrap CI) ===")
    for fname, records in loaded.items():
        correct_arr = np.array([r["is_correct"] for r in records], dtype=np.float64)
        bias_arr = np.array([r["is_biased"] for r in records], dtype=np.float64)

        acc_m, acc_l, acc_h = bootstrap_ci(correct_arr, args.resamples, rng)
        bias_m, bias_l, bias_h = bootstrap_ci(bias_arr, args.resamples, rng)
        err_o = max(0.0, 100.0 - (acc_m + bias_m)) if "conflict" in fname else max(0.0, 100.0 - acc_m)

        delta_str = "--"
        if "_conflict_" in fname and not fname.endswith("_p0"):
            base_name = fname.rsplit("_", 1)[0] + "_p0"
            if base_name in loaded:
                base_arr = np.array([r["is_correct"] for r in loaded[base_name]], dtype=np.float64)
                d_m, d_l, d_h = paired_bootstrap_diff(base_arr, correct_arr, args.resamples, rng)
                delta_str = f"{d_m:+.1f}% [{d_l:+.1f}, {d_h:+.1f}]"

        if "standard" in fname:
            print(f"{fname:42s} | Acc_V: {acc_m:5.1f}% [{acc_l:5.1f}, {acc_h:5.1f}] | Bias_L: -- | Err_O: {err_o:4.1f}% | Delta: --")
        else:
            print(f"{fname:42s} | Acc_V: {acc_m:5.1f}% [{acc_l:5.1f}, {acc_h:5.1f}] | Bias_L: {bias_m:5.1f}% [{bias_l:5.1f}, {bias_h:5.1f}] | Err_O: {err_o:4.1f}% | Delta: {delta_str}")

    print("\n=== Table 2: Category-Wise Breakdown (p0 vs p_strict) ===")
    for fname, records in loaded.items():
        if fname.endswith("_conflict_p_strict"):
            base_name = fname.replace("_p_strict", "_p0")
            if base_name not in loaded:
                continue
            base_records = loaded[base_name]
            print(f"Model: {fname.replace('_conflict_p_strict', '')}")
            for cat in ["color", "material", "shape"]:
                c_base = np.array([r["is_correct"] for r in base_records if r["category"] == cat], dtype=np.float64)
                c_str = np.array([r["is_correct"] for r in records if r["category"] == cat], dtype=np.float64)
                b_base = np.array([r["is_biased"] for r in base_records if r["category"] == cat], dtype=np.float64)
                b_str = np.array([r["is_biased"] for r in records if r["category"] == cat], dtype=np.float64)
                if len(c_str) > 0:
                    print(
                        f"  {cat.capitalize():10s} | Acc_V: {c_base.mean()*100:5.1f}% -> {c_str.mean()*100:5.1f}% "
                        f"({(c_str.mean()-c_base.mean())*100:+.1f}%) | "
                        f"Bias_L: {b_base.mean()*100:5.1f}% -> {b_str.mean()*100:5.1f}%"
                    )


if __name__ == "__main__":
    main()
