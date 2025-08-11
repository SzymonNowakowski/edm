import re, numpy as np, matplotlib.pyplot as plt

# ---------- parsing ----------
def parse_error_gamma_log(path):
    pat = re.compile(r'([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)\s*:\s*([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)')
    errs, gam = [], []
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            for m in pat.finditer(line):
                errs.append(float(m.group(1)))
                gam .append(float(m.group(2)))
    errs = np.asarray(errs, float)
    gam  = np.asarray(gam,  float)
    m = np.isfinite(errs) & np.isfinite(gam)
    return errs[m], gam[m]

# ---------- binning over (t-0.5, t+0.5] for integer t ----------
def bin_stats(errors, gammas, t_min=0, t_max=75):
    """
    For each integer t in [t_min, t_max], collect samples with gamma in (t-0.5, t+0.5].
    Returns dict with per-bin stats for:
      - raw mean/sd
      - normalized mean/sd (divide each error by t^2+1)
      - weighted mean/sd (normalized * w_t where w_t = (1/t^2 - 1/(t+1)^2)/(t^2+1), only for t>=1)
    Also returns total weight actually used in the weighted plot.
    """
    t_vals = np.arange(t_min, t_max + 1, dtype=int)
    n_bins = len(t_vals)

    counts = np.zeros(n_bins, dtype=int)
    raw_mean = np.full(n_bins, np.nan, dtype=float)
    raw_sd   = np.full(n_bins, np.nan, dtype=float)

    norm_mean = np.full(n_bins, np.nan, dtype=float)
    norm_sd   = np.full(n_bins, np.nan, dtype=float)

    # weights per t (define but only valid for t>=1)
    w = np.full(n_bins, np.nan, dtype=float)
    valid_w = t_vals >= 1
    w[valid_w] = (1.0/(t_vals[valid_w]**2) - 1.0/((t_vals[valid_w]+1.0)**2)) / (t_vals[valid_w]**2 + 1.0)

    for idx, t in enumerate(t_vals):
        lo, hi = t - 0.5, t + 0.5
        mask = (gammas > lo) & (gammas <= hi)
        cnt = int(mask.sum())
        counts[idx] = cnt
        if cnt == 0:
            continue

        errs_bin = errors[mask]

        # raw
        raw_mean[idx] = float(errs_bin.mean())
        raw_sd[idx]   = float(errs_bin.std(ddof=1)) if cnt > 1 else 0.0

        # normalized by t^2+1
        denom = t*t + 1.0
        norm_bin = errs_bin / denom
        norm_mean[idx] = float(norm_bin.mean())
        norm_sd[idx]   = float(norm_bin.std(ddof=1)) if cnt > 1 else 0.0

    # weighted = normalized * w_t (only where weight is defined and bin has data)
    weighted_mean = np.full(n_bins, np.nan, dtype=float)
    weighted_sd   = np.full(n_bins, np.nan, dtype=float)
    has_data = counts > 0
    use = has_data & valid_w
    weighted_mean[use] = norm_mean[use] * w[use]
    weighted_sd[use]   = norm_sd[use]   * w[use]  # scale SD by the same constant factor

    total_weight_used = np.nansum(w[use])  # sum of weights over bins that had data

    return {
        "t": t_vals,
        "counts": counts,
        "raw_mean": raw_mean, "raw_sd": raw_sd,
        "norm_mean": norm_mean, "norm_sd": norm_sd,
        "w": w,
        "weighted_mean": weighted_mean, "weighted_sd": weighted_sd,
        "total_weight_used": float(total_weight_used),
    }

# ---------- run ----------
log_path = "35fc9021_denoiser_l2_norm_squared.log"
errors, gammas = parse_error_gamma_log(log_path)

stats = bin_stats(errors, gammas, t_min=0, t_max=75)

t = stats["t"]
mask_data = stats["counts"] > 0
mask_weighted = mask_data & (t >= 1)

fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)

# 1) raw errors
axes[0].errorbar(t[mask_data], stats["raw_mean"][mask_data], yerr=stats["raw_sd"][mask_data],
                 fmt='o-', capsize=3, linewidth=1)
axes[0].set_ylabel("mean ± SD (raw error)")
axes[0].set_title("Raw errors per integer t  (bins: (t-0.5, t+0.5])")
axes[0].grid(True)

# 2) normalized errors
axes[1].errorbar(t[mask_data], stats["norm_mean"][mask_data], yerr=stats["norm_sd"][mask_data],
                 fmt='o-', capsize=3, linewidth=1)
axes[1].set_ylabel("mean ± SD (error / (t²+1))")
axes[1].set_title("Normalized errors per integer t")
axes[1].grid(True)

# 3) weighted errors (normalized * w_t)
axes[2].errorbar(t[mask_weighted], stats["weighted_mean"][mask_weighted], yerr=stats["weighted_sd"][mask_weighted],
                 fmt='o-', capsize=3, linewidth=1)
axes[2].set_xlabel("t (integer index)")
axes[2].set_ylabel("mean ± SD (weighted)")
axes[2].set_title(r"Weighted errors per integer t  (weight $w_t=\frac{1/t^2 - 1/(t+1)^2}{t^2+1}$)")
axes[2].set_yscale("log")  # <-- log scale here
axes[2].grid(True)

plt.tight_layout()
plt.show()

print(f"Parsed pairs: {len(errors):,}")
print(f"Bins with data: {mask_data.sum()}/{len(t)}")
print(f"Bins with data & t>=1 (weighted): {mask_weighted.sum()}/{len(t)}")
print(f"Total weight used (sum of w_t over plotted bins): {stats['total_weight_used']:.12f}")
