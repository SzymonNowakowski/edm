import re, numpy as np, matplotlib.pyplot as plt

# ---------- parsing ----------
def parse_error_sigma_log(path):
    pat = re.compile(r'([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)\s*:\s*([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)')
    errs, sigmas = [], []
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            for m in pat.finditer(line):
                errs.append(float(m.group(1)))
                sigmas.append(float(m.group(2)))
    errs = np.asarray(errs, float)
    sigmas = np.asarray(sigmas, float)
    m = np.isfinite(errs) & np.isfinite(sigmas)
    return errs[m], sigmas[m]

# ---------- Karras/EDM sigma steps (centers) ----------
def make_sigma_steps(sigma_min, sigma_max, num_steps, rho):
    step_indices = np.arange(num_steps, dtype=np.float64)
    return (sigma_max**(1.0/rho) + step_indices/(num_steps-1) * (sigma_min**(1.0/rho) - sigma_max**(1.0/rho)))**rho

# ---------- binning over Karras sigma schedule (ascending centers & edges) ----------
def bin_stats(errors, sigmas, sigma_min, sigma_max, num_steps=18, rho=7.0):
    # 1) Build Karras centers, then sort ascending (small -> large)
    centers_desc = make_sigma_steps(sigma_min, sigma_max, num_steps, rho)
    sigma_centers = np.sort(centers_desc)
    n_bins = len(sigma_centers)

    # 2) Build edges from midpoints, clamped to [sigma_min, sigma_max]
    midpoints = 0.5 * (sigma_centers[:-1] + sigma_centers[1:])
    sigma_edges = np.empty(n_bins + 1, dtype=float)
    sigma_edges[0]      = sigma_min
    sigma_edges[1:-1]   = midpoints
    sigma_edges[-1]     = sigma_max
    sigma_edges = np.maximum.accumulate(sigma_edges)  # ensure non-decreasing

    # 3) Per-bin stats (raw only)
    counts   = np.zeros(n_bins, dtype=int)
    raw_mean = np.full(n_bins, np.nan, dtype=float)
    raw_sd   = np.full(n_bins, np.nan, dtype=float)

    # 4) Weights based on current σ and next larger σ (previous in generation time):
    #    w_i = (1/σ_i^2 - 1/σ_{i+1}^2) / (σ_i^2 + 1),  for i = 0..n-2;  w_{n-1} = NaN
    w = np.full(n_bins, np.nan, dtype=float)
    for i in range(n_bins - 1):
        s_curr = sigma_centers[i]
        s_prev_time = sigma_centers[i+1]  # previous in time = next larger sigma
        if s_curr > 0.0 and s_prev_time > 0.0:
            w[i] = (1.0/(s_curr**2) - 1.0/(s_prev_time**2)) / (s_curr**2 + 1.0)

    # 5) Bin data using (edge_i, edge_{i+1}] (include min in first bin)
    for idx in range(n_bins):
        lo, hi = sigma_edges[idx], sigma_edges[idx+1]
        if idx == 0:
            mask = (sigmas >= lo) & (sigmas <= hi)
        else:
            mask = (sigmas >  lo) & (sigmas <= hi)

        cnt = int(mask.sum())
        counts[idx] = cnt
        if cnt == 0:
            continue

        errs_bin = errors[mask]
        raw_mean[idx] = float(errs_bin.mean())
        raw_sd[idx]   = float(errs_bin.std(ddof=1)) if cnt > 1 else 0.0

    # 6) Weighted values = raw * w  (no normalization)
    weighted_mean = np.full(n_bins, np.nan, dtype=float)
    weighted_sd   = np.full(n_bins, np.nan, dtype=float)
    has_data = counts > 0
    use = has_data & np.isfinite(w)
    weighted_mean[use] = raw_mean[use] * w[use]
    weighted_sd[use]   = raw_sd[use]   * w[use]
    total_weight_used = np.nansum(w[use])

    return {
        "sigma_centers": sigma_centers,
        "sigma_edges": sigma_edges,
        "counts": counts,
        "raw_mean": raw_mean, "raw_sd": raw_sd,
        "w": w,
        "weighted_mean": weighted_mean, "weighted_sd": weighted_sd,
        "total_weight_used": float(total_weight_used),
    }

# ---------- run ----------
log_path = "35fc9021_denoiser_l2_norm_squared.log"
errors, sigmas = parse_error_sigma_log(log_path)

sigma_min = 0.002
sigma_max = 80.0
rho = 7.0
num_steps = 18

print(f"Detected σ range in log: [{sigmas.min():.6f}, {sigmas.max():.6f}]")
stats = bin_stats(errors, sigmas, sigma_min=sigma_min, sigma_max=sigma_max,
                  num_steps=num_steps, rho=rho)

sigma_centers = stats["sigma_centers"]
mask_data = stats["counts"] > 0
mask_weighted = mask_data & np.isfinite(stats["w"])

# Print centers, weights, and counts for verification
print("Sigma bin centers (ascending), weights (prev-in-time), and counts:")
for i, (c, w_i, cnt) in enumerate(zip(sigma_centers, stats["w"], stats["counts"])):
    print(f"[{i:02d}] center={c:.6f}  w={w_i!s:>14}  count={cnt}")

# ---------- plotting ----------
fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
x_vals = sigma_centers

# 1) raw errors (log scale)
axes[0].errorbar(x_vals[mask_data], stats["raw_mean"][mask_data],
                 yerr=stats["raw_sd"][mask_data],
                 fmt='o-', capsize=3, linewidth=1)
axes[0].set_ylabel("mean ± SD (raw error)")
axes[0].set_title("Raw denoising errors per σ bin (log scale)")
axes[0].set_yscale("log")
axes[0].grid(True)

# 2) raw errors (linear scale)
axes[1].errorbar(x_vals[mask_data], stats["raw_mean"][mask_data],
                 yerr=stats["raw_sd"][mask_data],
                 fmt='o-', capsize=3, linewidth=1)
axes[1].set_ylabel("mean ± SD (raw error)")
axes[1].set_title("Raw denoising errors per σ bin (linear scale)")
axes[1].grid(True)

# 3) weighted errors (log scale)  -- uses raw*w
axes[2].errorbar(x_vals[mask_weighted], stats["weighted_mean"][mask_weighted],
                 yerr=stats["weighted_sd"][mask_weighted],
                 fmt='o-', capsize=3, linewidth=1)
axes[2].set_xlabel("σ (bin centers, Karras schedule)")
axes[2].set_ylabel("mean ± SD (weighted error)")
axes[2].set_title(r"Weighted denoising errors per σ bin (log scale)")
axes[2].set_yscale("log")
axes[2].grid(True)

plt.tight_layout()
plt.show()

print(f"Parsed pairs: {len(errors):,}")
print(f"Bins with data: {mask_data.sum()}/{len(sigma_centers)}")
print(f"Bins with data (weighted): {mask_weighted.sum()}/{len(sigma_centers)}")
print(f"Total weight used: {stats['total_weight_used']:.12f}")
