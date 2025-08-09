import re, numpy as np, matplotlib.pyplot as plt


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
    m = np.isfinite(errs) & np.isfinite(gam) & (gam > 0)
    return errs[m], gam[m]

def kernel_mean_sd_adaptive(gamma, err, eval_gamma=None, k=2000, num_pts=200):
    """
    Nadaraya–Watson with Gaussian kernel, adaptive bandwidth via kNN (linear gamma space).
    k ~ target effective sample size per estimate (set based on your data size; try 500–5000).
    Returns eval_gamma, mean, sd, neff
    """
    gamma = np.asarray(gamma, float)
    err   = np.asarray(err,   float)

    if eval_gamma is None:
        # Evaluate on quantiles to cover the empirical distribution uniformly
        qs = np.linspace(0.01, 0.99, num_pts)
        eval_gamma = np.quantile(gamma, qs)

    # Pre-sort gamma for faster kNN radii via partial selection
    order = np.argsort(gamma)
    g_sorted = gamma[order]
    e_sorted = err[order]

    means = np.full_like(eval_gamma, np.nan, dtype=float)
    sds   = np.full_like(eval_gamma, np.nan, dtype=float)
    neff  = np.zeros_like(eval_gamma, dtype=float)

    # Small jitter to avoid zero bandwidth if many identical gamma
    eps = 1e-12

    for j, gq in enumerate(eval_gamma):
        # Find k-th nearest neighbor distance (absolute, linear scale)
        d = np.abs(g_sorted - gq)
        if k >= len(d):
            # use max distance if k exceeds sample count
            hk = d.max() + eps
        else:
            # nth element in O(n)
            hk = np.partition(d, k)[k] + eps

        # Gaussian kernel with h = hk
        w = np.exp(-0.5 * (d / hk)**2)

        W = w.sum()
        if W <= 0:
            continue

        m = (w * e_sorted).sum() / W
        var = (w * (e_sorted - m)**2).sum() / W
        means[j] = m
        sds[j]   = np.sqrt(max(var, 0.0))

        # effective sample size (for diagnostics)
        neff[j] = W**2 / (w**2).sum()

    return eval_gamma, means, sds, neff

# ---- run on your file ----
log_path = "35fc9021_denoiser_l2_norm_squared.log"
errors, gammas = parse_error_gamma_log(log_path)

# Choose k ~ few thousand (tune as you like); more k => smoother
g_eval, m, s, neff = kernel_mean_sd_adaptive(gammas, errors, k=2000, num_pts=200)

# ---- plot ----
plt.figure(figsize=(8,5))
plt.plot(g_eval, m, label="Kernel mean (adaptive kNN)")
plt.fill_between(g_eval, m - s, m + s, alpha=0.2, label="±1 SD (kernel-weighted)")
plt.xlabel("gamma")
plt.ylabel("squared error")
plt.title("Squared error vs gamma — kernel mean ± SD")
# If you prefer log x for readability (doesn't affect the fit):
# plt.xscale("log")
plt.legend()
plt.tight_layout()
plt.show()

print(f"Parsed pairs: {len(errors):,}")
print(f"Gamma range: [{gammas.min():.6g}, {gammas.max():.6g}]")
print(f"Median effective N per estimate: {np.nanmedian(neff):.0f}")
