"""
EIV MSE Bound Experiment — Real Estate Data
============================================

WHAT THIS EXPERIMENT DOES
--------------------------
We validate the theoretical MSE bound from the EIV (Errors-in-Variables) report:

    MSE(n, m) ≈ σ²_ε + A/n + B/(n+m)

where:
  n = number of labeled samples  (have both Y and W)
  m = number of unlabeled samples  (have W and X, but not Y)
  σ²_ε = irreducible noise in Y
  A = finite-sample variance of the Stage-2 estimator  (~σ²_ε · cond(Σ_W̄))
  B = residual bias from imperfect denoising in Stage-1  (~σ²_W · ||β||²)

WHY SEMI-SYNTHETIC
------------------
The real estate dataset contains housing prices and POI (point-of-interest)
density features for 80 sectors over 67 months in China.

We tried using raw price columns as W, but the natural data does NOT satisfy
the classical EIV assumption:  noise(W) ⊥ Y.
Every price variable in the dataset is driven by the same latent sector quality,
so the "noise" in a transaction price is correlated with the outcome — EIV
two-stage actually performs *worse* than OLS on raw data.

To properly validate the theoretical bound, we use a semi-synthetic approach:
  - X  : real data — 5 principal components of 7 POI density features.
          These capture the real covariance structure of sector characteristics.
  - W̄  : synthetic latent = X @ G_true  (a linear function of real X)
  - W  : noisy observed proxy = W̄ + N(0, σ_W)
  - Y  : outcome = W̄ @ β_true + N(0, σ_ε)

This construction guarantees noise(W) ⊥ Y by design, so we can cleanly
verify when and how much unlabeled data helps.

THE TWO-STAGE EIV ESTIMATOR
----------------------------
  Stage 1 (uses n + m samples):  fit G_hat = OLS(W ~ X)  →  Ŵ = X @ G_hat
  Stage 2 (uses n labeled only): fit β_hat = OLS(Y ~ Ŵ)

  More unlabeled data (larger m) improves Stage-1, which reduces the
  residual bias in Stage-2. The bound predicts MSE decreases as B/(n+m).

BASELINES
---------
  OLS-noisy : OLS(Y ~ W)   on n labeled  — biased due to measurement error
  Oracle    : OLS(Y ~ W̄)  on n labeled  — uses the true latent (upper bound)

WHAT TO LOOK FOR IN THE RESULTS
--------------------------------
  1. EIV should significantly outperform OLS-noisy (bias correction)
  2. EIV MSE should decrease as m grows (unlabeled data helps Stage-1)
  3. EIV should approach Oracle as n and m both increase
  4. Fitted curve  c + A/n + B/(n+m)  should match empirical MSE
  5. Optimal fraction α* = n/(n+m) should decrease as σ_W increases
     (noisier W → invest more in unlabeled data)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# ──────────────────────────────────────────────────────────────────────────────
# 1. Build instrument matrix X from real POI density features
# ──────────────────────────────────────────────────────────────────────────────
DATA_PATH = "Data/real_estate_clean_merged.csv"

X_COLS = [
    "population_scale_dense",
    "residential_area_dense",
    "subway_station_cnt_dense",
    "bus_station_cnt_dense",
    "education_dense",
    "catering_dense",
    "retail_dense",
]

df      = pd.read_csv(DATA_PATH)
X_raw   = df[X_COLS].dropna().values                    # (5360, 7)
X_std   = (X_raw - X_raw.mean(0)) / X_raw.std(0)       # standardise

# PCA → top-5 scores (removes multicollinearity, cond ≈ 1)
U, S, _ = np.linalg.svd(X_std, full_matrices=False)
N_PC    = 5
X_inst  = U[:, :N_PC] * S[:N_PC]
X_inst  = (X_inst - X_inst.mean(0)) / X_inst.std(0)    # re-standardise
N_TOTAL = X_inst.shape[0]

print(f"Instrument matrix  : {X_inst.shape}")
print(f"Condition number   : {np.linalg.cond(X_inst.T @ X_inst / N_TOTAL):.2f}\n")

# ──────────────────────────────────────────────────────────────────────────────
# 2. Semi-synthetic DGP
# ──────────────────────────────────────────────────────────────────────────────
rng_dgp   = np.random.default_rng(0)
D_X       = N_PC          # instrument dimension  (5)
D_W       = 3             # latent W̄ dimension
SIGMA_EPS = 1.0           # irreducible noise in Y

G_true    = rng_dgp.normal(0, 1.0, size=(D_X, D_W))   # Stage-1 loading
beta_true = rng_dgp.normal(0, 1.0, size=D_W)
beta_true /= np.linalg.norm(beta_true)                 # unit-norm

W_bar_all = X_inst @ G_true                            # (N, 3) — true latent
Y_all     = W_bar_all @ beta_true + rng_dgp.normal(0, SIGMA_EPS, N_TOTAL)

# Fixed test set (last 1 000 rows — never used during training)
N_TEST    = 1000
test_idx  = np.arange(N_TOTAL - N_TEST, N_TOTAL)
train_idx = np.arange(0, N_TOTAL - N_TEST)

X_test     = X_inst[test_idx]
W_bar_test = W_bar_all[test_idx]
Y_test     = Y_all[test_idx]

print(f"Train pool : {len(train_idx)}   |   Test : {N_TEST}\n")

# ──────────────────────────────────────────────────────────────────────────────
# 3. Helpers
# ──────────────────────────────────────────────────────────────────────────────
def add1(M):
    """Prepend a column of ones (intercept)."""
    return np.column_stack([np.ones(len(M)), M])

def ols(A, b):
    return np.linalg.lstsq(A, b, rcond=None)[0]

# ──────────────────────────────────────────────────────────────────────────────
# 4. Experiment loop  —  fixed labeled set per seed to isolate the m-effect
# ──────────────────────────────────────────────────────────────────────────────
N_VALUES     = [20, 50, 100, 200, 400]
M_VALUES     = [0, 50, 150, 300, 600, 1200, 2400, 4000]
NOISE_LEVELS = [0.5, 1.0, 2.0]
N_SEEDS      = 40

records = []

for sigma_w in NOISE_LEVELS:
    print(f"── σ_W = {sigma_w} ───────────────────────────────")
    rng_noise  = np.random.default_rng(int(sigma_w * 1000))
    W_noisy    = W_bar_all + rng_noise.normal(0, sigma_w, W_bar_all.shape)
    W_noisy_test = W_noisy[test_idx]

    for n in N_VALUES:
        for seed in range(N_SEEDS):
            rng_s = np.random.default_rng(seed * 1000 + n)

            # Draw n labeled indices — held fixed across all m for this seed
            lab  = rng_s.choice(train_idx, size=n, replace=False)
            pool = np.setdiff1d(train_idx, lab)   # remaining rows for unlabeled

            X_lab       = X_inst[lab]
            W_noisy_lab = W_noisy[lab]
            W_bar_lab   = W_bar_all[lab]
            Y_lab       = Y_all[lab]

            # Oracle: OLS(Y ~ W̄)  — uses true latent, fixed per (n, seed)
            b_orc      = ols(add1(W_bar_lab), Y_lab)
            oracle_mse = np.mean((Y_test - add1(W_bar_test) @ b_orc) ** 2)

            # OLS-noisy: OLS(Y ~ W)  — fixed per (n, seed)
            b_ols   = ols(add1(W_noisy_lab), Y_lab)
            ols_mse = np.mean((Y_test - add1(W_noisy_test) @ b_ols) ** 2)

            for m in M_VALUES:
                if m > len(pool):
                    continue
                unlab = rng_s.choice(pool, size=m, replace=False)

                # Stage 1: OLS(W ~ X) on n + m samples
                all_s1     = np.concatenate([lab, unlab])
                G_hat      = ols(add1(X_inst[all_s1]), W_noisy[all_s1])
                W_hat_lab  = add1(X_lab)  @ G_hat
                W_hat_test = add1(X_test) @ G_hat

                # Stage 2: OLS(Y ~ Ŵ) on n labeled
                b_eiv   = ols(add1(W_hat_lab), Y_lab)
                eiv_mse = np.mean((Y_test - add1(W_hat_test) @ b_eiv) ** 2)

                records.append(dict(
                    sigma_w=sigma_w, n=n, m=m, seed=seed,
                    mse_eiv=eiv_mse, mse_ols=ols_mse, mse_oracle=oracle_mse,
                ))

        print(f"  n = {n:4d}  done")

results = pd.DataFrame(records)
results.to_csv("real_estate_trials_raw.csv", index=False)

# ──────────────────────────────────────────────────────────────────────────────
# 5. Aggregate (mean + SE over seeds)
# ──────────────────────────────────────────────────────────────────────────────
agg = (
    results
    .groupby(["sigma_w", "n", "m"])
    .agg(
        mse_eiv   =("mse_eiv",    "mean"),
        mse_ols   =("mse_ols",    "mean"),
        mse_oracle=("mse_oracle", "mean"),
        se_eiv    =("mse_eiv",    lambda x: x.std() / np.sqrt(len(x))),
    )
    .reset_index()
)
agg.to_csv("real_estate_trials_summary.csv", index=False)

# ──────────────────────────────────────────────────────────────────────────────
# 6. Fit theoretical bound:  MSE(n, m) = c + A/n + B/(n+m)
# ──────────────────────────────────────────────────────────────────────────────
def mse_bound(X_data, c, A, B):
    n_arr, nm_arr = X_data
    return c + A / n_arr + B / nm_arr

print("\n── Curve fits ────────────────────────────────────────")
fit_rows = []
for sigma_w in NOISE_LEVELS:
    sub    = agg[agg.sigma_w == sigma_w].copy()
    n_arr  = sub["n"].values.astype(float)
    nm_arr = (sub["n"] + sub["m"]).values.astype(float)
    y_arr  = sub["mse_eiv"].values
    try:
        popt, _ = curve_fit(
            mse_bound, (n_arr, nm_arr), y_arr,
            p0=[SIGMA_EPS**2, 5.0, 5.0],
            bounds=([0, 0, 0], [np.inf, np.inf, np.inf]),
            maxfev=20000,
        )
        c_f, A_f, B_f = popt
        alpha_star = np.sqrt(A_f / B_f) if B_f > 1e-6 else np.inf
        print(f"  σ_W={sigma_w}:  ĉ={c_f:.3f}  Â={A_f:.2f}  B̂={B_f:.2f}"
              f"  α*=n/(n+m)={alpha_star:.2f}")
        fit_rows.append(dict(sigma_w=sigma_w, c_fit=c_f,
                             A_fit=A_f, B_fit=B_f, alpha_star=alpha_star))
    except Exception as e:
        print(f"  σ_W={sigma_w}: fit failed — {e}")

fits = pd.DataFrame(fit_rows)

# ──────────────────────────────────────────────────────────────────────────────
# 7. Print summary table
# ──────────────────────────────────────────────────────────────────────────────
print("\n── EIV vs OLS-noisy  (σ_W=1.0, m=1200, median over seeds) ───")
r10 = results[results.sigma_w == 1.0]
med = (
    r10.groupby(["n", "m"])
    .agg(eiv_med=("mse_eiv", "median"), ols_med=("mse_ols", "median"),
         oracle_med=("mse_oracle", "median"))
    .reset_index()
)
med["improvement_%"] = ((med.ols_med - med.eiv_med) / med.ols_med * 100).round(1)
print(med[med.m.isin([0, 150, 1200])].sort_values(["n", "m"]).to_string(index=False))

# ──────────────────────────────────────────────────────────────────────────────
# 8. Plots
# ──────────────────────────────────────────────────────────────────────────────
cmap   = plt.get_cmap("tab10")
colors = {n: cmap(i) for i, n in enumerate(N_VALUES)}

# ── Plot 1: MSE vs m for each n  (one panel per noise level) ─────────────────
for sigma_w in NOISE_LEVELS:
    sub = agg[agg.sigma_w == sigma_w]

    fig, ax = plt.subplots(figsize=(9, 5))

    for n in N_VALUES:
        row = sub[sub.n == n].sort_values("m")
        ax.errorbar(row["m"], row["mse_eiv"],
                    yerr=1.96 * row["se_eiv"],
                    marker="o", lw=2, color=colors[n],
                    label=f"EIV  n={n}")
        ax.plot(row["m"], row["mse_ols"],
                linestyle="--", lw=1.5, color=colors[n], alpha=0.55,
                label=f"OLS-noisy  n={n}")

    ax.axhline(SIGMA_EPS**2, color="k", lw=1.5, linestyle="-",
               label="σ²_ε  (irreducible floor)")
    ax.set_xlabel("m  (unlabeled samples)", fontsize=12)
    ax.set_ylabel("Test MSE", fontsize=12)
    ax.set_title(f"EIV vs OLS-noisy  |  σ_W = {sigma_w}\n"
                 f"(solid = EIV two-stage,  dashed = OLS on noisy W)",
                 fontsize=11)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fname = f"plot_noise_{sigma_w}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {fname}")

# ── Plot 2: Fitted MSE bound overlay  (σ_W=1.0, n=50) ───────────────────────
sigma_ref, n_ref = 1.0, 50
sub_ref  = agg[(agg.sigma_w == sigma_ref) & (agg.n == n_ref)].sort_values("m")
fit_ref  = fits[fits.sigma_w == sigma_ref]

fig, ax = plt.subplots(figsize=(8, 5))
ax.errorbar(sub_ref["m"], sub_ref["mse_eiv"],
            yerr=1.96 * sub_ref["se_eiv"],
            marker="o", color="steelblue", lw=2, label="Empirical EIV")
ax.plot(sub_ref["m"], sub_ref["mse_ols"],
        linestyle="--", color="tomato", lw=1.8, label="OLS-noisy")
ax.axhline(sub_ref["mse_oracle"].mean(), linestyle=":", color="green",
           lw=1.5, label="Oracle (true W̄)")
ax.axhline(SIGMA_EPS**2, color="k", lw=1.2, label="σ²_ε floor")

if len(fit_ref) > 0:
    c_f, A_f, B_f = fit_ref[["c_fit", "A_fit", "B_fit"]].values[0]
    m_dense  = np.linspace(0, sub_ref["m"].max(), 400)
    curve    = c_f + A_f / n_ref + B_f / (n_ref + m_dense)
    ax.plot(m_dense, curve, "r--", lw=2.5,
            label=f"Fitted: {c_f:.2f} + {A_f:.2f}/n + {B_f:.2f}/(n+m)")
    alpha_s = fit_ref["alpha_star"].values[0]
    if alpha_s < 1:
        opt_m = max(0, n_ref / alpha_s - n_ref)
        ax.axvline(min(opt_m, m_dense.max()), color="purple", lw=1.5,
                   linestyle="-.", label=f"Optimal m*={opt_m:.0f}  (α*={alpha_s:.2f})")

ax.set_xlabel("m  (unlabeled samples)", fontsize=12)
ax.set_ylabel("Test MSE", fontsize=12)
ax.set_title(f"Fitted MSE Bound  |  σ_W={sigma_ref},  n={n_ref}", fontsize=12)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("plot_fitted_bound.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved plot_fitted_bound.png")

# ── Plot 3: MSE heatmap over (n, m) grid  (σ_W=1.0) ─────────────────────────
sub_heat = agg[agg.sigma_w == 1.0].copy()
sub_heat["improve_pct"] = (
    (sub_heat.mse_ols - sub_heat.mse_eiv) / sub_heat.mse_ols * 100
)

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
for ax, (col, title, cmap_name) in zip(axes, [
    ("mse_eiv",     "EIV Test MSE",                   "viridis_r"),
    ("improve_pct", "% Improvement: EIV over OLS",    "RdYlGn"),
]):
    pivot = sub_heat.pivot_table(index="n", columns="m", values=col)
    im = ax.imshow(pivot.values, aspect="auto", cmap=cmap_name, origin="lower")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([str(int(c)) for c in pivot.columns], rotation=45, fontsize=8)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([str(int(r)) for r in pivot.index])
    ax.set_xlabel("m  (unlabeled)", fontsize=11)
    ax.set_ylabel("n  (labeled)", fontsize=11)
    ax.set_title(f"{title}  (σ_W=1.0)", fontsize=11)
    plt.colorbar(im, ax=ax)

plt.tight_layout()
plt.savefig("plot_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved_plot_heatmap.png")

print("\nDone.")
