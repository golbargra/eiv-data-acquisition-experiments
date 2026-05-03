"""
Simple EIV Experiment — Real Estate Data
=========================================

QUESTION: What happens to Y ~ W as we increase noise in W?
          Can we recover the true W̄ using the two-stage EIV estimator?

SETUP (scalar W, D_W = 1 for simplicity)
-----------------------------------------
  X   : 5 PCA components of real POI density features  (clean instruments)
  W̄  : X @ g_true                                      (true latent, scalar)
  W   : W̄ + N(0, σ_W)                                 (noisy observed proxy)
  Y   : W̄ * beta_true + N(0, 1)                        (outcome)

THREE MODELS
------------
  Oracle    : OLS(Y ~ W̄)   — uses the TRUE latent  (best possible)
  OLS-noisy : OLS(Y ~ W)    — uses the NOISY proxy  (naive, biased)
  EIV       : Stage 1 OLS(W ~ X) → Ŵ, Stage 2 OLS(Y ~ Ŵ)  (two-stage)

EXPERIMENT 1: vary noise σ_W  (fixed n=500, m=1000)
  → shows how OLS degrades as W gets noisier
  → shows whether EIV can recover Oracle performance

EXPERIMENT 2: vary n and m  (fixed σ_W=1.0)
  → shows how much labeled vs unlabeled data matters
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ──────────────────────────────────────────────
# 1. Build X from real POI density features
# ──────────────────────────────────────────────
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

df     = pd.read_csv(DATA_PATH)
X_raw  = df[X_COLS].dropna().values
X_std  = (X_raw - X_raw.mean(0)) / X_raw.std(0)

U, S, _ = np.linalg.svd(X_std, full_matrices=False)
X_inst  = U[:, :5] * S[:5]
X_inst  = (X_inst - X_inst.mean(0)) / X_inst.std(0)
N_TOTAL = X_inst.shape[0]

# ──────────────────────────────────────────────
# 2. DGP  (scalar W̄, D_W = 1)
# ──────────────────────────────────────────────
rng       = np.random.default_rng(0)
g_true    = rng.normal(0, 1.0, size=(5, 1))   # X → W̄ loading
beta_true = 1.0                                # Y = W̄ * beta + noise
SIGMA_EPS = 1.0

W_bar_all = (X_inst @ g_true).ravel()          # (N,) true latent
Y_all     = W_bar_all * beta_true + rng.normal(0, SIGMA_EPS, N_TOTAL)

# Fixed test set
N_TEST    = 1000
test_idx  = np.arange(N_TOTAL - N_TEST, N_TOTAL)
train_idx = np.arange(0, N_TOTAL - N_TEST)

X_test     = X_inst[test_idx]
W_bar_test = W_bar_all[test_idx]
Y_test     = Y_all[test_idx]

# ──────────────────────────────────────────────
# 3. Helpers
# ──────────────────────────────────────────────
def add1(v):
    if v.ndim == 1:
        v = v.reshape(-1, 1)
    return np.column_stack([np.ones(len(v)), v])

def predict_mse(X_tr, y_tr, X_te, y_te):
    b = np.linalg.lstsq(X_tr, y_tr, rcond=None)[0]
    return np.mean((y_te - X_te @ b) ** 2)

N_SEEDS = 50

# ══════════════════════════════════════════════
# EXPERIMENT 1: vary σ_W
# ══════════════════════════════════════════════
print("Experiment 1: varying noise σ_W")
print(f"  fixed n=500, m=1000, {N_SEEDS} seeds\n")

SIGMA_W_VALUES = [0.0, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0]
N_LAB, N_UNLAB = 500, 1000

exp1 = []

for sigma_w in SIGMA_W_VALUES:
    rng_n = np.random.default_rng(int(sigma_w * 100 + 1))
    W_noisy = W_bar_all + rng_n.normal(0, sigma_w, N_TOTAL)

    oracle_list, ols_list, eiv_list = [], [], []

    for seed in range(N_SEEDS):
        rng_s = np.random.default_rng(seed)
        sel   = rng_s.choice(train_idx, size=N_LAB + N_UNLAB, replace=False)
        lab, unlab = sel[:N_LAB], sel[N_LAB:]

        # Oracle
        oracle_list.append(predict_mse(
            add1(W_bar_all[lab]), Y_all[lab],
            add1(W_bar_test),    Y_test))

        # OLS-noisy
        ols_list.append(predict_mse(
            add1(W_noisy[lab]), Y_all[lab],
            add1(W_noisy[test_idx]), Y_test))

        # EIV two-stage
        all_s1  = np.concatenate([lab, unlab])
        g_hat   = np.linalg.lstsq(
            add1(X_inst[all_s1]), W_noisy[all_s1], rcond=None)[0]
        W_hat_lab  = add1(X_inst[lab])  @ g_hat
        W_hat_test = add1(X_test)       @ g_hat

        eiv_list.append(predict_mse(
            add1(W_hat_lab), Y_all[lab],
            add1(W_hat_test), Y_test))

    exp1.append(dict(
        sigma_w   = sigma_w,
        oracle    = np.mean(oracle_list),
        ols_noisy = np.mean(ols_list),
        eiv       = np.mean(eiv_list),
        se_oracle = np.std(oracle_list)  / np.sqrt(N_SEEDS),
        se_ols    = np.std(ols_list)     / np.sqrt(N_SEEDS),
        se_eiv    = np.std(eiv_list)     / np.sqrt(N_SEEDS),
    ))
    print(f"  σ_W={sigma_w:.2f}:  Oracle={exp1[-1]['oracle']:.3f}"
          f"  OLS={exp1[-1]['ols_noisy']:.3f}"
          f"  EIV={exp1[-1]['eiv']:.3f}")

df1 = pd.DataFrame(exp1)

# ══════════════════════════════════════════════
# EXPERIMENT 2: vary n and m  (σ_W = 1.0)
# ══════════════════════════════════════════════
print("\nExperiment 2: varying n and m  (σ_W=1.0)")

SIGMA_W_FIXED = 1.0
rng_n2 = np.random.default_rng(200)
W_noisy2 = W_bar_all + rng_n2.normal(0, SIGMA_W_FIXED, N_TOTAL)

N_VALUES = [50, 100, 200, 500, 1000]
M_VALUES = [0, 100, 500, 1000, 2000]

exp2 = []

for n in N_VALUES:
    for m in M_VALUES:
        if n + m > len(train_idx):
            continue
        oracle_list, ols_list, eiv_list = [], [], []

        for seed in range(N_SEEDS):
            rng_s = np.random.default_rng(seed * 100 + n)
            lab  = rng_s.choice(train_idx, size=n, replace=False)
            pool = np.setdiff1d(train_idx, lab)
            unlab = rng_s.choice(pool, size=m, replace=False)

            oracle_list.append(predict_mse(
                add1(W_bar_all[lab]), Y_all[lab],
                add1(W_bar_test),    Y_test))

            ols_list.append(predict_mse(
                add1(W_noisy2[lab]),         Y_all[lab],
                add1(W_noisy2[test_idx]),    Y_test))

            all_s1     = np.concatenate([lab, unlab])
            g_hat      = np.linalg.lstsq(
                add1(X_inst[all_s1]), W_noisy2[all_s1], rcond=None)[0]
            W_hat_lab  = add1(X_inst[lab])  @ g_hat
            W_hat_test = add1(X_test)       @ g_hat

            eiv_list.append(predict_mse(
                add1(W_hat_lab), Y_all[lab],
                add1(W_hat_test), Y_test))

        exp2.append(dict(
            n=n, m=m,
            oracle    = np.mean(oracle_list),
            ols_noisy = np.mean(ols_list),
            eiv       = np.mean(eiv_list),
            se_eiv    = np.std(eiv_list) / np.sqrt(N_SEEDS),
        ))

    print(f"  n={n} done")

df2 = pd.DataFrame(exp2)

# ──────────────────────────────────────────────
# 4. Plots
# ──────────────────────────────────────────────

# ── Plot 1: MSE vs σ_W  (full + zoomed side by side) ────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Effect of measurement noise on regression MSE — Can EIV recover Oracle?",
             fontsize=12)

for ax, (title, ylim) in zip(axes, [
    ("Full scale",          None),
    ("Zoomed in (EIV vs Oracle)", (0.96, 1.10)),
]):
    ax.plot(df1.sigma_w, df1.oracle,    "g--", lw=2, marker="s", label="Oracle  (true W̄)")
    ax.plot(df1.sigma_w, df1.ols_noisy, "r-",  lw=2, marker="o", label="OLS-noisy  (Y ~ W)")
    ax.plot(df1.sigma_w, df1.eiv,       "b-",  lw=2, marker="^", label="EIV two-stage  (Y ~ Ŵ)")
    ax.fill_between(df1.sigma_w,
                    df1.eiv - 1.96 * df1.se_eiv,
                    df1.eiv + 1.96 * df1.se_eiv,
                    color="blue", alpha=0.15)
    ax.axhline(SIGMA_EPS**2, color="k", lw=1.2, linestyle=":", label="σ²_ε floor")
    if ylim:
        ax.set_ylim(ylim)
    ax.set_xlabel("σ_W  (noise level in W)", fontsize=12)
    ax.set_ylabel("Test MSE", fontsize=12)
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("simple_mse_vs_noise.png", dpi=150, bbox_inches="tight")
plt.close()
print("\nSaved simple_mse_vs_noise.png")

# ── Plot 2: MSE vs m for each n  (σ_W=1.0, full + zoomed) ───────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("EIV MSE vs unlabeled sample size m  (σ_W=1.0)\n"
             "solid=EIV, dashed=OLS-noisy, dotted=Oracle", fontsize=12)

cmap   = plt.get_cmap("tab10")
colors = {n: cmap(i) for i, n in enumerate(N_VALUES)}

for ax, (title, ylim) in zip(axes, [
    ("Full scale",   None),
    ("Zoomed in",    (0.95, 1.15)),
]):
    for n in N_VALUES:
        sub = df2[df2.n == n].sort_values("m")
        ax.errorbar(sub.m, sub.eiv,
                    yerr=1.96 * sub.se_eiv,
                    marker="o", lw=2, color=colors[n], label=f"EIV  n={n}")
        ols_val    = sub.ols_noisy.iloc[0]
        oracle_val = sub.oracle.iloc[0]
        ax.axhline(ols_val,    linestyle="--", lw=1.2,
                   color=colors[n], alpha=0.5, label=f"OLS  n={n}")
        ax.axhline(oracle_val, linestyle=":",  lw=1.2,
                   color=colors[n], alpha=0.5, label=f"Oracle  n={n}")

    ax.axhline(SIGMA_EPS**2, color="k", lw=1.5,
               linestyle="-.", label="σ²_ε floor")
    if ylim:
        ax.set_ylim(ylim)
    ax.set_xlabel("m  (unlabeled samples)", fontsize=12)
    ax.set_ylabel("Test MSE", fontsize=12)
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("simple_mse_vs_m.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved simple_mse_vs_m.png")

# ── Plot 3: Can we recover W̄?  (σ_W=1.0, n=500, m=1000) ──
rng_p = np.random.default_rng(7)
sel   = rng_p.choice(train_idx, size=500 + 1000, replace=False)
lab_p, unlab_p = sel[:500], sel[500:]

W_noisy_p = W_bar_all + np.random.default_rng(99).normal(0, 1.0, N_TOTAL)
g_hat_p   = np.linalg.lstsq(
    add1(X_inst[np.concatenate([lab_p, unlab_p])]),
    W_noisy_p[np.concatenate([lab_p, unlab_p])],
    rcond=None)[0]
W_hat_p = add1(X_test) @ g_hat_p

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Can we recover the true W̄?  (σ_W=1.0, n=500, m=1000)", fontsize=12)

for ax, (w_vals, title, color) in zip(axes, [
    (W_noisy_p[test_idx], "Noisy W  vs  true W̄", "tomato"),
    (W_hat_p,             "Denoised Ŵ  vs  true W̄", "steelblue"),
]):
    ax.scatter(W_bar_test, w_vals, alpha=0.3, s=8, color=color)
    lo = min(W_bar_test.min(), w_vals.min())
    hi = max(W_bar_test.max(), w_vals.max())
    ax.plot([lo, hi], [lo, hi], "k--", lw=1.5, label="perfect recovery")
    corr = np.corrcoef(W_bar_test, w_vals)[0, 1]
    ax.set_xlabel("True W̄", fontsize=11)
    ax.set_ylabel(title.split(" vs")[0], fontsize=11)
    ax.set_title(f"{title}\n(correlation = {corr:.3f})", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("simple_w_recovery.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved simple_w_recovery.png")

print("\nDone.")
