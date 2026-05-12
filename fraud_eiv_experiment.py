"""
Fraud EIV DA Experiment — Credit Card Data (Y = Amount)
========================================================

DGP (semi-synthetic):
  X   = first N_PC principal components of V1-V28  (real instruments, 284K rows)
  W̄   = g(X)                       (latent; linear or tanh nonlinear, D_W=3)
  W   = W̄ + ε_W,  ε_W ~ N(0,σ²_W)  (noisy proxy)
  Y   = Amount (standardised)        (REAL continuous outcome)
  C   = Class                        (REAL binary fraud label, used only in cost)

G_true[:,0] is aligned with the OLS(Amount~X) direction so W̄ is genuinely
predictive of Y; the other two columns are random orthogonal directions.

Imbalance fix for labeled set: stratified sampling — n/2 fraud + n/2 legit.
Unlabeled pool: full remaining ~283K transactions (no Y/C needed for Stage 1).
Test set: stratified 100 fraud + 900 legit.

Decision:  z = 1{score_i > T} — flag transaction for investigation
Cost:      ℓ(z, y, c) = (1−z)·c·y + z·(1−c)·c_fp
           miss fraud (z=0,c=1): lose Amount y
           false alarm (z=1,c=0): pay investigation cost c_fp
Oracle:    z* = c  (flags exactly fraud) → oracle cost = 0, regret = cost

4-way comparison:
  ols_pto  OLS Stage-1 + PtO Stage-2  (MSE on Amount)
  ols_da   OLS Stage-1 + DA Stage-2   (PG loss, professor's framework)
  nw_pto   NW  Stage-1 + PtO Stage-2
  nw_da    NW  Stage-1 + DA Stage-2
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
from decision_learning.modeling.loss import PG_Loss

# ─────────────────────────────────────────────────────────────────────────────
# 0.  Configuration
# ─────────────────────────────────────────────────────────────────────────────
CSV_PATH     = "Data/creditcard.csv"
N_PC         = 5
D_W          = 3
N_TEST_FRAUD = 100
N_TEST_LEGIT = 900
NOISE_LEVELS = [0.5, 1.0, 2.0]
N_VALUES     = [100, 200, 400]    # n/2 fraud + n/2 legit each
M_VALUES     = [0, 1000, 5000, 10000, 30000]
N_SEEDS      = 10
C_FP         = 0.1    # false-alarm cost (in standardised Amount units)

# ─────────────────────────────────────────────────────────────────────────────
# 1.  Load full dataset, standardise Amount, build PCA instruments
# ─────────────────────────────────────────────────────────────────────────────
df    = pd.read_csv(CSV_PATH)
C_all = df["Class"].values                       # real fraud labels
Y_raw = df["Amount"].values
Y_all = (Y_raw - Y_raw.mean()) / Y_raw.std()    # standardise

X_raw = df[[f"V{i}" for i in range(1, 29)]].values
X_std = (X_raw - X_raw.mean(0)) / X_raw.std(0)
U_full, S_full, _ = np.linalg.svd(X_std, full_matrices=False)

def make_instruments(k):
    X = U_full[:, :k] * S_full[:k]
    return (X - X.mean(0)) / X.std(0)

X_inst  = make_instruments(N_PC)
N_TOTAL = len(X_inst)
print(f"Dataset size      : {N_TOTAL}")
print(f"Instrument matrix : {X_inst.shape}")

# ─────────────────────────────────────────────────────────────────────────────
# 2.  Stratified test / train split
# ─────────────────────────────────────────────────────────────────────────────
rng_split  = np.random.default_rng(99)
all_fraud  = np.where(C_all == 1)[0]   # 492
all_legit  = np.where(C_all == 0)[0]   # 284315

test_fraud = rng_split.choice(all_fraud, size=N_TEST_FRAUD, replace=False)
test_legit = rng_split.choice(all_legit, size=N_TEST_LEGIT, replace=False)
test_idx   = np.concatenate([test_fraud, test_legit])

fraud_pool = np.setdiff1d(all_fraud, test_fraud)   # ~392 fraud for labeled
legit_pool = np.setdiff1d(all_legit, test_legit)   # ~283415 legit for labeled+unlabeled

X_test = X_inst[test_idx]
Y_test = Y_all[test_idx]
C_test = C_all[test_idx]

print(f"Test  fraud: {C_test.sum()}   legit: {(1-C_test).sum()}")
print(f"Fraud pool : {len(fraud_pool)}  Legit pool: {len(legit_pool)}")

# ─────────────────────────────────────────────────────────────────────────────
# 3.  DGP — build G_true with first column aligned to OLS(Amount~X) direction
# ─────────────────────────────────────────────────────────────────────────────
def add1(M):
    return np.column_stack([np.ones(len(M)), M])

def ols(A, b):
    return np.linalg.lstsq(A, b, rcond=None)[0]

# OLS direction: most predictive linear combination of X for Amount
beta_ols_amount = ols(add1(X_inst), Y_all)[1:]       # (N_PC,)
beta_ols_amount /= np.linalg.norm(beta_ols_amount)

rng_dgp = np.random.default_rng(0)
G_true  = rng_dgp.normal(size=(N_PC, D_W))
G_true[:, 0] = beta_ols_amount

# Gram-Schmidt to orthogonalise columns
for j in range(1, D_W):
    v = G_true[:, j].copy()
    for k in range(j):
        v -= (v @ G_true[:, k]) * G_true[:, k]
    G_true[:, j] = v / np.linalg.norm(v)

def make_W_bar(X, dgp):
    XG = X @ G_true
    return XG if dgp == "linear" else np.tanh(XG)

# ─────────────────────────────────────────────────────────────────────────────
# 4.  Nadaraya-Watson kernel smoother
# ─────────────────────────────────────────────────────────────────────────────
def _nw_chunk(X_tr, W_tr, X_ev, h):
    d       = X_ev[:, None, :] - X_tr[None, :, :]
    K       = np.exp(-np.einsum("ijk,ijk->ij", d, d) / (2 * h * h))
    row_sum = K.sum(1, keepdims=True)
    row_sum[row_sum < 1e-12] = 1.0
    return (K / row_sum) @ W_tr

def nw_predict(X_tr, W_tr, X_ev, h, chunk=200):
    out = np.zeros((len(X_ev), W_tr.shape[1]))
    for s in range(0, len(X_ev), chunk):
        e = min(s + chunk, len(X_ev))
        out[s:e] = _nw_chunk(X_tr, W_tr, X_ev[s:e], h)
    return out

def silverman(X):
    N, d = X.shape
    return 1.06 * X.std(0).mean() * N ** (-1.0 / (d + 4))

# ─────────────────────────────────────────────────────────────────────────────
# 5.  Decision cost  ℓ(z,y,c) = (1−z)·c·y + z·(1−c)·c_fp
# ─────────────────────────────────────────────────────────────────────────────
def decision_cost(z, y, c, c_fp):
    return float(np.mean((1 - z) * c * y + z * (1 - c) * c_fp))

# ─────────────────────────────────────────────────────────────────────────────
# 6.  Stage-2 DA: PG loss (professor's framework)
#     pred_cost = predicted Amount (β^T Ŵ)
#     true_cost = actual Amount (Y)
#     optmodel  = threshold at c_fp, evaluate against real Y and C
# ─────────────────────────────────────────────────────────────────────────────
def _make_fraud_optmodel(y_tensor, c_tensor, c_fp):
    def optmodel(pred_cost, solver_kwargs={}):
        z   = (pred_cost > c_fp).float()
        obj = (1 - z) * c_tensor * y_tensor + z * (1 - c_tensor) * c_fp
        return z, obj
    return optmodel

def fit_pg(W_hat, y, c, c_fp):
    n     = len(y)
    h     = n ** -0.25
    W_aug = torch.FloatTensor(add1(W_hat))          # (n, D_W+1)
    y_t   = torch.FloatTensor(y).reshape(-1, 1)
    c_t   = torch.FloatTensor(c).reshape(-1, 1)

    beta  = nn.Parameter(
        torch.FloatTensor(ols(add1(W_hat), y)).reshape(-1, 1)
    )
    pg_loss_fn = PG_Loss(
        optmodel       = _make_fraud_optmodel(y_t, c_t, c_fp),
        h              = h,
        finite_diff_type = 'B',
        reduction      = 'mean',
        minimize       = True,
    )
    optimizer = torch.optim.LBFGS([beta], max_iter=500, tolerance_grad=1e-10)

    def closure():
        optimizer.zero_grad()
        pred_cost = W_aug @ beta
        loss      = pg_loss_fn(pred_cost, y_t)
        loss.backward()
        return loss

    optimizer.step(closure)
    return beta.detach().numpy().flatten()

# ─────────────────────────────────────────────────────────────────────────────
# 7.  Threshold tuning and evaluation
# ─────────────────────────────────────────────────────────────────────────────
def tune_threshold(scores, y, c, c_fp, n_grid=200):
    ts    = np.percentile(scores, np.linspace(1, 99, n_grid))
    costs = [decision_cost((scores > T).astype(float), y, c, c_fp) for T in ts]
    return ts[int(np.argmin(costs))]

def eval_decision(beta, Wh_lab, Wh_test, y_lab, c_lab, y_test, c_test, c_fp):
    sc_lab  = add1(Wh_lab)  @ beta
    sc_test = add1(Wh_test) @ beta
    T       = tune_threshold(sc_lab, y_lab, c_lab, c_fp)
    z       = (sc_test > T).astype(float)
    return decision_cost(z, y_test, c_test, c_fp)

ORACLE_COST = 0.0

# ─────────────────────────────────────────────────────────────────────────────
# 8.  Main experiment loop
# ─────────────────────────────────────────────────────────────────────────────
records = []

for dgp in ("linear", "nonlinear"):
    W_bar_all  = make_W_bar(X_inst, dgp)
    W_bar_test = W_bar_all[test_idx]

    print(f"\n{'='*60}\nDGP={dgp.upper()}\n{'='*60}")

    for sigma_w in NOISE_LEVELS:
        rng_n        = np.random.default_rng(int(sigma_w * 1000))
        W_noisy_all  = W_bar_all + rng_n.normal(0, sigma_w, W_bar_all.shape)
        W_noisy_test = W_noisy_all[test_idx]

        print(f"\n── σ_W={sigma_w} ─────────────────────────────")

        for n in N_VALUES:
            for seed in range(N_SEEDS):
                rng_s = np.random.default_rng(seed * 1000 + n)

                # Stratified labeled sample: n/2 fraud + n/2 legit
                lab_f = rng_s.choice(fraud_pool, size=n // 2, replace=False)
                lab_l = rng_s.choice(legit_pool, size=n // 2, replace=False)
                lab   = np.concatenate([lab_f, lab_l])

                # Unlabeled pool: everything else (full ~283K remaining)
                pool  = np.setdiff1d(
                    np.concatenate([fraud_pool, legit_pool]), lab
                )

                X_lab       = X_inst[lab]
                W_noisy_lab = W_noisy_all[lab]
                W_bar_lab   = W_bar_all[lab]
                Y_lab       = Y_all[lab]
                C_lab       = C_all[lab]

                # Oracle Stage-2 (uses true W̄)
                b_orc       = ols(add1(W_bar_lab), Y_lab)
                cost_orc_s2 = eval_decision(
                    b_orc, W_bar_lab, W_bar_test,
                    Y_lab, C_lab, Y_test, C_test, C_FP
                )

                # Noisy-W baseline
                b_noisy    = ols(add1(W_noisy_lab), Y_lab)
                cost_noisy = eval_decision(
                    b_noisy, W_noisy_lab, W_noisy_test,
                    Y_lab, C_lab, Y_test, C_test, C_FP
                )

                for m in M_VALUES:
                    if m > len(pool):
                        continue

                    unlab  = rng_s.choice(pool, size=m, replace=False)
                    s1_idx = np.concatenate([lab, unlab])
                    X_s1   = X_inst[s1_idx]
                    W_s1   = W_noisy_all[s1_idx]

                    # Stage 1a: OLS
                    G_hat       = ols(add1(X_s1), W_s1)
                    Wh_ols_lab  = add1(X_lab)  @ G_hat
                    Wh_ols_test = add1(X_test) @ G_hat

                    # Stage 1b: NW kernel
                    h           = silverman(X_s1)
                    Wh_nw_lab   = nw_predict(X_s1, W_s1, X_lab,  h)
                    Wh_nw_test  = nw_predict(X_s1, W_s1, X_test, h)

                    for s1_name, Wh_lab_, Wh_test_ in [
                        ("ols", Wh_ols_lab, Wh_ols_test),
                        ("nw",  Wh_nw_lab,  Wh_nw_test),
                    ]:
                        # Stage 2: PtO (OLS on Amount)
                        b_pto    = ols(add1(Wh_lab_), Y_lab)
                        cost_pto = eval_decision(
                            b_pto, Wh_lab_, Wh_test_,
                            Y_lab, C_lab, Y_test, C_test, C_FP
                        )

                        # Stage 2: DA (PG loss)
                        b_da    = fit_pg(Wh_lab_, Y_lab, C_lab, C_FP)
                        cost_da = eval_decision(
                            b_da, Wh_lab_, Wh_test_,
                            Y_lab, C_lab, Y_test, C_test, C_FP
                        )

                        records.append(dict(
                            dgp=dgp, sigma_w=sigma_w, n=n, m=m, seed=seed,
                            stage1=s1_name,
                            cost_oracle_s2=cost_orc_s2,
                            cost_noisy=cost_noisy,
                            cost_pto=cost_pto,
                            cost_da=cost_da,
                            regret_pto=cost_pto - ORACLE_COST,
                            regret_da=cost_da  - ORACLE_COST,
                        ))

            print(f"  n={n:4d}  done")

results = pd.DataFrame(records)
results.to_csv("fraud_eiv_raw.csv", index=False)
print("\nSaved fraud_eiv_raw.csv")

# ─────────────────────────────────────────────────────────────────────────────
# 9.  Aggregate
# ─────────────────────────────────────────────────────────────────────────────
def se(x):
    return x.std() / np.sqrt(len(x))

agg = (
    results
    .groupby(["dgp", "sigma_w", "n", "m", "stage1"])
    .agg(
        regret_pto_mean=("regret_pto", "mean"),
        regret_da_mean =("regret_da",  "mean"),
        regret_pto_se  =("regret_pto", se),
        regret_da_se   =("regret_da",  se),
    )
    .reset_index()
)
agg.to_csv("fraud_eiv_summary.csv", index=False)
print("Saved fraud_eiv_summary.csv")

# ─────────────────────────────────────────────────────────────────────────────
# 10. Plots
# ─────────────────────────────────────────────────────────────────────────────
cmap     = plt.get_cmap("tab10")
n_colors = {n: cmap(i) for i, n in enumerate(N_VALUES)}

METHODS = [
    ("ols", "regret_pto_mean", "regret_pto_se", "s--", 0.55, "OLS S1 + PtO"),
    ("ols", "regret_da_mean",  "regret_da_se",  "s-",  0.90, "OLS S1 + DA"),
    ("nw",  "regret_pto_mean", "regret_pto_se", "o--", 0.55, "NW S1  + PtO"),
    ("nw",  "regret_da_mean",  "regret_da_se",  "o-",  1.00, "NW S1  + DA"),
]

# ── Plot 1: Decision cost vs m ────────────────────────────────────────────────
for dgp in ("linear", "nonlinear"):
    for sigma_w in NOISE_LEVELS:
        fig, ax = plt.subplots(figsize=(10, 5))
        for n in N_VALUES:
            for s1, cm, cs, fmt, alpha, lbl in METHODS:
                row = agg[
                    (agg.dgp == dgp) & (agg.sigma_w == sigma_w) &
                    (agg.n == n) & (agg.stage1 == s1)
                ].sort_values("m")
                if row.empty:
                    continue
                ax.errorbar(row["m"], row[cm], yerr=1.96 * row[cs],
                            fmt=fmt, lw=1.8, color=n_colors[n], alpha=alpha,
                            label=f"{lbl}  n={n}")
        ax.set_xlabel("m  (unlabeled samples)", fontsize=12)
        ax.set_ylabel("Decision Cost  (oracle = 0)", fontsize=12)
        ax.set_title(
            f"Decision Cost vs m  |  DGP: {dgp}  |  σ_W={sigma_w}\n"
            f"Y=Amount (std), c_fp={C_FP}  |  solid=DA, dashed=PtO  |  circle=NW, square=OLS",
            fontsize=11,
        )
        ax.legend(fontsize=6, ncol=2)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fname = f"fraud_regret_{dgp}_sw{sigma_w}.png"
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved {fname}")

# ── Plot 2: Bar comparison at m=10000, σ_W=1.0 ───────────────────────────────
for dgp in ("linear", "nonlinear"):
    sub = agg[(agg.dgp == dgp) & (agg.sigma_w == 1.0) & (agg.m == 10000)]
    fig, axes = plt.subplots(1, len(N_VALUES), figsize=(13, 4), sharey=True)
    bar_colors = ["#d9534f", "#f0ad4e", "#5bc0de", "#5cb85c"]
    bar_labels = ["OLS+PtO", "OLS+DA", "NW+PtO", "NW+DA"]

    for ax, n in zip(axes, N_VALUES):
        sub_n = sub[sub.n == n]
        ols_r = sub_n[sub_n.stage1 == "ols"]
        nw_r  = sub_n[sub_n.stage1 == "nw"]

        vals, errs = [], []
        for r, mc, sc in [
            (ols_r, "regret_pto_mean", "regret_pto_se"),
            (ols_r, "regret_da_mean",  "regret_da_se"),
            (nw_r,  "regret_pto_mean", "regret_pto_se"),
            (nw_r,  "regret_da_mean",  "regret_da_se"),
        ]:
            if len(r):
                vals.append(float(r[mc].iloc[0]))
                errs.append(1.96 * float(r[sc].iloc[0]))
            else:
                vals.append(0.0); errs.append(0.0)

        ax.bar(bar_labels, vals, yerr=errs,
               color=bar_colors, capsize=4, alpha=0.85)
        ax.set_title(f"n={n}", fontsize=10)
        ax.set_xticks(range(len(bar_labels)))
        ax.set_xticklabels(bar_labels, rotation=30, ha="right", fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")

    axes[0].set_ylabel("Decision Cost  (m=10000, σ_W=1.0)", fontsize=10)
    fig.suptitle(f"4-Way Comparison  |  DGP: {dgp}  |  c_fp={C_FP}", fontsize=12)
    plt.tight_layout()
    fname = f"fraud_bar_{dgp}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {fname}")

# ── Plot 3: Heatmap — DA % improvement over PtO (NW S1, σ_W=1.0) ────────────
for dgp in ("linear", "nonlinear"):
    sub_h = agg[
        (agg.dgp == dgp) & (agg.sigma_w == 1.0) & (agg.stage1 == "nw")
    ].copy()
    sub_h["da_impr_pct"] = (
        (sub_h.regret_pto_mean - sub_h.regret_da_mean) /
        sub_h.regret_pto_mean.abs() * 100
    )
    pivot = sub_h.pivot_table(index="n", columns="m", values="da_impr_pct")
    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn", origin="lower")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([str(int(c)) for c in pivot.columns], rotation=45, fontsize=8)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([str(int(r)) for r in pivot.index])
    ax.set_xlabel("m  (unlabeled)", fontsize=11)
    ax.set_ylabel("n  (labeled)", fontsize=11)
    ax.set_title(
        f"DA % improvement over PtO  (NW S1)  |  DGP={dgp}  σ_W=1.0",
        fontsize=10,
    )
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    fname = f"fraud_heatmap_da_improvement_{dgp}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {fname}")

print("\nDone.")
