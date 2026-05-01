import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# =========================
# 1. Load cleaned data
# =========================

DATA_PATH = "bank_cleaned.csv"
data = pd.read_csv(DATA_PATH)

y = data["y"]

W_features = ["duration", "pdays", "campaign"]

X_features = [
    col for col in data.columns
    if col not in W_features + ["y"]
]

X_stable = data[X_features]
W_clean = data[W_features]


# =========================
# 2. Scale X and W
# =========================

x_scaler = StandardScaler()
w_scaler = StandardScaler()

X_scaled = pd.DataFrame(
    x_scaler.fit_transform(X_stable),
    columns=X_features,
    index=data.index
)

W_clean_scaled = pd.DataFrame(
    w_scaler.fit_transform(W_clean),
    columns=W_features,
    index=data.index
)


# =========================
# 3. Train/test split
# =========================

X_train, X_test, W_train_clean, W_test_clean, y_train, y_test = train_test_split(
    X_scaled,
    W_clean_scaled,
    y,
    test_size=0.30,
    stratify=y,
    random_state=42
)


# =========================
# 4. Functions
# =========================

def add_noise(W_clean, noise_level, seed=42):
    W_noisy = W_clean.copy()
    rng = np.random.default_rng(seed)

    for col in W_noisy.columns:
        W_noisy[col] += rng.normal(0, noise_level, size=len(W_noisy))

    return W_noisy


def sample_labeled_unlabeled_indices(y_train, n, m, seed=123):
    rng = np.random.default_rng(seed)

    positive_idx = y_train[y_train == 1].index.to_numpy()
    negative_idx = y_train[y_train == 0].index.to_numpy()

    positive_rate = y_train.mean()

    n_positive = max(1, int(round(n * positive_rate)))
    n_positive = min(n_positive, len(positive_idx))

    n_negative = n - n_positive

    labeled_positive = rng.choice(positive_idx, size=n_positive, replace=False)
    labeled_negative = rng.choice(negative_idx, size=n_negative, replace=False)

    labeled_idx = np.concatenate([labeled_positive, labeled_negative])

    used = set(labeled_idx)
    remaining_idx = np.array([idx for idx in y_train.index if idx not in used])

    if m > len(remaining_idx):
        m = len(remaining_idx)

    unlabeled_idx = rng.choice(remaining_idx, size=m, replace=False)

    rng.shuffle(labeled_idx)
    rng.shuffle(unlabeled_idx)

    return labeled_idx, unlabeled_idx


def two_stage_denoise(X_labeled, W_clean_labeled, X_unlabeled, W_clean_unlabeled, X_target):
    X_stage1 = pd.concat([X_labeled, X_unlabeled])
    W_stage1 = pd.concat([W_clean_labeled, W_clean_unlabeled])

    denoiser = LinearRegression()
    denoiser.fit(X_stage1, W_stage1)

    W_hat = denoiser.predict(X_target)

    return pd.DataFrame(W_hat, columns=W_clean_labeled.columns, index=X_target.index), denoiser


def train_and_eval(W_train, y_train, W_test, y_test):
    model = LinearRegression()
    model.fit(W_train, y_train)

    y_pred = model.predict(W_test)

    return {"mse": mean_squared_error(y_test, y_pred)}


# =========================
# 5. Experiment loop
# =========================

noise_levels = [0.5, 1.0, 2.0, 3.0]
n_values = [500, 1000, 2000, 5000]
m_values = [0, 1000, 5000, 10000, 20000]

results = []
base_seed = 123


for noise in noise_levels:
    print(f"\n=== Noise level = {noise} ===")

    W_train_noisy = add_noise(W_train_clean, noise, seed=base_seed)
    W_test_noisy = add_noise(W_test_clean, noise, seed=base_seed + 1)

    for n in n_values:
        print(f"\n--- n = {n} ---")

        # -------------------------
        # Sample ONCE for clean & noisy
        # -------------------------
        labeled_idx, _ = sample_labeled_unlabeled_indices(
            y_train=y_train,
            n=n,
            m=0,  # no unlabeled needed here
            seed=base_seed + n
        )

        X_labeled = X_train.loc[labeled_idx]

        W_clean_labeled = W_train_clean.loc[labeled_idx]
        W_noisy_labeled = W_train_noisy.loc[labeled_idx]

        y_labeled = y_train.loc[labeled_idx]

        # -------------------------
        # CLEAN (does NOT depend on m or noise)
        # -------------------------
        clean_metrics = train_and_eval(
            W_clean_labeled,
            y_labeled,
            W_test_clean,
            y_test
        )

        results.append({
            "noise": noise,
            "n": n,
            "m": None,  # important
            "model": "clean",
            **clean_metrics
        })

        # -------------------------
        # NOISY (depends on noise, NOT m)
        # -------------------------
        noisy_metrics = train_and_eval(
            W_noisy_labeled,
            y_labeled,
            W_test_noisy,
            y_test
        )

        results.append({
            "noise": noise,
            "n": n,
            "m": None,  # important
            "model": "noisy",
            **noisy_metrics
        })

        # -------------------------
        # TWO-STAGE (depends on m)
        # -------------------------
        for m in m_values:
            print(f"noise={noise}, n={n}, m={m}")

            labeled_idx, unlabeled_idx = sample_labeled_unlabeled_indices(
                y_train=y_train,
                n=n,
                m=m,
                seed=base_seed + n + m
            )

            X_labeled = X_train.loc[labeled_idx]
            X_unlabeled = X_train.loc[unlabeled_idx]

            W_clean_labeled = W_train_clean.loc[labeled_idx]
            W_clean_unlabeled = W_train_clean.loc[unlabeled_idx]

            y_labeled = y_train.loc[labeled_idx]

            # denoise using X → W
            W_hat_labeled, denoiser = two_stage_denoise(
                X_labeled,
                W_clean_labeled,
                X_unlabeled,
                W_clean_unlabeled,
                X_labeled
            )

            W_hat_test = pd.DataFrame(
                denoiser.predict(X_test),
                columns=W_features,
                index=X_test.index
            )

            denoised_metrics = train_and_eval(
                W_hat_labeled,
                y_labeled,
                W_hat_test,
                y_test
            )

            results.append({
                "noise": noise,
                "n": n,
                "m": m,
                "model": "two_stage",
                **denoised_metrics
            })

# =========================
# 6. Save results
# =========================

results_df = pd.DataFrame(results)
results_df.to_csv("bank_experiment_results.csv", index=False)

# -------------------------
# Separate pivots
# -------------------------

# 1. clean & noisy (no m dependence)
baseline_df = results_df[results_df["model"].isin(["clean", "noisy"])]

baseline_pivot = baseline_df.pivot_table(
    index=["noise", "n", "model"],
    values="mse"
)

baseline_pivot.to_csv("bank_baseline_mse.csv")


# 2. two-stage (depends on m)
two_stage_df = results_df[results_df["model"] == "two_stage"]

two_stage_pivot = two_stage_df.pivot_table(
    index=["noise", "n", "m"],
    values="mse"
)

two_stage_pivot.to_csv("bank_two_stage_mse.csv")


print("\nExperiment finished.")
print(results_df.head(100))


# =========================
# 7. Plots
# =========================

for noise in sorted(results_df["noise"].unique()):

    df_noise = results_df[results_df["noise"] == noise]

    two_stage_df = df_noise[df_noise["model"] == "two_stage"]
    clean_df = df_noise[df_noise["model"] == "clean"]
    noisy_df = df_noise[df_noise["model"] == "noisy"]

    plt.figure(figsize=(8, 5))

    for n in sorted(two_stage_df["n"].unique()):

        temp = two_stage_df[two_stage_df["n"] == n].sort_values("m")

        plt.plot(
            temp["m"],
            temp["mse"],
            marker="o",
            label=f"two-stage, n={n}"
        )

        clean_mse = clean_df[clean_df["n"] == n]["mse"].iloc[0]
        noisy_mse = noisy_df[noisy_df["n"] == n]["mse"].iloc[0]

        plt.axhline(
            clean_mse,
            linestyle="--",
            alpha=0.4
        )

        plt.axhline(
            noisy_mse,
            linestyle=":",
            alpha=0.4
        )

    plt.xlabel("m (unlabeled sample size)")
    plt.ylabel("MSE")
    plt.title(f"Two-stage MSE vs m, noise={noise}")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"bank_plot_noise_{noise}.png", dpi=300, bbox_inches="tight")
    plt.close()

print("Plots saved.")