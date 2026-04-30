import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# =========================
# 1. Generate synthetic EIV data
# =========================

def generate_data(N=100000, d_x=10, d_w=5, noise_w=1.0, noise_y=0.5, seed=42):
    rng = np.random.default_rng(seed)

    X = rng.normal(0, 1, size=(N, d_x))

    G = rng.normal(0, 1, size=(d_x, d_w))
    beta = rng.normal(0, 1, size=d_w)

    W_clean = X @ G

    W_noisy = W_clean + rng.normal(0, noise_w, size=W_clean.shape)

    y = W_clean @ beta + rng.normal(0, noise_y, size=N)

    return X, W_clean, W_noisy, y


# =========================
# 2. Two-stage denoising
# =========================

def two_stage_denoise(X_stage1, W_noisy_stage1, X_target):
    denoiser = LinearRegression()
    denoiser.fit(X_stage1, W_noisy_stage1)

    W_hat = denoiser.predict(X_target)

    return W_hat, denoiser


# =========================
# 3. Train and evaluate
# =========================

def train_and_eval(W_train, y_train, W_test, y_test):
    model = LinearRegression()
    model.fit(W_train, y_train)

    y_pred = model.predict(W_test)

    return mean_squared_error(y_test, y_pred)


# =========================
# 4. Experiment loop
# =========================

noise_levels = [0.25, 0.5, 1.0, 2.0]
n_values = [100, 250, 500, 1000, 2000]
m_values = [0, 500, 1000, 5000, 10000, 30000]

results = []

base_seed = 123

for noise_w in noise_levels:
    print(f"Running noise level = {noise_w}")

    X, W_clean, W_noisy, y = generate_data(
        N=100000,
        d_x=10,
        d_w=5,
        noise_w=noise_w,
        noise_y=0.5,
        seed=base_seed
    )

    # train/test split
    N = len(y)
    test_size = 20000

    train_idx = np.arange(0, N - test_size)
    test_idx = np.arange(N - test_size, N)

    X_train = X[train_idx]
    W_clean_train = W_clean[train_idx]
    W_noisy_train = W_noisy[train_idx]
    y_train = y[train_idx]

    X_test = X[test_idx]
    W_clean_test = W_clean[test_idx]
    W_noisy_test = W_noisy[test_idx]
    y_test = y[test_idx]

    for n in n_values:
        for m in m_values:

            if n + m > len(y_train):
                continue

            rng = np.random.default_rng(base_seed + n + m)

            selected_idx = rng.choice(len(y_train), size=n + m, replace=False)

            labeled_idx = selected_idx[:n]
            unlabeled_idx = selected_idx[n:]

            # labeled data
            X_labeled = X_train[labeled_idx]
            W_clean_labeled = W_clean_train[labeled_idx]
            W_noisy_labeled = W_noisy_train[labeled_idx]
            y_labeled = y_train[labeled_idx]

            # unlabeled data
            X_unlabeled = X_train[unlabeled_idx]
            W_noisy_unlabeled = W_noisy_train[unlabeled_idx]

            # =========================
            # Model 1: clean benchmark
            # =========================
            clean_mse = train_and_eval(
                W_clean_labeled,
                y_labeled,
                W_clean_test,
                y_test
            )

            results.append({
                "noise": noise_w,
                "n": n,
                "m": m,
                "model": "clean",
                "mse": clean_mse
            })

            # =========================
            # Model 2: naive noisy
            # =========================
            noisy_mse = train_and_eval(
                W_noisy_labeled,
                y_labeled,
                W_noisy_test,
                y_test
            )

            results.append({
                "noise": noise_w,
                "n": n,
                "m": m,
                "model": "noisy",
                "mse": noisy_mse
            })

            # =========================
            # Model 3: two-stage denoised
            # =========================

            X_stage1 = np.vstack([X_labeled, X_unlabeled])
            W_noisy_stage1 = np.vstack([W_noisy_labeled, W_noisy_unlabeled])

            W_hat_labeled, denoiser = two_stage_denoise(
                X_stage1,
                W_noisy_stage1,
                X_labeled
            )

            W_hat_test = denoiser.predict(X_test)

            denoised_mse = train_and_eval(
                W_hat_labeled,
                y_labeled,
                W_hat_test,
                y_test
            )

            results.append({
                "noise": noise_w,
                "n": n,
                "m": m,
                "model": "two_stage",
                "mse": denoised_mse
            })


# =========================
# 5. Save results
# =========================

results_df = pd.DataFrame(results)
results_df.to_csv("synthetic_experiment_results.csv", index=False)

pivot_mse = results_df.pivot_table(
    index=["noise", "n", "m"],
    columns="model",
    values="mse"
)

pivot_mse.to_csv("synthetic_pivot_mse.csv")

print("Experiment finished.")
print(results_df.head())


# =========================
# 6. Plots
# =========================

models = results_df["model"].unique()

for noise in sorted(results_df["noise"].unique()):
    df_noise = results_df[results_df["noise"] == noise]

    plt.figure(figsize=(12, 4))

    for i, model in enumerate(models):
        plt.subplot(1, len(models), i + 1)

        df_model = df_noise[df_noise["model"] == model]

        for n in sorted(df_model["n"].unique()):
            temp = df_model[df_model["n"] == n].sort_values("m")
            plt.plot(temp["m"], temp["mse"], marker="o", label=f"n={n}")

        plt.title(f"{model}, noise={noise}")
        plt.xlabel("m")
        plt.ylabel("MSE")
        plt.grid(True)

        if i == 0:
            plt.legend()

    plt.tight_layout()
    plt.savefig(f"synthetic_plot_noise_{noise}.png", dpi=300, bbox_inches="tight")
    plt.close()


print("Plots saved.")