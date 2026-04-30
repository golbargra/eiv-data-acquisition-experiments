import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# =========================
# 1. Load data
# =========================

DATA_PATH = "Data/creditcard.csv"

data = pd.read_csv(DATA_PATH)

scaler = StandardScaler()
data[["Amount", "Time"]] = scaler.fit_transform(data[["Amount", "Time"]])

X = data.drop(columns=["Class"])
y = data["Class"]

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
)


# =========================
# 2. Feature setup
# =========================

weak_features_to_drop = ["V23"]

strong_features_to_noise = ["V17", "V14", "V12", "V10", "V16"]

X_train_clean = X_train.drop(columns=weak_features_to_drop)
X_test_clean = X_test.drop(columns=weak_features_to_drop)


# =========================
# 3. Functions
# =========================

def add_noise(X, features, noise_level, seed=42):
    X_noisy = X.copy()
    rng = np.random.default_rng(seed)

    X_noisy[features] += rng.normal(
        0,
        noise_level,
        X_noisy[features].shape
    )

    return X_noisy


#This function splits training data into a labeled set of size n and an unlabeled set of size m in a controlled way.
# It first separates indices of fraud and non-fraud samples,
# then constructs the labeled set by sampling approximately the same class proportion as the original data while forcing at least one fraud case so the model can train properly.
# After selecting these labeled indices, it removes them from the pool and randomly samples the remaining points to form the unlabeled set.
# Finally, it shuffles both sets and returns their indices, ensuring a valid, non-overlapping split between labeled and unlabeled data for your experiment.
def sample_labeled_unlabeled_indices(y_train, n, m, seed=123):
    rng = np.random.default_rng(seed)

    fraud_idx = y_train[y_train == 1].index.to_numpy()
    nonfraud_idx = y_train[y_train == 0].index.to_numpy()#so we know where fraud cases are

    fraud_rate = y_train.mean()
    n_fraud = max(1, int(round(n * fraud_rate))) #keeps class imbalance realistic, but ensures at least one fraud
    n_fraud = min(n_fraud, len(fraud_idx))

    n_nonfraud = n - n_fraud

    labeled_fraud = rng.choice(fraud_idx, size=n_fraud, replace=False)
    labeled_nonfraud = rng.choice(nonfraud_idx, size=n_nonfraud, replace=False)

    labeled_idx = np.concatenate([labeled_fraud, labeled_nonfraud])

    used = set(labeled_idx)
    remaining_idx = np.array([idx for idx in y_train.index if idx not in used])

    unlabeled_idx = rng.choice(remaining_idx, size=m, replace=False)

    rng.shuffle(labeled_idx)
    rng.shuffle(unlabeled_idx)

    return labeled_idx, unlabeled_idx


#It first combines both labeled and unlabeled data to learn a mapping from noisy features to clean features by fitting a linear regression model (the “denoiser”)
#Then, it applies this learned mapping to any given noisy dataset (X_noisy_target) to produce a denoised version of those features.
def two_stage_denoise(
    X_noisy_labeled,
    X_clean_labeled,
    X_noisy_unlabeled,
    X_clean_unlabeled,
    X_noisy_target
):
    X_noisy_stage1 = pd.concat([X_noisy_labeled, X_noisy_unlabeled])
    X_clean_stage1 = pd.concat([X_clean_labeled, X_clean_unlabeled])

    denoiser = LinearRegression()
    denoiser.fit(X_noisy_stage1, X_clean_stage1)

    X_denoised = denoiser.predict(X_noisy_target)

    return pd.DataFrame(
        X_denoised,
        columns=X_noisy_target.columns,
        index=X_noisy_target.index
    ), denoiser


#evaluation: The model is set up with class weighting to handle the strong class imbalance in fraud data. After fitting, it outputs predicted probabilities for the positive class (fraud) on the test set, and computes three evaluation metrics
def train_and_eval(X_tr, y_tr, X_te, y_te):
    model = LinearRegression()
    model.fit(X_tr, y_tr)

    y_pred = model.predict(X_te)

    return {
        "mse": mean_squared_error(y_te, y_pred)
    }


# =========================
# 4. Experiment loop
# =========================

noise_levels = [0.1, 0.3, 0.5, 1.0]
n_values = [1000, 2000, 5000, 10000]
m_values = [0, 1000, 5000, 10000, 30000]

results = []
base_seed = 123

for noise in noise_levels:
    print(f"Running noise level = {noise}")

    X_train_noisy = add_noise(
        X_train_clean,
        strong_features_to_noise,
        noise,
        seed=base_seed
    )

    X_test_noisy = add_noise(
        X_test_clean,
        strong_features_to_noise,
        noise,
        seed=base_seed + 1
    )

    for n in n_values:
        for m in m_values:

            labeled_idx, unlabeled_idx = sample_labeled_unlabeled_indices(
                y_train=y_train,
                n=n,
                m=m,
                seed=base_seed + n + m
            )

            X_clean_labeled = X_train_clean.loc[labeled_idx]
            X_noisy_labeled = X_train_noisy.loc[labeled_idx]
            y_labeled = y_train.loc[labeled_idx]

            X_clean_unlabeled = X_train_clean.loc[unlabeled_idx]
            X_noisy_unlabeled = X_train_noisy.loc[unlabeled_idx]

            # Clean benchmark
            clean_metrics = train_and_eval(
                X_clean_labeled,
                y_labeled,
                X_test_clean,
                y_test
            )

            results.append({
                "noise": noise,
                "n": n,
                "m": m,
                "model": "clean",
                **clean_metrics
            })

            # Naive noisy
            noisy_metrics = train_and_eval(
                X_noisy_labeled,
                y_labeled,
                X_test_noisy,
                y_test
            )

            results.append({
                "noise": noise,
                "n": n,
                "m": m,
                "model": "noisy",
                **noisy_metrics
            })

            # Two-stage denoised
            X_denoised_labeled, denoiser = two_stage_denoise(
                X_noisy_labeled,
                X_clean_labeled,
                X_noisy_unlabeled,
                X_clean_unlabeled,
                X_noisy_labeled
            )

            X_denoised_test = pd.DataFrame(
                denoiser.predict(X_test_noisy),
                columns=X_test_noisy.columns,
                index=X_test_noisy.index
            )

            denoised_metrics = train_and_eval(
                X_denoised_labeled,
                y_labeled,
                X_denoised_test,
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
# 5. Save results
# =========================

results_df = pd.DataFrame(results)

results_df.to_csv("experiment_results.csv", index=False)

pivot_mse = results_df.pivot_table(
    index=["noise", "n", "m"],
    columns="model",
    values="mse"
)

pivot_mse.to_csv("pivot_mse.csv")

print("\nExperiment finished.")
print(results_df.head(300))

import matplotlib.pyplot as plt

models = results_df["model"].unique()
noise_levels = sorted(results_df["noise"].unique())

for noise in noise_levels:
    print(f"Plotting noise = {noise}")

    df_noise = results_df[results_df["noise"] == noise]

    # one figure per noise
    plt.figure(figsize=(12, 4))

    for i, model in enumerate(models):
        plt.subplot(1, len(models), i+1)

        df_model = df_noise[df_noise["model"] == model]

        # plot one line per n
        for n in sorted(df_model["n"].unique()):
            temp = df_model[df_model["n"] == n].sort_values("m")

            plt.plot(temp["m"], temp["mse"], marker='o', label=f"n={n}")

        plt.title(f"{model} (noise={noise})")
        plt.xlabel("m")
        plt.ylabel("MSE")
        plt.grid()

        if i == 0:
            plt.legend()

    plt.tight_layout()
    plt.savefig(f"plot_noise_{noise}.png", dpi=300, bbox_inches="tight")
    plt.close()