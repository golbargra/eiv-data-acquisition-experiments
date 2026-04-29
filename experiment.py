import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, roc_auc_score, average_precision_score


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


def sample_labeled_unlabeled_indices(y_train, n, m, seed=123):
    rng = np.random.default_rng(seed)

    fraud_idx = y_train[y_train == 1].index.to_numpy()
    nonfraud_idx = y_train[y_train == 0].index.to_numpy()

    fraud_rate = y_train.mean()
    n_fraud = max(1, int(round(n * fraud_rate)))
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


def train_and_eval(X_tr, y_tr, X_te, y_te):
    model = LogisticRegression(
        max_iter=3000,
        class_weight="balanced",
        solver="lbfgs"
    )

    model.fit(X_tr, y_tr)

    prob = model.predict_proba(X_te)[:, 1]

    return {
        "mse": mean_squared_error(y_te, prob),
        "auc": roc_auc_score(y_te, prob),
        "auprc": average_precision_score(y_te, prob)
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