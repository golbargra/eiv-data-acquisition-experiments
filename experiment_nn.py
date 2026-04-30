import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import mean_squared_error, roc_auc_score, average_precision_score


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


weak_features_to_drop = ["V23"]
strong_features_to_noise = ["V17", "V14", "V12", "V10", "V16"]

X_train_clean = X_train.drop(columns=weak_features_to_drop)
X_test_clean = X_test.drop(columns=weak_features_to_drop)


def add_nonlinear_noise(X, features, noise_level, seed=42):
    X_noisy = X.copy()
    rng = np.random.default_rng(seed)

    for f in features:
        X_noisy[f] += noise_level * np.sin(X_noisy[f]) \
                      + 0.2 * (X_noisy[f] ** 2)
    return X_noisy


def sample_labeled_unlabeled_indices(y_train, n, m, seed=123):
    rng = np.random.default_rng(seed)

    fraud_idx = y_train[y_train == 1].index.to_numpy()
    nonfraud_idx = y_train[y_train == 0].index.to_numpy()

    fraud_rate = y_train.mean()   # ✅ ADD THIS

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


def train_and_eval_nn(X_tr, y_tr, X_te, y_te):
    model = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        max_iter=100,
        random_state=42,
        early_stopping=True
    )

    model.fit(X_tr, y_tr)

    prob = model.predict_proba(X_te)[:, 1]

    return {
        "mse": mean_squared_error(y_te, prob),
        "auc": roc_auc_score(y_te, prob),
        "auprc": average_precision_score(y_te, prob)
    }


noise_levels = [0.1, 0.3, 0.5, 1.0]
n_values = [1000, 2000, 5000, 10000]
m_values = [0, 1000, 5000, 10000, 30000]

results = []
base_seed = 123

for noise in noise_levels:
    print(f"Running noise level = {noise}")

    X_train_noisy = add_nonlinear_noise(
        X_train_clean,
        strong_features_to_noise,
        noise,
        seed=base_seed
    )

    X_test_noisy = add_nonlinear_noise(
        X_test_clean,
        strong_features_to_noise,
        noise,
        seed=base_seed + 1
    )

    for n in n_values:
        for m in m_values:
            print(f"noise={noise}, n={n}, m={m}")

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

            # clean benchmark
            clean_metrics = train_and_eval_nn(
                X_clean_labeled,
                y_labeled,
                X_test_clean,
                y_test
            )

            results.append({
                "noise": noise,
                "n": n,
                "m": m,
                "model": "clean_nn",
                **clean_metrics
            })

            # naive noisy
            noisy_metrics = train_and_eval_nn(
                X_noisy_labeled,
                y_labeled,
                X_test_noisy,
                y_test
            )

            results.append({
                "noise": noise,
                "n": n,
                "m": m,
                "model": "noisy_nn",
                **noisy_metrics
            })

            # two-stage denoised
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

            denoised_metrics = train_and_eval_nn(
                X_denoised_labeled,
                y_labeled,
                X_denoised_test,
                y_test
            )

            results.append({
                "noise": noise,
                "n": n,
                "m": m,
                "model": "two_stage_nn",
                **denoised_metrics
            })


results_df = pd.DataFrame(results)

results_df.to_csv("nn_experiment_results.csv", index=False)

pivot_mse = results_df.pivot_table(
    index=["noise", "n", "m"],
    columns="model",
    values="mse"
)

pivot_mse.to_csv("nn_pivot_mse.csv")


# plots: one plot per noise level
models = results_df["model"].unique()
noise_levels_plot = sorted(results_df["noise"].unique())

for noise in noise_levels_plot:
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
    plt.savefig(f"nn_plot_noise_{noise}.png", dpi=300, bbox_inches="tight")
    plt.close()


print("NN experiment finished.")
print(results_df.head())