import numpy as np
import pandas as pd
import statsmodels.api as sm

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score


# =========================
# 1. Load data
# =========================

DATA_PATH = "Data/bank-additional-full.csv"

data = pd.read_csv(DATA_PATH, sep=";")


# =========================
# 2. Clean target
# =========================

data["y"] = data["y"].map({"yes": 1, "no": 0})


# =========================
# 3. Encode month cyclically
# =========================

month_map = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4,
    "may": 5, "jun": 6, "jul": 7, "aug": 8,
    "sep": 9, "oct": 10, "nov": 11, "dec": 12
}

data["month_num"] = data["month"].map(month_map)
data["month_sin"] = np.sin(2 * np.pi * data["month_num"] / 12)
data["month_cos"] = np.cos(2 * np.pi * data["month_num"] / 12)

data = data.drop(columns=["month", "month_num"])


# =========================
# 4. Encode day cyclically
# =========================

day_map = {
    "mon": 1,
    "tue": 2,
    "wed": 3,
    "thu": 4,
    "fri": 5
}

data["day_num"] = data["day_of_week"].map(day_map)
data["day_sin"] = np.sin(2 * np.pi * data["day_num"] / 5)
data["day_cos"] = np.cos(2 * np.pi * data["day_num"] / 5)

data = data.drop(columns=["day_of_week", "day_num"])


# =========================
# 5. One-hot encode remaining categorical variables
# =========================

categorical_cols = data.select_dtypes(include=["object"]).columns.tolist()

data = pd.get_dummies(
    data,
    columns=categorical_cols,
    drop_first=True
)

# make sure everything is numeric
data = data.astype(float)


# =========================
# 6. Split X and y
# =========================

X = data.drop(columns=["y"])
y = data["y"]


# =========================
# 7. Train/test split
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)


# =========================
# 8. Scale features
# =========================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_df = pd.DataFrame(
    X_train_scaled,
    columns=X.columns,
    index=X_train.index
)

X_test_df = pd.DataFrame(
    X_test_scaled,
    columns=X.columns,
    index=X_test.index
)


# =========================
# 9. OLS academic regression
# =========================

X_train_ols = sm.add_constant(X_train_df)

ols_model = sm.OLS(y_train, X_train_ols).fit()

print("\nOLS Regression Summary")
print("======================")
print(ols_model.summary())


# =========================
# 10. Test-set prediction
# =========================

X_train_ols = sm.add_constant(X_train_df, has_constant="add")
X_test_ols = sm.add_constant(X_test_df, has_constant="add")
y_pred = ols_model.predict(X_test_ols)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nTest Performance")
print("================")
print("Test MSE:", mse)
print("Test R2:", r2)


# =========================
# 11. Save coefficient table
# =========================

coef_table = pd.DataFrame({
    "feature": ols_model.params.index,
    "coefficient": ols_model.params.values,
    "std_error": ols_model.bse.values,
    "t_stat": ols_model.tvalues.values,
    "p_value": ols_model.pvalues.values
})

coef_table["abs_coefficient"] = coef_table["coefficient"].abs()

coef_table = coef_table.sort_values(
    by="abs_coefficient",
    ascending=False
)


print("\nTop 20 coefficients:")
print(coef_table.head(20))

data.to_csv("bank_cleaned.csv", index=False)
print("Saved cleaned dataset as bank_cleaned.csv")
print("Number of features:", X.shape[1])