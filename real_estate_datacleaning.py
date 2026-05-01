import pandas as pd


# =========================
# 1. Load datasets
# =========================

pre = pd.read_csv("Data/pre_owned_house_transactions.csv")
pre_near = pd.read_csv("Data/pre_owned_house_transactions_nearby_sectors.csv")

new = pd.read_csv("Data/new_house_transactions.csv")
new_near = pd.read_csv("Data/new_house_transactions_nearby_sectors.csv")

land = pd.read_csv("Data/land_transactions.csv")
land_near = pd.read_csv("Data/land_transactions_nearby_sectors.csv")

poi = pd.read_csv("sector_POI.csv")


# =========================
# 2. Check shapes
# =========================

print("Pre-owned:", pre.shape)
print("Pre-owned nearby:", pre_near.shape)
print("New house:", new.shape)
print("New house nearby:", new_near.shape)
print("Land:", land.shape)
print("Land nearby:", land_near.shape)
print("POI:", poi.shape)


# =========================
# 3. Merge sector-month datasets
# =========================

df = pre.copy()

df = df.merge(
    pre_near,
    on=["month", "sector"],
    how="left"
)

df = df.merge(
    new,
    on=["month", "sector"],
    how="left"
)

df = df.merge(
    new_near,
    on=["month", "sector"],
    how="left"
)

df = df.merge(
    land,
    on=["month", "sector"],
    how="left"
)

df = df.merge(
    land_near,
    on=["month", "sector"],
    how="left"
)


# =========================
# 4. Merge sector-level POI
# =========================

df = df.merge(
    poi,
    on="sector",
    how="left"
)


# =========================
# 5. Clean missing values
# =========================

# Replace missing numeric values with column means
numeric_cols = df.select_dtypes(include=["number"]).columns

df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())


# =========================
# 6. Save merged dataset
# =========================

df.to_csv("real_estate_clean_merged.csv", index=False)


# =========================
# 7. Inspect final dataset
# =========================

print("\nFinal merged dataset")
print("====================")
print("Shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst rows:")
print(df.head())

print("\nMissing values:")
print(df.isna().sum().sort_values(ascending=False).head(20))

print("\nBasic statistics:")
print(df.describe().T.head(30))

