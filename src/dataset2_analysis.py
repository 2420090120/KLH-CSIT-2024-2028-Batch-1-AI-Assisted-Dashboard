import pandas as pd
import numpy as np
import zipfile
import io
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# DATASET 2 - GLOBAL SUPERSTORE
# Preprocessing + Exploratory Data Analysis
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "datasets"
OUTPUT_DIR = BASE_DIR / "outputs"
CLEAN_DIR = OUTPUT_DIR / "cleaned"
FIG_DIR = OUTPUT_DIR / "figures"

CLEAN_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

input_file = DATA_DIR / "Dataset 2 Global_Superstore2.csv.zip"
output_file = CLEAN_DIR / "global_superstore_cleaned.csv"

print("=" * 70)
print("DATASET 2 - GLOBAL SUPERSTORE")
print("PREPROCESSING AND EDA")
print("=" * 70)

# ------------------------------------------------------------
# 1. LOAD CSV FROM ZIP
# ------------------------------------------------------------

with zipfile.ZipFile(input_file) as z:
    df = pd.read_csv(
        io.BytesIO(z.read("Global_Superstore2.csv")),
        encoding="latin1"
    )

print("\n[1] DATA LOADED")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

# ------------------------------------------------------------
# 2. STANDARDIZE COLUMN NAMES
# ------------------------------------------------------------

df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
    .str.replace(r"[^a-zA-Z0-9_]", "", regex=True)
)

# ------------------------------------------------------------
# 3. DATE CONVERSION
# ------------------------------------------------------------

df["order_date"] = pd.to_datetime(
    df["order_date"],
    dayfirst=True,
    errors="coerce"
)

df["ship_date"] = pd.to_datetime(
    df["ship_date"],
    dayfirst=True,
    errors="coerce"
)

# ------------------------------------------------------------
# 4. DATA QUALITY ASSESSMENT
# ------------------------------------------------------------

print("\n[2] DATA QUALITY ASSESSMENT")

missing = df.isnull().sum()
duplicates = df.duplicated().sum()

print("Total missing values:", int(missing.sum()))
print("Duplicate rows:", int(duplicates))

print("\nMissing values by column:")
print(missing[missing > 0])

# ------------------------------------------------------------
# 5. HANDLE HIGHLY INCOMPLETE POSTAL CODE
# ------------------------------------------------------------

postal_missing = int(df["postal_code"].isnull().sum())
postal_percentage = (postal_missing / len(df)) * 100

print("\n[3] POSTAL CODE ASSESSMENT")
print("Missing postal codes:", postal_missing)
print("Missing percentage:", round(postal_percentage, 2), "%")

# Postal Code is excluded from core analysis because of
# extensive missingness. We retain geographic fields such
# as City, State, Country, Market and Region.

df = df.drop(columns=["postal_code"])

# ------------------------------------------------------------
# 6. REMOVE EXACT DUPLICATES
# ------------------------------------------------------------

df = df.drop_duplicates()

# ------------------------------------------------------------
# 7. FEATURE ENGINEERING
# ------------------------------------------------------------

df["shipping_days"] = (
    df["ship_date"] - df["order_date"]
).dt.days

df["profit_margin"] = np.where(
    df["sales"] != 0,
    (df["profit"] / df["sales"]) * 100,
    0
)

df["order_year"] = df["order_date"].dt.year
df["order_month"] = df["order_date"].dt.month
df["order_month_name"] = df["order_date"].dt.month_name()

print("\n[4] FEATURE ENGINEERING")
print("Added:")
print("- shipping_days")
print("- profit_margin")
print("- order_year")
print("- order_month")
print("- order_month_name")

# ------------------------------------------------------------
# 8. FINAL VALIDATION
# ------------------------------------------------------------

print("\n[5] FINAL VALIDATION")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])
print("Missing values:", int(df.isnull().sum().sum()))
print("Duplicate rows:", int(df.duplicated().sum()))

# ------------------------------------------------------------
# 9. SAVE CLEAN DATASET
# ------------------------------------------------------------

df.to_csv(output_file, index=False)

print("\n[6] CLEAN DATASET SAVED")
print(output_file)

# ============================================================
# BUSINESS METRICS
# ============================================================

print("\n" + "=" * 70)
print("BUSINESS METRICS")
print("=" * 70)

print("\nTotal Sales:", round(df["sales"].sum(), 2))
print("Total Profit:", round(df["profit"].sum(), 2))
print("Total Quantity:", int(df["quantity"].sum()))
print("Average Discount:", round(df["discount"].mean(), 3))
print("Average Shipping Cost:", round(df["shipping_cost"].mean(), 2))
print("Average Profit Margin:", round(df["profit_margin"].mean(), 2), "%")

# ------------------------------------------------------------
# CHART 1 - SALES BY MARKET
# ------------------------------------------------------------

market_sales = df.groupby("market")["sales"].sum().sort_values()

plt.figure(figsize=(9, 5))
market_sales.plot(kind="barh")
plt.title("Sales by Market")
plt.xlabel("Sales")
plt.ylabel("Market")
plt.tight_layout()
plt.savefig(FIG_DIR / "06_global_sales_by_market.png", dpi=150)
plt.close()

# ------------------------------------------------------------
# CHART 2 - PROFIT BY MARKET
# ------------------------------------------------------------

market_profit = df.groupby("market")["profit"].sum().sort_values()

plt.figure(figsize=(9, 5))
market_profit.plot(kind="barh")
plt.title("Profit by Market")
plt.xlabel("Profit")
plt.ylabel("Market")
plt.tight_layout()
plt.savefig(FIG_DIR / "07_global_profit_by_market.png", dpi=150)
plt.close()

# ------------------------------------------------------------
# CHART 3 - SALES BY CATEGORY
# ------------------------------------------------------------

category_sales = df.groupby("category")["sales"].sum().sort_values()

plt.figure(figsize=(9, 5))
category_sales.plot(kind="barh")
plt.title("Global Sales by Category")
plt.xlabel("Sales")
plt.ylabel("Category")
plt.tight_layout()
plt.savefig(FIG_DIR / "08_global_sales_by_category.png", dpi=150)
plt.close()

# ------------------------------------------------------------
# CHART 4 - PROFIT BY CATEGORY
# ------------------------------------------------------------

category_profit = df.groupby("category")["profit"].sum().sort_values()

plt.figure(figsize=(9, 5))
category_profit.plot(kind="barh")
plt.title("Global Profit by Category")
plt.xlabel("Profit")
plt.ylabel("Category")
plt.tight_layout()
plt.savefig(FIG_DIR / "09_global_profit_by_category.png", dpi=150)
plt.close()

# ------------------------------------------------------------
# CHART 5 - YEARLY SALES
# ------------------------------------------------------------

yearly_sales = df.groupby("order_year")["sales"].sum()

plt.figure(figsize=(9, 5))
yearly_sales.plot(kind="line", marker="o")
plt.title("Global Yearly Sales Trend")
plt.xlabel("Year")
plt.ylabel("Sales")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(FIG_DIR / "10_global_yearly_sales.png", dpi=150)
plt.close()

# ------------------------------------------------------------
# TOP COUNTRIES BY SALES
# ------------------------------------------------------------

top_countries = (
    df.groupby("country")
    .agg(
        sales=("sales", "sum"),
        profit=("profit", "sum")
    )
    .sort_values("sales", ascending=False)
    .head(10)
)

print("\nTop 10 Countries by Sales")
print(top_countries)

# ------------------------------------------------------------
# MARKET SUMMARY
# ------------------------------------------------------------

market_summary = (
    df.groupby("market")
    .agg(
        sales=("sales", "sum"),
        profit=("profit", "sum"),
        quantity=("quantity", "sum"),
        avg_discount=("discount", "mean")
    )
    .sort_values("sales", ascending=False)
)

print("\nMarket Summary")
print(market_summary)

print("\n" + "=" * 70)
print("DATASET 2 EDA COMPLETE")
print("=" * 70)

print("\nClean dataset:")
print(output_file)

print("\nGenerated figures:")
print(FIG_DIR)
