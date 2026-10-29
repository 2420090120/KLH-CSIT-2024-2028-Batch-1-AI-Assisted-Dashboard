import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# AI ASSISTED DATA VISUALIZATION
# Stage 1: Data Preprocessing + Exploratory Data Analysis
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "datasets"
OUTPUT_DIR = BASE_DIR / "outputs"
CLEAN_DIR = OUTPUT_DIR / "cleaned"
FIG_DIR = OUTPUT_DIR / "figures"

CLEAN_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

input_file = DATA_DIR / "Dataset 1 sample_-_superstore.xls"
output_file = CLEAN_DIR / "superstore_cleaned.csv"

print("=" * 70)
print("AI ASSISTED DATA VISUALIZATION")
print("STAGE 1 - DATA PREPROCESSING AND EDA")
print("=" * 70)

# ------------------------------------------------------------
# 1. LOAD DATA
# ------------------------------------------------------------

df = pd.read_excel(input_file)

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
# 3. DATA TYPE CONVERSION
# ------------------------------------------------------------

df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
df["ship_date"] = pd.to_datetime(df["ship_date"], errors="coerce")

# ------------------------------------------------------------
# 4. DATA QUALITY CHECK
# ------------------------------------------------------------

missing_before = int(df.isnull().sum().sum())
duplicates_before = int(df.duplicated().sum())

print("\n[2] DATA QUALITY CHECK")
print("Missing values:", missing_before)
print("Duplicate rows:", duplicates_before)

# Remove exact duplicate rows if any
df = df.drop_duplicates()

# ------------------------------------------------------------
# 5. FEATURE ENGINEERING
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

print("\n[3] FEATURE ENGINEERING")
print("Added:")
print("- shipping_days")
print("- profit_margin")
print("- order_year")
print("- order_month")
print("- order_month_name")

# ------------------------------------------------------------
# 6. FINAL VALIDATION
# ------------------------------------------------------------

missing_after = int(df.isnull().sum().sum())

print("\n[4] FINAL DATASET VALIDATION")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])
print("Missing values:", missing_after)
print("Duplicate rows:", df.duplicated().sum())

# ------------------------------------------------------------
# 7. SAVE CLEAN DATASET
# ------------------------------------------------------------

df.to_csv(output_file, index=False)

print("\n[5] CLEAN DATASET SAVED")
print(output_file)

# ============================================================
# EDA
# ============================================================

print("\n" + "=" * 70)
print("EXPLORATORY DATA ANALYSIS")
print("=" * 70)

print("\nOverall Business Metrics")
print("-" * 40)
print("Total Sales:  ", round(df["sales"].sum(), 2))
print("Total Profit: ", round(df["profit"].sum(), 2))
print("Total Quantity:", int(df["quantity"].sum()))
print("Average Discount:", round(df["discount"].mean(), 3))
print("Average Profit Margin:", round(df["profit_margin"].mean(), 2), "%")

# ------------------------------------------------------------
# CHART 1: SALES BY CATEGORY
# ------------------------------------------------------------

category_sales = df.groupby("category")["sales"].sum().sort_values()

plt.figure(figsize=(9, 5))
category_sales.plot(kind="barh")
plt.title("Sales by Product Category")
plt.xlabel("Sales")
plt.ylabel("Category")
plt.tight_layout()
plt.savefig(FIG_DIR / "01_sales_by_category.png", dpi=150)
plt.close()

# ------------------------------------------------------------
# CHART 2: PROFIT BY CATEGORY
# ------------------------------------------------------------

category_profit = df.groupby("category")["profit"].sum().sort_values()

plt.figure(figsize=(9, 5))
category_profit.plot(kind="barh")
plt.title("Profit by Product Category")
plt.xlabel("Profit")
plt.ylabel("Category")
plt.tight_layout()
plt.savefig(FIG_DIR / "02_profit_by_category.png", dpi=150)
plt.close()

# ------------------------------------------------------------
# CHART 3: SALES BY REGION
# ------------------------------------------------------------

region_sales = df.groupby("region")["sales"].sum().sort_values()

plt.figure(figsize=(9, 5))
region_sales.plot(kind="bar")
plt.title("Sales by Region")
plt.xlabel("Region")
plt.ylabel("Sales")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(FIG_DIR / "03_sales_by_region.png", dpi=150)
plt.close()

# ------------------------------------------------------------
# CHART 4: PROFIT BY REGION
# ------------------------------------------------------------

region_profit = df.groupby("region")["profit"].sum().sort_values()

plt.figure(figsize=(9, 5))
region_profit.plot(kind="bar")
plt.title("Profit by Region")
plt.xlabel("Region")
plt.ylabel("Profit")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(FIG_DIR / "04_profit_by_region.png", dpi=150)
plt.close()

# ------------------------------------------------------------
# CHART 5: YEARLY SALES
# ------------------------------------------------------------

yearly_sales = df.groupby("order_year")["sales"].sum()

plt.figure(figsize=(9, 5))
yearly_sales.plot(kind="line", marker="o")
plt.title("Yearly Sales Trend")
plt.xlabel("Year")
plt.ylabel("Sales")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(FIG_DIR / "05_yearly_sales_trend.png", dpi=150)
plt.close()

# ------------------------------------------------------------
# TOP PRODUCTS
# ------------------------------------------------------------

top_products = (
    df.groupby("product_name")
    .agg(
        sales=("sales", "sum"),
        profit=("profit", "sum")
    )
    .sort_values("sales", ascending=False)
    .head(10)
)

print("\nTop 10 Products by Sales")
print(top_products)

# ------------------------------------------------------------
# CATEGORY SUMMARY
# ------------------------------------------------------------

category_summary = (
    df.groupby("category")
    .agg(
        sales=("sales", "sum"),
        profit=("profit", "sum"),
        quantity=("quantity", "sum"),
        avg_discount=("discount", "mean")
    )
    .sort_values("sales", ascending=False)
)

print("\nCategory Summary")
print(category_summary)

# ------------------------------------------------------------
# REGION SUMMARY
# ------------------------------------------------------------

region_summary = (
    df.groupby("region")
    .agg(
        sales=("sales", "sum"),
        profit=("profit", "sum"),
        quantity=("quantity", "sum")
    )
    .sort_values("sales", ascending=False)
)

print("\nRegion Summary")
print(region_summary)

print("\n" + "=" * 70)
print("EDA COMPLETE")
print("=" * 70)

print("\nGenerated files:")
print("Clean dataset:", output_file)
print("Charts:", FIG_DIR)
