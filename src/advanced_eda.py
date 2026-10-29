import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# ADVANCED EXPLORATORY DATA ANALYSIS
# AI-ASSISTED DATA VISUALIZATION PROJECT
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
CLEAN_DIR = BASE_DIR / "outputs" / "cleaned"
FIG_DIR = BASE_DIR / "outputs" / "figures"
RESULT_DIR = BASE_DIR / "outputs" / "results"

FIG_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("ADVANCED EXPLORATORY DATA ANALYSIS")
print("=" * 70)


# ============================================================
# FUNCTION 1 - OUTLIER DETECTION
# ============================================================

def detect_outliers(df, column, dataset_name):

    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)

    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = df[
        (df[column] < lower_bound) |
        (df[column] > upper_bound)
    ]

    print(f"\n{dataset_name} - {column}")
    print("Q1:", round(Q1, 2))
    print("Q3:", round(Q3, 2))
    print("IQR:", round(IQR, 2))
    print("Lower Bound:", round(lower_bound, 2))
    print("Upper Bound:", round(upper_bound, 2))
    print("Outliers:", len(outliers))

    return len(outliers)


# ============================================================
# DATASET 1 - SUPERSTORE
# ============================================================

print("\n" + "=" * 70)
print("DATASET 1 - SUPERSTORE")
print("=" * 70)

superstore_file = CLEAN_DIR / "superstore_cleaned.csv"
superstore = pd.read_csv(superstore_file)

print("\nRows:", len(superstore))
print("Columns:", len(superstore.columns))


# ------------------------------------------------------------
# 1. DESCRIPTIVE STATISTICS
# ------------------------------------------------------------

print("\n[1] DESCRIPTIVE STATISTICS")

print(
    superstore[
        ["sales", "profit", "quantity", "discount"]
    ].describe()
)


# ------------------------------------------------------------
# 2. OUTLIER DETECTION
# ------------------------------------------------------------

print("\n[2] OUTLIER DETECTION")

for column in ["sales", "profit", "quantity", "discount"]:

    detect_outliers(
        superstore,
        column,
        "Superstore"
    )


# ------------------------------------------------------------
# 3. UNIVARIATE ANALYSIS
# ------------------------------------------------------------

print("\n[3] UNIVARIATE ANALYSIS")

plt.figure(figsize=(9, 5))
superstore["sales"].plot(kind="hist", bins=30)
plt.title("Superstore Sales Distribution")
plt.xlabel("Sales")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(
    FIG_DIR / "14_superstore_sales_distribution.png",
    dpi=150
)
plt.close()


plt.figure(figsize=(9, 5))
superstore["profit"].plot(kind="hist", bins=30)
plt.title("Superstore Profit Distribution")
plt.xlabel("Profit")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(
    FIG_DIR / "15_superstore_profit_distribution.png",
    dpi=150
)
plt.close()


# ------------------------------------------------------------
# 4. BIVARIATE ANALYSIS
# ------------------------------------------------------------

print("\n[4] BIVARIATE ANALYSIS")

plt.figure(figsize=(9, 5))
plt.scatter(
    superstore["sales"],
    superstore["profit"],
    alpha=0.5
)

plt.title("Sales vs Profit")
plt.xlabel("Sales")
plt.ylabel("Profit")
plt.tight_layout()

plt.savefig(
    FIG_DIR / "16_superstore_sales_vs_profit.png",
    dpi=150
)

plt.close()


# ------------------------------------------------------------
# 5. MULTIVARIATE ANALYSIS
# ------------------------------------------------------------

print("\n[5] MULTIVARIATE ANALYSIS")

multivariate = (
    superstore
    .groupby("category")
    [["sales", "profit", "quantity", "discount"]]
    .mean()
)

print(multivariate)


# ------------------------------------------------------------
# 6. CORRELATION ANALYSIS
# ------------------------------------------------------------

print("\n[6] CORRELATION ANALYSIS")

corr = (
    superstore[
        ["sales", "profit", "quantity", "discount"]
    ]
    .corr()
)

print(corr)

corr.to_csv(
    RESULT_DIR / "superstore_correlation.csv"
)


# ============================================================
# DATASET 2 - GLOBAL SUPERSTORE
# ============================================================

print("\n" + "=" * 70)
print("DATASET 2 - GLOBAL SUPERSTORE")
print("=" * 70)

global_file = CLEAN_DIR / "global_superstore_cleaned.csv"
global_df = pd.read_csv(global_file)

print("\nRows:", len(global_df))
print("Columns:", len(global_df.columns))


# ------------------------------------------------------------
# 7. DESCRIPTIVE STATISTICS
# ------------------------------------------------------------

print("\n[7] DESCRIPTIVE STATISTICS")

print(
    global_df[
        ["sales", "profit", "quantity", "discount", "shipping_cost"]
    ].describe()
)


# ------------------------------------------------------------
# 8. OUTLIER DETECTION
# ------------------------------------------------------------

print("\n[8] OUTLIER DETECTION")

for column in [
    "sales",
    "profit",
    "quantity",
    "discount",
    "shipping_cost"
]:

    detect_outliers(
        global_df,
        column,
        "Global Superstore"
    )


# ------------------------------------------------------------
# 9. UNIVARIATE ANALYSIS
# ------------------------------------------------------------

print("\n[9] UNIVARIATE ANALYSIS")

plt.figure(figsize=(9, 5))
global_df["sales"].plot(kind="hist", bins=30)

plt.title("Global Superstore Sales Distribution")
plt.xlabel("Sales")
plt.ylabel("Frequency")
plt.tight_layout()

plt.savefig(
    FIG_DIR / "17_global_sales_distribution.png",
    dpi=150
)

plt.close()


# ------------------------------------------------------------
# 10. BIVARIATE ANALYSIS
# ------------------------------------------------------------

print("\n[10] BIVARIATE ANALYSIS")

plt.figure(figsize=(9, 5))

plt.scatter(
    global_df["sales"],
    global_df["profit"],
    alpha=0.5
)

plt.title("Global Superstore Sales vs Profit")
plt.xlabel("Sales")
plt.ylabel("Profit")
plt.tight_layout()

plt.savefig(
    FIG_DIR / "18_global_sales_vs_profit.png",
    dpi=150
)

plt.close()


# ------------------------------------------------------------
# 11. MULTIVARIATE ANALYSIS
# ------------------------------------------------------------

print("\n[11] MULTIVARIATE ANALYSIS")

global_multivariate = (
    global_df
    .groupby("market")
    [["sales", "profit", "quantity", "discount"]]
    .mean()
)

print(global_multivariate)


# ------------------------------------------------------------
# 12. CORRELATION ANALYSIS
# ------------------------------------------------------------

print("\n[12] CORRELATION ANALYSIS")

global_corr = (
    global_df[
        [
            "sales",
            "profit",
            "quantity",
            "discount",
            "shipping_cost"
        ]
    ]
    .corr()
)

print(global_corr)

global_corr.to_csv(
    RESULT_DIR / "global_superstore_correlation.csv"
)


# ============================================================
# DATASET 3 - OLIST
# ============================================================

print("\n" + "=" * 70)
print("DATASET 3 - OLIST")
print("=" * 70)

olist_file = CLEAN_DIR / "olist_customers_cleaned.csv"
olist = pd.read_csv(olist_file)

print("\nRows:", len(olist))
print("Columns:", len(olist.columns))


# ------------------------------------------------------------
# 13. CUSTOMER DISTRIBUTION
# ------------------------------------------------------------

print("\n[13] CUSTOMER DISTRIBUTION")

state_counts = (
    olist["customer_state"]
    .value_counts()
)

print(state_counts.head(10))


# ------------------------------------------------------------
# 14. UNIVARIATE ANALYSIS
# ------------------------------------------------------------

plt.figure(figsize=(9, 5))

state_counts.head(10).sort_values().plot(
    kind="barh"
)

plt.title("Top Customer States")
plt.xlabel("Number of Customers")
plt.ylabel("State")
plt.tight_layout()

plt.savefig(
    FIG_DIR / "19_olist_state_distribution.png",
    dpi=150
)

plt.close()


# ------------------------------------------------------------
# 15. CUSTOMER PROFILE FREQUENCY
# ------------------------------------------------------------

profile_frequency = (
    olist
    .groupby("customer_unique_id")
    .size()
)

print("\nCustomer profile frequency:")
print(profile_frequency.describe())


plt.figure(figsize=(9, 5))

profile_frequency.value_counts().sort_index().head(10).plot(
    kind="bar"
)

plt.title("Customer Profile Frequency")
plt.xlabel("Number of Records")
plt.ylabel("Customer Profiles")
plt.tight_layout()

plt.savefig(
    FIG_DIR / "20_olist_profile_frequency.png",
    dpi=150
)

plt.close()


# ============================================================
# FINAL MESSAGE
# ============================================================

print("\n" + "=" * 70)
print("ADVANCED EDA COMPLETE")
print("=" * 70)

print("\nGenerated advanced EDA figures:")
print(FIG_DIR)

print("\nCorrelation results:")
print(RESULT_DIR)