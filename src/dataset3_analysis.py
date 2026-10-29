import pandas as pd
import zipfile
import io
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# DATASET 3 - OLIST CUSTOMER DATASET
# Preprocessing + Exploratory Data Analysis
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "datasets"
OUTPUT_DIR = BASE_DIR / "outputs"
CLEAN_DIR = OUTPUT_DIR / "cleaned"
FIG_DIR = OUTPUT_DIR / "figures"

CLEAN_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

input_file = DATA_DIR / "Dataset 3 olist_customers_dataset.csv.zip"
output_file = CLEAN_DIR / "olist_customers_cleaned.csv"

print("=" * 70)
print("DATASET 3 - OLIST CUSTOMER DATASET")
print("PREPROCESSING AND EDA")
print("=" * 70)

# ------------------------------------------------------------
# 1. LOAD DATA
# ------------------------------------------------------------

with zipfile.ZipFile(input_file) as z:
    df = pd.read_csv(
        io.BytesIO(z.read("olist_customers_dataset.csv"))
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
# 3. DATA QUALITY ASSESSMENT
# ------------------------------------------------------------

print("\n[2] DATA QUALITY ASSESSMENT")

missing = df.isnull().sum()
duplicates = df.duplicated().sum()

print("Total missing values:", int(missing.sum()))
print("Duplicate rows:", int(duplicates))

# ------------------------------------------------------------
# 4. DUPLICATE REMOVAL
# ------------------------------------------------------------

df = df.drop_duplicates()

# ------------------------------------------------------------
# 5. CUSTOMER ID VALIDATION
# ------------------------------------------------------------

unique_customer_ids = df["customer_id"].nunique()
unique_customer_profiles = df["customer_unique_id"].nunique()

print("\n[3] CUSTOMER IDENTIFIER ANALYSIS")
print("Unique customer_id:", unique_customer_ids)
print("Unique customer_unique_id:", unique_customer_profiles)

# ------------------------------------------------------------
# 6. GEOGRAPHIC FEATURES
# ------------------------------------------------------------

df["customer_state"] = df["customer_state"].str.upper()
df["customer_city"] = df["customer_city"].str.strip().str.title()

df["zip_prefix"] = (
    df["customer_zip_code_prefix"]
    .astype(str)
    .str.zfill(5)
)

print("\n[4] FEATURE ENGINEERING")
print("Added:")
print("- zip_prefix")
print("- standardized customer_state")
print("- standardized customer_city")

# ------------------------------------------------------------
# 7. FINAL VALIDATION
# ------------------------------------------------------------

print("\n[5] FINAL VALIDATION")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])
print("Missing values:", int(df.isnull().sum().sum()))
print("Duplicate rows:", int(df.duplicated().sum()))

# ------------------------------------------------------------
# 8. SAVE CLEAN DATASET
# ------------------------------------------------------------

df.to_csv(output_file, index=False)

print("\n[6] CLEAN DATASET SAVED")
print(output_file)

# ============================================================
# GEOGRAPHIC EDA
# ============================================================

print("\n" + "=" * 70)
print("GEOGRAPHIC EXPLORATORY DATA ANALYSIS")
print("=" * 70)

# ------------------------------------------------------------
# STATE SUMMARY
# ------------------------------------------------------------

state_summary = (
    df.groupby("customer_state")
    .size()
    .sort_values(ascending=False)
)

print("\nTop Customer States")
print(state_summary.head(10))

# ------------------------------------------------------------
# CITY SUMMARY
# ------------------------------------------------------------

city_summary = (
    df.groupby("customer_city")
    .size()
    .sort_values(ascending=False)
)

print("\nTop Customer Cities")
print(city_summary.head(10))

# ------------------------------------------------------------
# CHART 1 - CUSTOMERS BY STATE
# ------------------------------------------------------------

plt.figure(figsize=(10, 6))
state_summary.head(10).sort_values().plot(kind="barh")
plt.title("Top 10 States by Customer Count")
plt.xlabel("Number of Customers")
plt.ylabel("State")
plt.tight_layout()
plt.savefig(FIG_DIR / "11_olist_customers_by_state.png", dpi=150)
plt.close()

# ------------------------------------------------------------
# CHART 2 - CUSTOMERS BY CITY
# ------------------------------------------------------------

plt.figure(figsize=(10, 6))
city_summary.head(10).sort_values().plot(kind="barh")
plt.title("Top 10 Cities by Customer Count")
plt.xlabel("Number of Customers")
plt.ylabel("City")
plt.tight_layout()
plt.savefig(FIG_DIR / "12_olist_customers_by_city.png", dpi=150)
plt.close()

# ------------------------------------------------------------
# CHART 3 - CUSTOMER UNIQUE ID DISTRIBUTION
# ------------------------------------------------------------

customer_profile_counts = (
    df.groupby("customer_unique_id")
    .size()
    .value_counts()
    .sort_index()
)

plt.figure(figsize=(9, 5))
customer_profile_counts.head(10).plot(kind="bar")
plt.title("Customer Record Frequency")
plt.xlabel("Number of Records per Customer Profile")
plt.ylabel("Number of Customer Profiles")
plt.tight_layout()
plt.savefig(FIG_DIR / "13_olist_customer_record_frequency.png", dpi=150)
plt.close()

print("\n" + "=" * 70)
print("DATASET 3 EDA COMPLETE")
print("=" * 70)

print("\nClean dataset:")
print(output_file)

print("\nGenerated figures:")
print(FIG_DIR)
