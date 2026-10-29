# ============================================================
# FEATURE SELECTION
# Project: AI-Assisted Data Visualization for Business
# Decision Support Systems
#
# Dataset: Superstore Sales Dataset
# Target: Profit
# ============================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.feature_selection import mutual_info_regression
from sklearn.ensemble import RandomForestRegressor


# ============================================================
# 1. PROJECT PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

INPUT_FILE = os.path.join(
    BASE_DIR,
    "outputs",
    "cleaned",
    "superstore_cleaned.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "outputs",
    "results"
)

FIGURE_DIR = os.path.join(
    BASE_DIR,
    "outputs",
    "figures"
)

# Create output folders if they do not exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)


# ============================================================
# 2. LOAD CLEANED DATASET
# ============================================================

print("=" * 70)
print("FEATURE SELECTION - SUPERSTORE DATASET")
print("=" * 70)

df = pd.read_csv(INPUT_FILE)

print("\nDataset loaded successfully.")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

print("\nAvailable columns:")
print(df.columns.tolist())


# ============================================================
# 3. DEFINE TARGET VARIABLE
# ============================================================

TARGET = "profit"

if TARGET not in df.columns:
    raise ValueError(
        f"Target column '{TARGET}' was not found in the dataset."
    )

print("\nTarget variable:", TARGET)


# ============================================================
# 4. DEFINE CANDIDATE FEATURES
# ============================================================

# These features are selected because they contain
# potentially useful business information for predicting profit.

candidate_features = [
    "sales",
    "quantity",
    "discount",
    "shipping_days",
    "category",
    "subcategory",
    "region",
    "stateprovince",
    "segment",
    "ship_mode",
    "order_year",
    "order_month"
]


# Keep only columns that actually exist in the dataset.

features = [
    column
    for column in candidate_features
    if column in df.columns
]

print("\nCandidate features selected:")

for feature in features:
    print(" -", feature)


# ============================================================
# 5. REMOVE TARGET LEAKAGE
# ============================================================

# profit is our target, so it must not be included as a feature.
#
# profit_margin was created using profit.
# Therefore, using profit_margin to predict profit would
# introduce target leakage.

leakage_columns = [
    "profit",
    "profit_margin"
]

features = [
    column
    for column in features
    if column not in leakage_columns
]

print("\nFeatures after removing target leakage:")

for feature in features:
    print(" -", feature)


# ============================================================
# 6. CREATE X AND y
# ============================================================

X = df[features].copy()

y = df[TARGET].copy()


# ============================================================
# 7. HANDLE MISSING VALUES
# ============================================================

print("\nChecking and handling missing values...")

for column in X.columns:

    # Correctly detect numeric columns, including
    # Pandas nullable numeric dtypes.

    if pd.api.types.is_numeric_dtype(X[column]):

        # Replace missing numeric values with median.

        X[column] = X[column].fillna(
            X[column].median()
        )

    else:

        # Replace missing categorical/text values
        # with "Unknown".

        X[column] = X[column].fillna(
            "Unknown"
        )


# Handle missing values in target.

if pd.api.types.is_numeric_dtype(y):

    y = y.fillna(
        y.median()
    )


# ============================================================
# 8. IDENTIFY NUMERIC AND CATEGORICAL FEATURES
# ============================================================

categorical_columns = X.select_dtypes(
    include=["object", "string", "category"]
).columns.tolist()

numeric_columns = X.select_dtypes(
    include=[np.number]
).columns.tolist()


print("\nNumeric features:")

for feature in numeric_columns:
    print(" -", feature)


print("\nCategorical features:")

for feature in categorical_columns:
    print(" -", feature)


# ============================================================
# 9. ENCODE CATEGORICAL FEATURES
# ============================================================

print("\nEncoding categorical features...")

X_encoded = pd.get_dummies(
    X,
    columns=categorical_columns,
    prefix=categorical_columns,
    dtype=float
)

print(
    "\nNumber of encoded features:",
    X_encoded.shape[1]
)


# ============================================================
# 10. MUTUAL INFORMATION
# ============================================================

print("\n" + "-" * 70)
print("CALCULATING MUTUAL INFORMATION")
print("-" * 70)

mi_values = mutual_info_regression(
    X_encoded,
    y,
    random_state=42
)


mi_results = pd.DataFrame({

    "encoded_feature":
        X_encoded.columns,

    "mutual_information":
        mi_values
})


# ============================================================
# 11. MAP ENCODED FEATURES TO ORIGINAL FEATURES
# ============================================================

def get_original_feature(encoded_name):

    """
    Convert an encoded feature name back to its
    original feature name.

    Example:

    category_Technology
        ↓
    category
    """

    for feature in features:

        prefix = feature + "_"

        if encoded_name.startswith(prefix):

            return feature

    return encoded_name


mi_results["feature"] = mi_results[
    "encoded_feature"
].apply(
    get_original_feature
)


# ============================================================
# 12. SUM MUTUAL INFORMATION BY ORIGINAL FEATURE
# ============================================================

mi_summary = (

    mi_results

    .groupby("feature")[
        "mutual_information"
    ]

    .sum()

    .reset_index()
)


# ============================================================
# 13. RANDOM FOREST FEATURE IMPORTANCE
# ============================================================

print("\n" + "-" * 70)
print("CALCULATING RANDOM FOREST FEATURE IMPORTANCE")
print("-" * 70)

rf = RandomForestRegressor(

    n_estimators=200,

    random_state=42,

    n_jobs=-1
)


rf.fit(
    X_encoded,
    y
)


rf_results = pd.DataFrame({

    "encoded_feature":
        X_encoded.columns,

    "rf_importance":
        rf.feature_importances_
})


# ============================================================
# 14. MAP RANDOM FOREST FEATURES
# ============================================================

rf_results["feature"] = rf_results[
    "encoded_feature"
].apply(
    get_original_feature
)


# ============================================================
# 15. SUM RANDOM FOREST IMPORTANCE
# ============================================================

rf_summary = (

    rf_results

    .groupby("feature")[
        "rf_importance"
    ]

    .sum()

    .reset_index()
)


# ============================================================
# 16. COMBINE FEATURE SELECTION RESULTS
# ============================================================

results = pd.merge(

    mi_summary,

    rf_summary,

    on="feature",

    how="outer"
)


results = results.fillna(0)


# ============================================================
# 17. NORMALIZE MUTUAL INFORMATION
# ============================================================

mi_max = results[
    "mutual_information"
].max()


if mi_max > 0:

    results[
        "mi_normalized"
    ] = (

        results[
            "mutual_information"
        ]

        / mi_max
    )

else:

    results[
        "mi_normalized"
    ] = 0


# ============================================================
# 18. NORMALIZE RANDOM FOREST IMPORTANCE
# ============================================================

rf_max = results[
    "rf_importance"
].max()


if rf_max > 0:

    results[
        "rf_normalized"
    ] = (

        results[
            "rf_importance"
        ]

        / rf_max
    )

else:

    results[
        "rf_normalized"
    ] = 0


# ============================================================
# 19. CALCULATE COMBINED SCORE
# ============================================================

# Equal weighting:
#
# 50% Mutual Information
# 50% Random Forest Importance

results[
    "combined_score"
] = (

    results[
        "mi_normalized"
    ] * 0.5

    +

    results[
        "rf_normalized"
    ] * 0.5
)


# ============================================================
# 20. RANK FEATURES
# ============================================================

results = results.sort_values(

    "combined_score",

    ascending=False
)


results = results.reset_index(
    drop=True
)


results["rank"] = (
    results.index + 1
)


# ============================================================
# 21. REORDER RESULT COLUMNS
# ============================================================

results = results[
    [
        "rank",
        "feature",
        "mutual_information",
        "rf_importance",
        "combined_score"
    ]
]


# ============================================================
# 22. DISPLAY RESULTS
# ============================================================

print("\n")
print("=" * 70)
print("FEATURE SELECTION RESULTS")
print("=" * 70)

print(
    results.to_string(
        index=False
    )
)


# ============================================================
# 23. SELECT TOP FEATURES
# ============================================================

# We select the top 8 features as the initial
# feature set for the ML stage.

TOP_N = min(
    8,
    len(results)
)


selected_features = (
    results
    .head(TOP_N)
    ["feature"]
    .tolist()
)


print("\n")
print("=" * 70)
print("SELECTED FEATURES FOR ML")
print("=" * 70)


for index, feature in enumerate(
    selected_features,
    start=1
):

    print(
        f"{index}. {feature}"
    )


# ============================================================
# 24. SAVE COMPLETE FEATURE RANKING
# ============================================================

results_file = os.path.join(

    OUTPUT_DIR,

    "feature_selection_results.csv"
)


results.to_csv(

    results_file,

    index=False
)


print("\nComplete feature ranking saved to:")

print(
    results_file
)


# ============================================================
# 25. SAVE SELECTED FEATURES
# ============================================================

selected_file = os.path.join(

    OUTPUT_DIR,

    "selected_features.csv"
)


selected_features_df = pd.DataFrame({

    "selected_feature":
        selected_features

})


selected_features_df.to_csv(

    selected_file,

    index=False
)


print("\nSelected features saved to:")

print(
    selected_file
)


# ============================================================
# 26. CREATE FEATURE IMPORTANCE CHART
# ============================================================

plot_data = (

    results

    .head(TOP_N)

    .sort_values(
        "combined_score"
    )
)


plt.figure(
    figsize=(10, 6)
)


plt.barh(

    plot_data[
        "feature"
    ],

    plot_data[
        "combined_score"
    ]
)


plt.xlabel(
    "Combined Feature Selection Score"
)


plt.ylabel(
    "Feature"
)


plt.title(
    "Top Features for Superstore Profit Prediction"
)


plt.tight_layout()


figure_file = os.path.join(

    FIGURE_DIR,

    "21_feature_selection.png"
)


plt.savefig(

    figure_file,

    dpi=300,

    bbox_inches="tight"
)


plt.close()


print("\nFeature selection chart saved to:")

print(
    figure_file
)


# ============================================================
# 27. FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("FEATURE SELECTION COMPLETE")
print("=" * 70)

print(
    "Target variable:",
    TARGET
)

print(
    "Candidate features:",
    len(features)
)

print(
    "Selected features:",
    len(selected_features)
)


print("\nSelected features:")

for feature in selected_features:

    print(
        " -",
        feature
    )


print("\nNext stage:")
print(
    "Build the Machine Learning model using the selected features."
)


print("=" * 70)