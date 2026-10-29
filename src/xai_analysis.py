# ============================================================
# EXPLAINABLE AI (XAI) ANALYSIS
# Project: AI-Assisted Data Visualization for Business
# Decision Support Systems
#
# Dataset: Superstore Sales Dataset
# Model: Random Forest Regression
# Target: Profit
# ============================================================

import os
import pickle

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.inspection import permutation_importance


# ============================================================
# 1. PROJECT PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATA_FILE = os.path.join(
    BASE_DIR,
    "outputs",
    "cleaned",
    "superstore_cleaned.csv"
)

MODEL_FILE = os.path.join(
    BASE_DIR,
    "models",
    "best_profit_prediction_model.pkl"
)

SELECTED_FEATURES_FILE = os.path.join(
    BASE_DIR,
    "outputs",
    "results",
    "selected_features.csv"
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


os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

os.makedirs(
    FIGURE_DIR,
    exist_ok=True
)


# ============================================================
# 2. HEADER
# ============================================================

print("=" * 70)
print("EXPLAINABLE AI (XAI) - SUPERSTORE PROFIT MODEL")
print("=" * 70)


# ============================================================
# 3. LOAD DATASET
# ============================================================

df = pd.read_csv(
    DATA_FILE
)

print("\nDataset loaded successfully.")

print(
    "Rows:",
    df.shape[0]
)

print(
    "Columns:",
    df.shape[1]
)


# ============================================================
# 4. LOAD SELECTED FEATURES
# ============================================================

selected_df = pd.read_csv(
    SELECTED_FEATURES_FILE
)

selected_features = (
    selected_df[
        "selected_feature"
    ]
    .tolist()
)

print("\nSelected features:")

for feature in selected_features:

    print(
        " -",
        feature
    )


# ============================================================
# 5. LOAD TRAINED MODEL
# ============================================================

print("\nLoading trained model...")

with open(
    MODEL_FILE,
    "rb"
) as file:

    model = pickle.load(
        file
    )


print(
    "Model loaded successfully."
)


# ============================================================
# 6. PREPARE DATA
# ============================================================

TARGET = "profit"

X = df[
    selected_features
].copy()

y = df[
    TARGET
].copy()


# Remove rows where target is missing.

valid_rows = y.notna()

X = X.loc[
    valid_rows
].copy()

y = y.loc[
    valid_rows
].copy()


# ============================================================
# 7. GENERATE PREDICTIONS
# ============================================================

print("\nGenerating predictions...")

y_pred = model.predict(
    X
)

print(
    "Predictions generated successfully."
)


# ============================================================
# 8. PERMUTATION IMPORTANCE
# ============================================================

print("\n" + "-" * 70)
print("CALCULATING PERMUTATION IMPORTANCE")
print("-" * 70)

print(
    "This measures how much model performance changes when"
)

print(
    "each feature is randomly shuffled."
)


permutation = permutation_importance(

    model,

    X,

    y,

    n_repeats=10,

    random_state=42,

    scoring="r2",

    n_jobs=-1
)


# ============================================================
# 9. CREATE FEATURE IMPORTANCE TABLE
# ============================================================

importance_df = pd.DataFrame({

    "feature":
        selected_features,

    "importance_mean":
        permutation.importances_mean,

    "importance_std":
        permutation.importances_std

})


importance_df = (
    importance_df
    .sort_values(
        "importance_mean",
        ascending=False
    )
    .reset_index(
        drop=True
    )
)


importance_df[
    "rank"
] = (
    importance_df.index + 1
)


importance_df = importance_df[
    [
        "rank",
        "feature",
        "importance_mean",
        "importance_std"
    ]
]


# ============================================================
# 10. DISPLAY FEATURE IMPORTANCE
# ============================================================

print("\n")
print("=" * 70)
print("PERMUTATION FEATURE IMPORTANCE")
print("=" * 70)

print(
    importance_df.to_string(
        index=False
    )
)


# ============================================================
# 11. SAVE FEATURE IMPORTANCE
# ============================================================

importance_file = os.path.join(

    OUTPUT_DIR,

    "xai_feature_importance.csv"

)


importance_df.to_csv(

    importance_file,

    index=False
)


print("\nFeature importance saved to:")

print(
    importance_file
)


# ============================================================
# 12. CREATE FEATURE IMPORTANCE CHART
# ============================================================

plot_data = (
    importance_df
    .sort_values(
        "importance_mean"
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
        "importance_mean"
    ]
)


plt.xlabel(
    "Permutation Importance (R² decrease)"
)


plt.ylabel(
    "Feature"
)


plt.title(
    "Explainable AI - Feature Importance"
)


plt.tight_layout()


importance_plot = os.path.join(

    FIGURE_DIR,

    "24_xai_feature_importance.png"

)


plt.savefig(

    importance_plot,

    dpi=300,

    bbox_inches="tight"

)


plt.close()


print("\nFeature importance chart saved to:")

print(
    importance_plot
)


# ============================================================
# 13. CREATE ACTUAL VS PREDICTED DATA
# ============================================================

explanation_df = X.copy()

explanation_df[
    "actual_profit"
] = y.values

explanation_df[
    "predicted_profit"
] = y_pred

explanation_df[
    "prediction_error"
] = (

    explanation_df[
        "actual_profit"
    ]

    -

    explanation_df[
        "predicted_profit"
    ]
)


# ============================================================
# 14. IDENTIFY HIGH-PROFIT AND LOW-PROFIT CASES
# ============================================================

high_profit_case = (
    explanation_df
    .loc[
        explanation_df[
            "predicted_profit"
        ].idxmax()
    ]
)


low_profit_case = (
    explanation_df
    .loc[
        explanation_df[
            "predicted_profit"
        ].idxmin()
    ]
)


# ============================================================
# 15. DISPLAY EXAMPLE PREDICTIONS
# ============================================================

print("\n")
print("=" * 70)
print("EXAMPLE HIGH-PROFIT PREDICTION")
print("=" * 70)

print(
    "Predicted profit:",
    round(
        high_profit_case[
            "predicted_profit"
        ],
        2
    )
)


print("\nFeature values:")

for feature in selected_features:

    print(
        f" - {feature}: "
        f"{high_profit_case[feature]}"
    )


print("\n")
print("=" * 70)
print("EXAMPLE LOW-PROFIT PREDICTION")
print("=" * 70)

print(
    "Predicted profit:",
    round(
        low_profit_case[
            "predicted_profit"
        ],
        2
    )
)


print("\nFeature values:")

for feature in selected_features:

    print(
        f" - {feature}: "
        f"{low_profit_case[feature]}"
    )


# ============================================================
# 16. SAVE PREDICTION EXPLANATION DATA
# ============================================================

explanation_file = os.path.join(

    OUTPUT_DIR,

    "xai_prediction_explanations.csv"

)


explanation_df.to_csv(

    explanation_file,

    index=False
)


print("\nPrediction explanation data saved to:")

print(
    explanation_file
)


# ============================================================
# 17. PREDICTED PROFIT DISTRIBUTION
# ============================================================

plt.figure(
    figsize=(9, 6)
)


plt.hist(

    y_pred,

    bins=40
)


plt.xlabel(
    "Predicted Profit"
)


plt.ylabel(
    "Number of Transactions"
)


plt.title(
    "Distribution of Predicted Profit"
)


plt.tight_layout()


prediction_distribution_plot = os.path.join(

    FIGURE_DIR,

    "25_predicted_profit_distribution.png"

)


plt.savefig(

    prediction_distribution_plot,

    dpi=300,

    bbox_inches="tight"

)


plt.close()


print(
    "\nPredicted profit distribution saved to:"
)

print(
    prediction_distribution_plot
)


# ============================================================
# 18. ACTUAL VS PREDICTED PROFIT
# ============================================================

plt.figure(
    figsize=(9, 6)
)


plt.scatter(

    y,

    y_pred,

    alpha=0.35

)


minimum = min(
    y.min(),
    y_pred.min()
)

maximum = max(
    y.max(),
    y_pred.max()
)


plt.plot(

    [minimum, maximum],

    [minimum, maximum]

)


plt.xlabel(
    "Actual Profit"
)


plt.ylabel(
    "Predicted Profit"
)


plt.title(
    "XAI Model Prediction Analysis"
)


plt.tight_layout()


prediction_plot = os.path.join(

    FIGURE_DIR,

    "26_xai_actual_vs_predicted.png"

)


plt.savefig(

    prediction_plot,

    dpi=300,

    bbox_inches="tight"

)


plt.close()


print(
    "\nXAI actual vs predicted plot saved to:"
)

print(
    prediction_plot
)


# ============================================================
# 19. FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("EXPLAINABLE AI ANALYSIS COMPLETE")
print("=" * 70)

print(
    "Model:",
    "Random Forest Regression"
)

print(
    "Target:",
    TARGET
)

print(
    "Features explained:",
    len(selected_features)
)

print(
    "Transactions analyzed:",
    len(X)
)


print("\nGenerated XAI files:")

print(
    " - xai_feature_importance.csv"
)

print(
    " - xai_prediction_explanations.csv"
)

print(
    " - 24_xai_feature_importance.png"
)

print(
    " - 25_predicted_profit_distribution.png"
)

print(
    " - 26_xai_actual_vs_predicted.png"
)


print("\nNext stage:")
print(
    "Generate AI-assisted business recommendations."
)


print("=" * 70)