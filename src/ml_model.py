# ============================================================
# MACHINE LEARNING MODEL
# Project: AI-Assisted Data Visualization for Business
# Decision Support Systems
#
# Dataset: Superstore Sales Dataset
# Target: Profit
# Task: Profit Prediction (Regression)
# ============================================================

import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


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

SELECTED_FEATURES_FILE = os.path.join(
    BASE_DIR,
    "outputs",
    "results",
    "selected_features.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "outputs",
    "model_metrics"
)

RESULTS_DIR = os.path.join(
    BASE_DIR,
    "outputs",
    "results"
)

FIGURE_DIR = os.path.join(
    BASE_DIR,
    "outputs",
    "figures"
)

MODELS_DIR = os.path.join(
    BASE_DIR,
    "models"
)


# Create directories if they do not exist.

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)

os.makedirs(
    FIGURE_DIR,
    exist_ok=True
)

os.makedirs(
    MODELS_DIR,
    exist_ok=True
)


# ============================================================
# 2. HEADER
# ============================================================

print("=" * 70)
print("MACHINE LEARNING - SUPERSTORE PROFIT PREDICTION")
print("=" * 70)


# ============================================================
# 3. LOAD CLEANED DATASET
# ============================================================

df = pd.read_csv(
    INPUT_FILE
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
# 5. DEFINE TARGET
# ============================================================

TARGET = "profit"

if TARGET not in df.columns:

    raise ValueError(
        f"Target column '{TARGET}' "
        "was not found in the dataset."
    )


# ============================================================
# 6. CHECK FEATURES
# ============================================================

missing_features = [

    feature

    for feature in selected_features

    if feature not in df.columns

]


if missing_features:

    raise ValueError(
        "The following selected features "
        "are missing from the dataset: "
        + str(missing_features)
    )


# ============================================================
# 7. CREATE X AND y
# ============================================================

X = df[
    selected_features
].copy()

y = df[
    TARGET
].copy()


print("\nFeature matrix shape:")
print(
    X.shape
)

print("\nTarget shape:")
print(
    y.shape
)


# ============================================================
# 8. HANDLE MISSING TARGET VALUES
# ============================================================

valid_target = y.notna()

X = X.loc[
    valid_target
].copy()

y = y.loc[
    valid_target
].copy()


# ============================================================
# 9. IDENTIFY FEATURE TYPES
# ============================================================

numeric_features = X.select_dtypes(
    include=[np.number]
).columns.tolist()


categorical_features = X.select_dtypes(
    include=[
        "object",
        "string",
        "category"
    ]
).columns.tolist()


print("\nNumeric features:")

for feature in numeric_features:

    print(
        " -",
        feature
    )


print("\nCategorical features:")

for feature in categorical_features:

    print(
        " -",
        feature
    )


# ============================================================
# 10. PREPROCESSING PIPELINE
# ============================================================

# Numeric features:
# Missing values → median

numeric_transformer = Pipeline(
    steps=[

        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        )

    ]
)


# Categorical features:
# Missing values → most frequent
# Then one-hot encoding

categorical_transformer = Pipeline(
    steps=[

        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            )
        ),

        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )

    ]
)


# Combine numeric and categorical preprocessing.

preprocessor = ColumnTransformer(

    transformers=[

        (
            "numeric",
            numeric_transformer,
            numeric_features
        ),

        (
            "categorical",
            categorical_transformer,
            categorical_features
        )

    ]
)


# ============================================================
# 11. TRAIN / TEST SPLIT
# ============================================================

print("\n" + "-" * 70)
print("TRAIN / TEST SPLIT")
print("-" * 70)

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=42
)


print(
    "Training samples:",
    len(X_train)
)

print(
    "Testing samples:",
    len(X_test)
)


# ============================================================
# 12. DEFINE MODELS
# ============================================================

linear_regression = LinearRegression()


random_forest = RandomForestRegressor(

    n_estimators=200,

    random_state=42,

    n_jobs=-1

)


# ============================================================
# 13. CREATE MODEL PIPELINES
# ============================================================

linear_model = Pipeline(

    steps=[

        (
            "preprocessor",
            preprocessor
        ),

        (
            "model",
            linear_regression
        )

    ]
)


random_forest_model = Pipeline(

    steps=[

        (
            "preprocessor",
            preprocessor
        ),

        (
            "model",
            random_forest
        )

    ]
)


# ============================================================
# 14. STORE MODELS
# ============================================================

models = {

    "Linear Regression":
        linear_model,

    "Random Forest":
        random_forest_model

}


# ============================================================
# 15. TRAIN AND EVALUATE MODELS
# ============================================================

model_results = []

predictions = {}


for model_name, model in models.items():

    print("\n" + "=" * 70)

    print(
        "TRAINING:",
        model_name
    )

    print("=" * 70)


    # Train model

    model.fit(
        X_train,
        y_train
    )


    # Predict test data

    y_pred = model.predict(
        X_test
    )


    predictions[
        model_name
    ] = y_pred


    # --------------------------------------------------------
    # Evaluation metrics
    # --------------------------------------------------------

    mae = mean_absolute_error(
        y_test,
        y_pred
    )


    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            y_pred
        )
    )


    r2 = r2_score(
        y_test,
        y_pred
    )


    model_results.append({

        "Model":
            model_name,

        "MAE":
            mae,

        "RMSE":
            rmse,

        "R2":
            r2

    })


    # Display metrics

    print(
        "\nMean Absolute Error (MAE):",
        round(mae, 4)
    )

    print(
        "Root Mean Squared Error (RMSE):",
        round(rmse, 4)
    )

    print(
        "R² Score:",
        round(r2, 4)
    )


# ============================================================
# 16. CREATE MODEL COMPARISON DATAFRAME
# ============================================================

metrics_df = pd.DataFrame(
    model_results
)


print("\n")
print("=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

print(
    metrics_df.to_string(
        index=False
    )
)


# ============================================================
# 17. SAVE MODEL METRICS
# ============================================================

metrics_file = os.path.join(

    OUTPUT_DIR,

    "model_comparison.csv"

)


metrics_df.to_csv(

    metrics_file,

    index=False

)


print("\nModel comparison saved to:")

print(
    metrics_file
)


# ============================================================
# 18. IDENTIFY BEST MODEL
# ============================================================

# R²:
# Higher is better.

# RMSE:
# Lower is better.

# We select the model with the highest R².

best_model_name = (
    metrics_df
    .sort_values(
        "R2",
        ascending=False
    )
    .iloc[0]["Model"]
)


best_model = models[
    best_model_name
]


print("\n")
print("=" * 70)
print("BEST MODEL")
print("=" * 70)

print(
    "Selected model:",
    best_model_name
)


# ============================================================
# 19. SAVE BEST MODEL
# ============================================================

model_file = os.path.join(

    MODELS_DIR,

    "best_profit_prediction_model.pkl"

)


with open(
    model_file,
    "wb"
) as file:

    pickle.dump(
        best_model,
        file
    )


print("\nBest model saved to:")

print(
    model_file
)


# ============================================================
# 20. SAVE TEST PREDICTIONS
# ============================================================

best_predictions = predictions[
    best_model_name
]


prediction_df = pd.DataFrame({

    "actual_profit":
        y_test.values,

    "predicted_profit":
        best_predictions

})


prediction_file = os.path.join(

    RESULTS_DIR,

    "profit_predictions.csv"

)


prediction_df.to_csv(

    prediction_file,

    index=False

)


print("\nPredictions saved to:")

print(
    prediction_file
)


# ============================================================
# 21. ACTUAL VS PREDICTED PLOT
# ============================================================

plt.figure(
    figsize=(8, 6)
)


plt.scatter(

    y_test,

    best_predictions,

    alpha=0.5

)


# Reference line

minimum = min(
    y_test.min(),
    best_predictions.min()
)

maximum = max(
    y_test.max(),
    best_predictions.max()
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
    f"Actual vs Predicted Profit - {best_model_name}"
)


plt.tight_layout()


prediction_plot = os.path.join(

    FIGURE_DIR,

    "22_actual_vs_predicted_profit.png"

)


plt.savefig(

    prediction_plot,

    dpi=300,

    bbox_inches="tight"

)


plt.close()


print("\nActual vs predicted plot saved to:")

print(
    prediction_plot
)


# ============================================================
# 22. MODEL COMPARISON CHART
# ============================================================

plt.figure(
    figsize=(9, 6)
)


plt.bar(

    metrics_df["Model"],

    metrics_df["R2"]

)


plt.xlabel(
    "Model"
)


plt.ylabel(
    "R² Score"
)


plt.title(
    "Model Performance Comparison"
)


plt.xticks(
    rotation=15
)


plt.tight_layout()


comparison_plot = os.path.join(

    FIGURE_DIR,

    "23_model_comparison.png"

)


plt.savefig(

    comparison_plot,

    dpi=300,

    bbox_inches="tight"

)


plt.close()


print("\nModel comparison chart saved to:")

print(
    comparison_plot
)


# ============================================================
# 23. FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("MACHINE LEARNING COMPLETE")
print("=" * 70)

print(
    "Target:",
    TARGET
)

print(
    "Training samples:",
    len(X_train)
)

print(
    "Testing samples:",
    len(X_test)
)

print(
    "Models trained:",
    len(models)
)

print(
    "Best model:",
    best_model_name
)


print("\nEvaluation metrics:")

print(
    metrics_df.to_string(
        index=False
    )
)


print("\nGenerated files:")

print(
    " - model_comparison.csv"
)

print(
    " - profit_predictions.csv"
)

print(
    " - best_profit_prediction_model.pkl"
)

print(
    " - 22_actual_vs_predicted_profit.png"
)

print(
    " - 23_model_comparison.png"
)


print("\nNext stage:")
print(
    "Explainable AI (XAI) using the trained ML model."
)


print("=" * 70)