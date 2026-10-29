import os
import itertools
import warnings
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_STATE = 42
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASETS = {
    "D1": os.path.join(
        BASE_DIR,
        "outputs",
        "cleaned",
        "superstore_cleaned.csv"
    ),
    "D2": os.path.join(
        BASE_DIR,
        "outputs",
        "cleaned",
        "global_superstore_cleaned.csv"
    )
}

OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "results")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# METRICS
# ============================================================

def smape(y_true, y_pred):
    denominator = np.abs(y_true) + np.abs(y_pred)
    numerator = 2 * np.abs(y_pred - y_true)

    mask = denominator != 0

    if not np.any(mask):
        return 0.0

    return np.mean(
        numerator[mask] / denominator[mask]
    ) * 100


def safe_mape(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    mask = np.abs(y_true) > 1e-8

    if not np.any(mask):
        return 0.0

    return np.mean(
        np.abs(
            (y_true[mask] - y_pred[mask])
            / y_true[mask]
        )
    ) * 100


def calculate_metrics(y_true, y_pred):

    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(
            mean_squared_error(y_true, y_pred)
        ),
        "MAPE": safe_mape(y_true, y_pred),
        "sMAPE": smape(y_true, y_pred),
        "R2": r2_score(y_true, y_pred)
    }


# ============================================================
# DATA PREPARATION
# ============================================================

def prepare_data(dataset_name, filepath):

    df = pd.read_csv(filepath)

    df.columns = [
        str(c).strip().lower().replace(" ", "_")
        for c in df.columns
    ]

    target = "profit"

    if target not in df.columns:
        raise ValueError(
            f"{dataset_name}: target column 'profit' not found."
        )

    # --------------------------------------------------------
    # Dataset 1
    # --------------------------------------------------------

    if dataset_name == "D1":

        numeric_features = [
            "sales",
            "quantity",
            "discount",
            "shipping_days",
            "order_year",
            "order_month"
        ]

        categorical_features = [
            "category",
            "subcategory",
            "region",
            "stateprovince",
            "segment",
            "ship_mode"
        ]

    # --------------------------------------------------------
    # Dataset 2
    # --------------------------------------------------------

    else:

        numeric_features = [
            "sales",
            "quantity",
            "discount",
            "shipping_days",
            "order_month"
        ]

        categorical_features = [
            "market",
            "region",
            "category",
            "subcategory",
            "state"
        ]

    numeric_features = [
        c for c in numeric_features
        if c in df.columns
    ]

    categorical_features = [
        c for c in categorical_features
        if c in df.columns
    ]

    features = numeric_features + categorical_features

    X = df[features].copy()
    y = df[target].copy()

    # --------------------------------------------------------
    # Remove rows with missing target
    # --------------------------------------------------------

    valid = y.notna()

    X = X.loc[valid].reset_index(drop=True)
    y = y.loc[valid].reset_index(drop=True)

    # --------------------------------------------------------
    # 80 / 10 / 10 split
    # --------------------------------------------------------

    n = len(X)

    train_end = int(n * 0.80)
    validation_end = int(n * 0.90)

    X_train = X.iloc[:train_end].copy()
    y_train = y.iloc[:train_end].copy()

    X_val = X.iloc[train_end:validation_end].copy()
    y_val = y.iloc[train_end:validation_end].copy()

    X_test = X.iloc[validation_end:].copy()
    y_test = y.iloc[validation_end:].copy()

    return (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        numeric_features,
        categorical_features
    )


# ============================================================
# PREPROCESSOR
# ============================================================

def create_preprocessor(
    numeric_features,
    categorical_features
):

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median")
            )
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                )
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                )
            )
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                numeric_pipeline,
                numeric_features
            ),
            (
                "cat",
                categorical_pipeline,
                categorical_features
            )
        ],
        remainder="drop"
    )

    return preprocessor


# ============================================================
# MODEL BUILDERS
# ============================================================

def build_models(
    numeric_features,
    categorical_features
):

    def pipeline(model):

        return Pipeline(
            steps=[
                (
                    "preprocessor",
                    create_preprocessor(
                        numeric_features,
                        categorical_features
                    )
                ),
                (
                    "model",
                    model
                )
            ]
        )

    models = {

        "Gradient Boosting": pipeline(
            GradientBoostingRegressor(
                n_estimators=250,
                learning_rate=0.05,
                max_depth=3,
                random_state=RANDOM_STATE
            )
        ),

        "Random Forest": pipeline(
            RandomForestRegressor(
                n_estimators=250,
                max_depth=None,
                min_samples_leaf=1,
                random_state=RANDOM_STATE,
                n_jobs=-1
            )
        ),

        "Ridge Regression": pipeline(
            Ridge(alpha=10.0)
        )
    }

    return models


# ============================================================
# RESIDUAL HYBRID
# ============================================================

def residual_hybrid(
    base_model,
    residual_model,
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    correction_strength
):

    # --------------------------------------------------------
    # Step 1:
    # Train base model on original target
    # --------------------------------------------------------

    base_model.fit(X_train, y_train)

    train_base_pred = base_model.predict(X_train)
    val_base_pred = base_model.predict(X_val)
    test_base_pred = base_model.predict(X_test)

    # --------------------------------------------------------
    # Step 2:
    # Calculate training residuals
    # --------------------------------------------------------

    train_residuals = (
        y_train.values - train_base_pred
    )

    # --------------------------------------------------------
    # Step 3:
    # Train second model specifically
    # on the errors of the first model
    # --------------------------------------------------------

    residual_model.fit(
        X_train,
        train_residuals
    )

    val_correction = residual_model.predict(X_val)
    test_correction = residual_model.predict(X_test)

    # --------------------------------------------------------
    # Step 4:
    # Correct the base predictions
    # --------------------------------------------------------

    val_prediction = (
        val_base_pred
        + correction_strength * val_correction
    )

    test_prediction = (
        test_base_pred
        + correction_strength * test_correction
    )

    return (
        val_prediction,
        test_prediction
    )


# ============================================================
# RUN ONE DATASET
# ============================================================

def run_dataset(dataset_name, filepath):

    print("\n" + "=" * 80)
    print(f"FINAL HYBRID EXPERIMENT — {dataset_name}")
    print("=" * 80)

    (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        numeric_features,
        categorical_features
    ) = prepare_data(
        dataset_name,
        filepath
    )

    print(f"Train rows      : {len(X_train)}")
    print(f"Validation rows : {len(X_val)}")
    print(f"Test rows       : {len(X_test)}")

    models = build_models(
        numeric_features,
        categorical_features
    )

    # ========================================================
    # BASELINE MODELS
    # ========================================================

    fitted_models = {}

    validation_predictions = {}
    test_predictions = {}

    results = []

    for name, model in models.items():

        print(f"\nTraining {name}...")

        model.fit(
            X_train,
            y_train
        )

        val_pred = model.predict(X_val)
        test_pred = model.predict(X_test)

        fitted_models[name] = model

        validation_predictions[name] = val_pred
        test_predictions[name] = test_pred

        val_metrics = calculate_metrics(
            y_val,
            val_pred
        )

        test_metrics = calculate_metrics(
            y_test,
            test_pred
        )

        results.append({
            "Dataset": dataset_name,
            "Model": name,
            "Family": "Baseline",
            "Validation_MAE": val_metrics["MAE"],
            "Validation_RMSE": val_metrics["RMSE"],
            "Validation_sMAPE": val_metrics["sMAPE"],
            "Validation_R2": val_metrics["R2"],
            "MAE": test_metrics["MAE"],
            "RMSE": test_metrics["RMSE"],
            "MAPE": test_metrics["MAPE"],
            "sMAPE": test_metrics["sMAPE"],
            "R2": test_metrics["R2"]
        })

    # ========================================================
    # CANDIDATE 1
    # GB + RF RESIDUAL CORRECTION
    # ========================================================

    print("\nTesting Hybrid 1: GB + RF residual correction")

    gb = build_models(
        numeric_features,
        categorical_features
    )["Gradient Boosting"]

    rf = build_models(
        numeric_features,
        categorical_features
    )["Random Forest"]

    best_h1 = None

    for strength in np.arange(
        0.0,
        1.01,
        0.10
    ):

        val_pred, test_pred = residual_hybrid(
            gb,
            rf,
            X_train,
            y_train,
            X_val,
            y_val,
            X_test,
            strength
        )

        metrics = calculate_metrics(
            y_val,
            val_pred
        )

        if (
            best_h1 is None
            or metrics["RMSE"] < best_h1["Validation_RMSE"]
        ):

            best_h1 = {
                "strength": strength,
                "Validation_RMSE": metrics["RMSE"],
                "val_pred": val_pred,
                "test_pred": test_pred
            }

    test_metrics = calculate_metrics(
        y_test,
        best_h1["test_pred"]
    )

    results.append({
        "Dataset": dataset_name,
        "Model": "Hybrid 1 - GB + RF Residual",
        "Family": "Proposed Hybrid Candidate",
        "Validation_MAE": calculate_metrics(
            y_val,
            best_h1["val_pred"]
        )["MAE"],
        "Validation_RMSE": best_h1["Validation_RMSE"],
        "Validation_sMAPE": calculate_metrics(
            y_val,
            best_h1["val_pred"]
        )["sMAPE"],
        "Validation_R2": calculate_metrics(
            y_val,
            best_h1["val_pred"]
        )["R2"],
        "MAE": test_metrics["MAE"],
        "RMSE": test_metrics["RMSE"],
        "MAPE": test_metrics["MAPE"],
        "sMAPE": test_metrics["sMAPE"],
        "R2": test_metrics["R2"],
        "Parameter": best_h1["strength"]
    })

    # ========================================================
    # CANDIDATE 2
    # RF + GB RESIDUAL CORRECTION
    # ========================================================

    print("\nTesting Hybrid 2: RF + GB residual correction")

    rf = build_models(
        numeric_features,
        categorical_features
    )["Random Forest"]

    gb = build_models(
        numeric_features,
        categorical_features
    )["Gradient Boosting"]

    best_h2 = None

    for strength in np.arange(
        0.0,
        1.01,
        0.10
    ):

        val_pred, test_pred = residual_hybrid(
            rf,
            gb,
            X_train,
            y_train,
            X_val,
            y_val,
            X_test,
            strength
        )

        metrics = calculate_metrics(
            y_val,
            val_pred
        )

        if (
            best_h2 is None
            or metrics["RMSE"] < best_h2["Validation_RMSE"]
        ):

            best_h2 = {
                "strength": strength,
                "Validation_RMSE": metrics["RMSE"],
                "val_pred": val_pred,
                "test_pred": test_pred
            }

    test_metrics = calculate_metrics(
        y_test,
        best_h2["test_pred"]
    )

    results.append({
        "Dataset": dataset_name,
        "Model": "Hybrid 2 - RF + GB Residual",
        "Family": "Proposed Hybrid Candidate",
        "Validation_MAE": calculate_metrics(
            y_val,
            best_h2["val_pred"]
        )["MAE"],
        "Validation_RMSE": best_h2["Validation_RMSE"],
        "Validation_sMAPE": calculate_metrics(
            y_val,
            best_h2["val_pred"]
        )["sMAPE"],
        "Validation_R2": calculate_metrics(
            y_val,
            best_h2["val_pred"]
        )["R2"],
        "MAE": test_metrics["MAE"],
        "RMSE": test_metrics["RMSE"],
        "MAPE": test_metrics["MAPE"],
        "sMAPE": test_metrics["sMAPE"],
        "R2": test_metrics["R2"],
        "Parameter": best_h2["strength"]
    })

    # ========================================================
    # CANDIDATE 3
    # GB + RIDGE RESIDUAL CORRECTION
    # ========================================================

    print("\nTesting Hybrid 3: GB + Ridge residual correction")

    gb = build_models(
        numeric_features,
        categorical_features
    )["Gradient Boosting"]

    ridge = build_models(
        numeric_features,
        categorical_features
    )["Ridge Regression"]

    best_h3 = None

    for strength in np.arange(
        0.0,
        1.01,
        0.10
    ):

        val_pred, test_pred = residual_hybrid(
            gb,
            ridge,
            X_train,
            y_train,
            X_val,
            y_val,
            X_test,
            strength
        )

        metrics = calculate_metrics(
            y_val,
            val_pred
        )

        if (
            best_h3 is None
            or metrics["RMSE"] < best_h3["Validation_RMSE"]
        ):

            best_h3 = {
                "strength": strength,
                "Validation_RMSE": metrics["RMSE"],
                "val_pred": val_pred,
                "test_pred": test_pred
            }

    test_metrics = calculate_metrics(
        y_test,
        best_h3["test_pred"]
    )

    results.append({
        "Dataset": dataset_name,
        "Model": "Hybrid 3 - GB + Ridge Residual",
        "Family": "Proposed Hybrid Candidate",
        "Validation_MAE": calculate_metrics(
            y_val,
            best_h3["val_pred"]
        )["MAE"],
        "Validation_RMSE": best_h3["Validation_RMSE"],
        "Validation_sMAPE": calculate_metrics(
            y_val,
            best_h3["val_pred"]
        )["sMAPE"],
        "Validation_R2": calculate_metrics(
            y_val,
            best_h3["val_pred"]
        )["R2"],
        "MAE": test_metrics["MAE"],
        "RMSE": test_metrics["RMSE"],
        "MAPE": test_metrics["MAPE"],
        "sMAPE": test_metrics["sMAPE"],
        "R2": test_metrics["R2"],
        "Parameter": best_h3["strength"]
    })

    # ========================================================
    # CANDIDATE 4
    # CONSTRAINED THREE-MODEL ENSEMBLE
    # ========================================================

    print(
        "\nTesting Hybrid 4: constrained "
        "GB + RF + Ridge ensemble"
    )

    gb = build_models(
        numeric_features,
        categorical_features
    )["Gradient Boosting"]

    rf = build_models(
        numeric_features,
        categorical_features
    )["Random Forest"]

    ridge = build_models(
        numeric_features,
        categorical_features
    )["Ridge Regression"]

    gb.fit(X_train, y_train)
    rf.fit(X_train, y_train)
    ridge.fit(X_train, y_train)

    gb_val = gb.predict(X_val)
    rf_val = rf.predict(X_val)
    ridge_val = ridge.predict(X_val)

    gb_test = gb.predict(X_test)
    rf_test = rf.predict(X_test)
    ridge_test = ridge.predict(X_test)

    best_h4 = None

    # Every model must contribute at least 10%.
    weights = np.arange(
        0.10,
        0.81,
        0.10
    )

    for w_gb in weights:

        for w_rf in weights:

            w_ridge = round(
                1.0 - w_gb - w_rf,
                10
            )

            if w_ridge < 0.10:
                continue

            if w_ridge > 0.80:
                continue

            val_pred = (
                w_gb * gb_val
                + w_rf * rf_val
                + w_ridge * ridge_val
            )

            metrics = calculate_metrics(
                y_val,
                val_pred
            )

            if (
                best_h4 is None
                or metrics["RMSE"]
                < best_h4["Validation_RMSE"]
            ):

                best_h4 = {
                    "w_gb": w_gb,
                    "w_rf": w_rf,
                    "w_ridge": w_ridge,
                    "Validation_RMSE": metrics["RMSE"],
                    "val_pred": val_pred
                }

    test_pred = (
        best_h4["w_gb"] * gb_test
        + best_h4["w_rf"] * rf_test
        + best_h4["w_ridge"] * ridge_test
    )

    test_metrics = calculate_metrics(
        y_test,
        test_pred
    )

    val_metrics = calculate_metrics(
        y_val,
        best_h4["val_pred"]
    )

    results.append({
        "Dataset": dataset_name,
        "Model": "Hybrid 4 - Constrained GB+RF+Ridge",
        "Family": "Proposed Hybrid Candidate",
        "Validation_MAE": val_metrics["MAE"],
        "Validation_RMSE": val_metrics["RMSE"],
        "Validation_sMAPE": val_metrics["sMAPE"],
        "Validation_R2": val_metrics["R2"],
        "MAE": test_metrics["MAE"],
        "RMSE": test_metrics["RMSE"],
        "MAPE": test_metrics["MAPE"],
        "sMAPE": test_metrics["sMAPE"],
        "R2": test_metrics["R2"],
        "Parameter": (
            f"GB={best_h4['w_gb']:.1f}, "
            f"RF={best_h4['w_rf']:.1f}, "
            f"Ridge={best_h4['w_ridge']:.1f}"
        )
    })

    # ========================================================
    # FINAL COMPARISON
    # ========================================================

    results_df = pd.DataFrame(results)

    # --------------------------------------------------------
    # Rank models
    # Lower is better for MAE/RMSE/MAPE/sMAPE
    # Higher is better for R2
    # --------------------------------------------------------

    results_df["Rank_MAE"] = (
        results_df["MAE"]
        .rank(method="min", ascending=True)
    )

    results_df["Rank_RMSE"] = (
        results_df["RMSE"]
        .rank(method="min", ascending=True)
    )

    results_df["Rank_MAPE"] = (
        results_df["MAPE"]
        .rank(method="min", ascending=True)
    )

    results_df["Rank_sMAPE"] = (
        results_df["sMAPE"]
        .rank(method="min", ascending=True)
    )

    results_df["Rank_R2"] = (
        results_df["R2"]
        .rank(method="min", ascending=False)
    )

    results_df["Average_Rank"] = results_df[
        [
            "Rank_MAE",
            "Rank_RMSE",
            "Rank_MAPE",
            "Rank_sMAPE",
            "Rank_R2"
        ]
    ].mean(axis=1)

    results_df = results_df.sort_values(
        "Average_Rank"
    ).reset_index(drop=True)

    results_df["Overall_Rank"] = (
        np.arange(len(results_df)) + 1
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output_file = os.path.join(
        OUTPUT_DIR,
        f"{dataset_name.lower()}_final_hybrid_results.csv"
    )

    results_df.to_csv(
        output_file,
        index=False
    )

    print("\n" + "-" * 80)
    print(f"FINAL {dataset_name} RESULTS")
    print("-" * 80)

    display_columns = [
        "Model",
        "Family",
        "MAE",
        "RMSE",
        "MAPE",
        "sMAPE",
        "R2",
        "Average_Rank",
        "Overall_Rank"
    ]

    print(
        results_df[
            display_columns
        ].to_string(index=False)
    )

    print(
        f"\nSaved: {output_file}"
    )

    return results_df


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    all_results = []

    for dataset_name, filepath in DATASETS.items():

        if not os.path.exists(filepath):

            print(
                f"\nWARNING: {dataset_name} file not found:"
                f"\n{filepath}"
            )

            continue

        result = run_dataset(
            dataset_name,
            filepath
        )

        all_results.append(result)

    if all_results:

        combined = pd.concat(
            all_results,
            ignore_index=True
        )

        combined_file = os.path.join(
            OUTPUT_DIR,
            "final_hybrid_experiment_all_datasets.csv"
        )

        combined.to_csv(
            combined_file,
            index=False
        )

        print("\n" + "=" * 80)
        print("ALL DATASETS — FINAL HYBRID EXPERIMENT")
        print("=" * 80)

        print(
            combined[
                [
                    "Dataset",
                    "Model",
                    "MAE",
                    "RMSE",
                    "MAPE",
                    "sMAPE",
                    "R2",
                    "Average_Rank",
                    "Overall_Rank"
                ]
            ].to_string(index=False)
        )

        print(
            f"\nSaved combined results: {combined_file}"
        )