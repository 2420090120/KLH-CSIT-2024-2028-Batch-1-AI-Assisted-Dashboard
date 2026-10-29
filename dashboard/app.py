
import io
import pickle
import zipfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")


# ============================================================
# 1. PROJECT PATHS
# ============================================================

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent

DATASETS_DIR = PROJECT_ROOT / "datasets"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
CLEANED_DIR = OUTPUTS_DIR / "cleaned"
RESULTS_DIR = OUTPUTS_DIR / "results"
MODELS_DIR = PROJECT_ROOT / "models"


DATASET_CONFIG = {
    "Dataset 1 — Superstore": {
        "description": (
            "Retail transaction data containing sales, profit, discount, "
            "quantity, products, customers, regions and dates."
        ),
        "files": [
            CLEANED_DIR / "superstore_cleaned.csv",
            DATASETS_DIR / "Dataset 1 sample_-_superstore.xls",
        ],
        "type": "transaction",
    },
    "Dataset 2 — Global Superstore": {
        "description": (
            "Multinational retail transaction data containing markets, "
            "categories, sales, profit, shipping and regional information."
        ),
        "files": [
            CLEANED_DIR / "global_superstore_cleaned.csv",
            DATASETS_DIR / "Dataset 2 Global_Superstore2.csv.zip",
        ],
        "type": "transaction",
    },
    "Dataset 3 — Olist Customers": {
        "description": (
            "Brazilian e-commerce customer data containing customer IDs, "
            "cities, states and ZIP-code prefixes."
        ),
        "files": [
            CLEANED_DIR / "olist_customers_cleaned.csv",
            DATASETS_DIR / "Dataset 3 olist_customers_dataset.csv.zip",
        ],
        "type": "customer",
    },
}


XAI_FILE = RESULTS_DIR / "xai_feature_importance.csv"
RECOMMENDATION_FILE = RESULTS_DIR / "business_recommendations.csv"
MODEL_FILE = MODELS_DIR / "best_profit_prediction_model.pkl"


# ============================================================
# 2. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI-Assisted Business Decision Support",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# 3. CSS
# ============================================================

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.6rem;
        padding-bottom: 2rem;
    }

    .hero {
        padding: 24px;
        border: 1px solid rgba(128,128,128,.25);
        border-radius: 16px;
        margin-bottom: 18px;
    }

    .hero h1 {
        margin: 0 0 7px;
        font-size: 37px;
    }

    .hero p {
        margin: 0;
        opacity: .75;
        font-size: 15px;
    }

    .dataset-card {
        padding: 15px 18px;
        border: 1px solid rgba(128,128,128,.25);
        border-radius: 12px;
        margin-bottom: 18px;
    }

    .model-card {
        padding: 18px;
        border: 1px solid rgba(128,128,128,.25);
        border-radius: 12px;
        height: 100%;
    }

    .small-muted {
        opacity: .68;
        font-size: 13px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 4. GENERAL HELPERS
# ============================================================

def money(value):
    return f"${float(value):,.2f}"


def has_col(df, column):
    return column in df.columns and df[column].notna().any()


def first_existing(df, candidates):
    for column in candidates:
        if column in df.columns:
            return column
    return None


def standardize_columns(df):
    df = df.copy()

    df.columns = [
        str(c)
        .strip()
        .lower()
        .replace("/", "_")
        .replace("-", "_")
        .replace(" ", "_")
        for c in df.columns
    ]

    aliases = {
        "orderid": "order_id",
        "order_date": "order_date",
        "orderdate": "order_date",
        "shipdate": "ship_date",
        "ship_date": "ship_date",
        "customerid": "customer_id",
        "customername": "customer_name",
        "productid": "product_id",
        "productname": "product_name",
        "postalcode": "postal_code",
        "country_region": "countryregion",
        "countryregion": "countryregion",
        "shippingcost": "shipping_cost",
        "customer_unique_id": "customer_unique_id",
        "customer_zip_code": "customer_zip_code_prefix",
    }

    df = df.rename(
        columns={c: aliases.get(c, c) for c in df.columns}
    )

    for c in ["order_date", "ship_date"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")

    numeric_cols = [
        "sales",
        "profit",
        "quantity",
        "discount",
        "shipping_cost",
        "order_month",
        "order_year",
        "customer_zip_code_prefix",
        "zip_prefix",
    ]

    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "order_year" not in df.columns and "order_date" in df.columns:
        df["order_year"] = df["order_date"].dt.year

    if "order_month" not in df.columns and "order_date" in df.columns:
        df["order_month"] = df["order_date"].dt.month

    if (
        "profit_margin" not in df.columns
        and "profit" in df.columns
        and "sales" in df.columns
    ):
        df["profit_margin"] = (
            df["profit"] / df["sales"].replace(0, np.nan)
        ) * 100

    if (
        "shipping_days" not in df.columns
        and "order_date" in df.columns
        and "ship_date" in df.columns
    ):
        df["shipping_days"] = (
            df["ship_date"] - df["order_date"]
        ).dt.days

    if (
        "zip_prefix" not in df.columns
        and "customer_zip_code_prefix" in df.columns
    ):
        df["zip_prefix"] = df["customer_zip_code_prefix"]

    return df


def chart_layout(fig, height=430):
    fig.update_layout(
        height=height,
        margin=dict(l=20, r=20, t=60, b=25),
        legend_title_text="",
    )
    return fig


def show_table(df, formats=None):
    if df is None or df.empty:
        st.info("No data available for this view.")
        return

    try:
        if formats:
            st.dataframe(
                df.style.format(formats),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True,
            )
    except Exception:
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
        )


# ============================================================
# 5. DATA READING
# ============================================================

def read_file_bytes(file_name, raw_bytes):
    name = file_name.lower()
    buffer = io.BytesIO(raw_bytes)

    if name.endswith(".zip"):
        with zipfile.ZipFile(buffer) as archive:
            csv_files = [
                x for x in archive.namelist()
                if x.lower().endswith(".csv")
            ]

            if not csv_files:
                raise ValueError("The ZIP file does not contain a CSV file.")

            selected = csv_files[0]

            if "global" in name:
                matches = [
                    x for x in csv_files
                    if "global_superstore" in x.lower()
                ]
                if matches:
                    selected = matches[0]

            if "olist" in name or "customer" in name:
                matches = [
                    x for x in csv_files
                    if "customer" in x.lower()
                ]
                if matches:
                    selected = matches[0]

            with archive.open(selected) as f:
                return pd.read_csv(f, low_memory=False)

    if name.endswith(".csv"):
        return pd.read_csv(buffer, low_memory=False)

    if name.endswith(".xls") or name.endswith(".xlsx"):
        excel = pd.ExcelFile(buffer)
        sheet = "Orders" if "Orders" in excel.sheet_names else excel.sheet_names[0]
        return pd.read_excel(buffer, sheet_name=sheet)

    raise ValueError("Supported formats are CSV, XLS, XLSX and ZIP containing CSV.")


@st.cache_data(show_spinner=False)
def read_project_file(path_string):
    path = Path(path_string)

    with open(path, "rb") as f:
        raw = f.read()

    return standardize_columns(
        read_file_bytes(path.name, raw)
    )


def get_uploaded_dataset(dataset_name):
    uploaded = st.session_state.get("uploaded_datasets", {})
    item = uploaded.get(dataset_name)

    if item is None:
        return None

    return standardize_columns(
        read_file_bytes(
            item["name"],
            item["bytes"]
        )
    )


def load_dataset(dataset_name):
    uploaded = get_uploaded_dataset(dataset_name)

    if uploaded is not None:
        return uploaded, "Uploaded during this Streamlit session"

    for path in DATASET_CONFIG[dataset_name]["files"]:
        if path.exists():
            return read_project_file(str(path)), f"Project file: {path.name}"

    return None, None


# ============================================================
# 6. DATASET UPLOADS
# ============================================================

def upload_section():
    st.sidebar.markdown("---")

    with st.sidebar.expander(
        "📤 Upload / Replace All 3 Datasets",
        expanded=False,
    ):
        st.caption(
            "You can upload all three datasets. "
            "The selected dataset is switched from the Dataset selector above."
        )

        upload_specs = [
            (
                "Dataset 1 — Superstore",
                "upload_superstore",
                ["csv", "xls", "xlsx"],
            ),
            (
                "Dataset 2 — Global Superstore",
                "upload_global",
                ["csv", "zip"],
            ),
            (
                "Dataset 3 — Olist Customers",
                "upload_olist",
                ["csv", "zip"],
            ),
        ]

        for dataset_name, key, types in upload_specs:
            file = st.file_uploader(
                dataset_name,
                type=types,
                key=key,
            )

            if file is not None:
                st.session_state.setdefault(
                    "uploaded_datasets",
                    {}
                )

                st.session_state[
                    "uploaded_datasets"
                ][dataset_name] = {
                    "name": file.name,
                    "bytes": file.getvalue(),
                }

                st.success("Loaded")


# ============================================================
# 7. DATASET-SPECIFIC FILTERS
# ============================================================

def render_filters(df, dataset_name):
    filtered = df.copy()

    st.sidebar.markdown("---")
    st.sidebar.subheader("🎛️ Filters")

    # Dataset 1
    if dataset_name == "Dataset 1 — Superstore":

        if has_col(df, "category"):
            values = sorted(
                df["category"].dropna().astype(str).unique()
            )
            selected = st.sidebar.multiselect(
                "Category",
                values,
                default=values,
                key="filter_d1_category",
            )
            filtered = filtered[
                filtered["category"].astype(str).isin(selected)
            ]

        if has_col(df, "region"):
            values = sorted(
                df["region"].dropna().astype(str).unique()
            )
            selected = st.sidebar.multiselect(
                "Region",
                values,
                default=values,
                key="filter_d1_region",
            )
            filtered = filtered[
                filtered["region"].astype(str).isin(selected)
            ]

        if has_col(df, "order_year"):
            values = sorted(
                pd.to_numeric(
                    df["order_year"],
                    errors="coerce"
                ).dropna().astype(int).unique()
            )
            selected = st.sidebar.multiselect(
                "Order Year",
                values,
                default=values,
                key="filter_d1_year",
            )
            filtered = filtered[
                pd.to_numeric(
                    filtered["order_year"],
                    errors="coerce"
                ).isin(selected)
            ]

    # Dataset 2
    elif dataset_name == "Dataset 2 — Global Superstore":

        if has_col(df, "market"):
            values = sorted(
                df["market"].dropna().astype(str).unique()
            )
            selected = st.sidebar.multiselect(
                "Market",
                values,
                default=values,
                key="filter_d2_market",
            )
            filtered = filtered[
                filtered["market"].astype(str).isin(selected)
            ]

        if has_col(df, "category"):
            values = sorted(
                df["category"].dropna().astype(str).unique()
            )
            selected = st.sidebar.multiselect(
                "Category",
                values,
                default=values,
                key="filter_d2_category",
            )
            filtered = filtered[
                filtered["category"].astype(str).isin(selected)
            ]

        if has_col(df, "region"):
            values = sorted(
                df["region"].dropna().astype(str).unique()
            )
            selected = st.sidebar.multiselect(
                "Region",
                values,
                default=values,
                key="filter_d2_region",
            )
            filtered = filtered[
                filtered["region"].astype(str).isin(selected)
            ]

        if has_col(df, "order_year"):
            values = sorted(
                pd.to_numeric(
                    df["order_year"],
                    errors="coerce"
                ).dropna().astype(int).unique()
            )
            selected = st.sidebar.multiselect(
                "Order Year",
                values,
                default=values,
                key="filter_d2_year",
            )
            filtered = filtered[
                pd.to_numeric(
                    filtered["order_year"],
                    errors="coerce"
                ).isin(selected)
            ]

    # Dataset 3
    else:

        if has_col(df, "customer_state"):
            values = sorted(
                df["customer_state"]
                .dropna()
                .astype(str)
                .unique()
            )

            # A selectbox is smoother for Olist because there are fewer
            # useful geographic categories than transaction filters.
            selected = st.sidebar.selectbox(
                "Customer State",
                ["All States"] + values,
                key="filter_d3_state",
            )

            if selected != "All States":
                filtered = filtered[
                    filtered["customer_state"]
                    .astype(str)
                    == selected
                ]

        if has_col(df, "customer_city"):
            st.sidebar.caption(
                "City filtering is optional and available below."
            )

            city_search = st.sidebar.text_input(
                "City contains",
                "",
                key="filter_d3_city",
            )

            if city_search.strip():
                filtered = filtered[
                    filtered["customer_city"]
                    .astype(str)
                    .str.contains(
                        city_search.strip(),
                        case=False,
                        na=False,
                    )
                ]

    st.sidebar.markdown("---")
    st.sidebar.write(
        f"Showing **{len(filtered):,}** of **{len(df):,}** records"
    )

    return filtered


# ============================================================
# 8. SAVED ARTIFACTS FOR DATASET 1
# ============================================================

@st.cache_data(show_spinner=False)
def load_xai_file():
    if XAI_FILE.exists():
        return pd.read_csv(XAI_FILE)
    return None


@st.cache_data(show_spinner=False)
def load_recommendations_file():
    if RECOMMENDATION_FILE.exists():
        return pd.read_csv(RECOMMENDATION_FILE)
    return None


@st.cache_resource(show_spinner=False)
def load_saved_model():
    if MODEL_FILE.exists():
        with open(MODEL_FILE, "rb") as f:
            return pickle.load(f)
    return None


# ============================================================
# 9. TRANSACTION DATASET ML — 6-MODEL COMPARATIVE PIPELINE
# ============================================================

D1_ALL_FEATURES = [
    "sales", "quantity", "discount", "shipping_days",
    "category", "subcategory", "region", "stateprovince",
    "segment", "ship_mode", "order_year", "order_month",
]

D1_SELECTED_FEATURES = [
    "sales", "discount", "subcategory", "stateprovince",
    "category", "quantity", "region", "order_month",
]

D2_ALL_FEATURES = [
    "sales", "quantity", "discount", "shipping_cost",
    "market", "category", "subcategory", "region",
    "segment", "ship_mode", "order_year", "order_month",
]

D2_SELECTED_FEATURES = [
    "sales", "quantity", "discount", "shipping_cost",
    "market", "category", "subcategory", "region",
]


def regression_metrics(y_true, prediction):
    y_true = np.asarray(y_true, dtype=float)
    prediction = np.asarray(prediction, dtype=float)

    mae = mean_absolute_error(y_true, prediction)
    rmse = mean_squared_error(y_true, prediction) ** 0.5

    # Profit can be zero or negative, so percentage errors are reported
    # with a small denominator floor. They should be interpreted cautiously.
    denominator = np.maximum(np.abs(y_true), 1e-8)
    mape = np.mean(np.abs((y_true - prediction) / denominator)) * 100

    smape_denominator = np.maximum(
        np.abs(y_true) + np.abs(prediction),
        1e-8,
    )
    smape = (
        np.mean(
            2 * np.abs(y_true - prediction) / smape_denominator
        ) * 100
    )

    return {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape,
        "sMAPE": smape,
        "R²": r2_score(y_true, prediction),
    }


def _make_preprocessor(numeric, categorical):
    transformers = []

    if numeric:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    [("imputer", SimpleImputer(strategy="median"))]
                ),
                numeric,
            )
        )

    if categorical:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "encoder",
                            OneHotEncoder(
                                handle_unknown="ignore",
                                sparse_output=False,
                            ),
                        ),
                    ]
                ),
                categorical,
            )
        )

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )


def _build_pipeline(features, df, estimator):
    numeric = [
        c for c in features
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c])
    ]
    categorical = [
        c for c in features
        if c in df.columns and c not in numeric
    ]

    return Pipeline(
        [
            (
                "preprocessor",
                _make_preprocessor(numeric, categorical),
            ),
            ("model", estimator),
        ]
    )


@st.cache_resource(show_spinner=True)
def train_transaction_models(df_for_training, dataset_name):
    """Train the six faculty-required regression models on the active data."""
    df = df_for_training.copy()

    if "profit" not in df.columns:
        return None

    if dataset_name == "Dataset 1 — Superstore":
        all_candidates = D1_ALL_FEATURES
        selected_candidates = D1_SELECTED_FEATURES
        max_rows = 30000
    else:
        all_candidates = D2_ALL_FEATURES
        selected_candidates = D2_SELECTED_FEATURES
        max_rows = 25000

    all_features = [c for c in all_candidates if c in df.columns]
    selected_features = [c for c in selected_candidates if c in df.columns]

    if not selected_features or "sales" not in selected_features:
        return None

    model_df = df[all_features + ["profit"]].copy()
    model_df = model_df.dropna(subset=["profit"])

    if len(model_df) < 50:
        return None

    if len(model_df) > max_rows:
        model_df = model_df.sample(max_rows, random_state=42)

    from sklearn.model_selection import train_test_split

    train_df, test_df = train_test_split(
        model_df,
        test_size=0.20,
        random_state=42,
    )

    y_train = train_df["profit"]
    y_test = test_df["profit"]

    predictions = {}
    fitted_models = {}

    # 1. Naive baseline
    naive_value = float(y_train.mean())
    predictions["Naive Model"] = np.full(len(y_test), naive_value)

    # 2. Ridge Regression
    ridge = _build_pipeline(
        all_features,
        model_df,
        Ridge(alpha=1.0),
    )
    ridge.fit(train_df[all_features], y_train)
    predictions["Ridge Regression"] = ridge.predict(test_df[all_features])
    fitted_models["Ridge Regression"] = ridge

    # 3. Random Forest - all features
    rf_all = _build_pipeline(
        all_features,
        model_df,
        RandomForestRegressor(
            n_estimators=150,
            random_state=42,
            n_jobs=-1,
            max_features="sqrt",
        ),
    )
    rf_all.fit(train_df[all_features], y_train)
    predictions["Random Forest"] = rf_all.predict(test_df[all_features])
    fitted_models["Random Forest"] = rf_all

    # 4. Random Forest - selected features
    rf_selected = _build_pipeline(
        selected_features,
        model_df,
        RandomForestRegressor(
            n_estimators=150,
            random_state=42,
            n_jobs=-1,
            max_features="sqrt",
        ),
    )
    rf_selected.fit(train_df[selected_features], y_train)
    predictions["RF Selected"] = rf_selected.predict(test_df[selected_features])
    fitted_models["RF Selected"] = rf_selected

    # 5. Gradient Boosting
    gb = _build_pipeline(
        selected_features,
        model_df,
        GradientBoostingRegressor(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=3,
            random_state=42,
            loss="huber",
        ),
    )
    gb.fit(train_df[selected_features], y_train)
    predictions["Gradient Boosting"] = gb.predict(test_df[selected_features])
    fitted_models["Gradient Boosting"] = gb

    # 6. Proposed hybrid candidate: equal-weight RF Selected + GB.
    # It is explicitly presented as a candidate hybrid in the dashboard;
    # novelty is established separately by the literature review.
    predictions["Proposed Hybrid"] = (
        0.50 * predictions["RF Selected"]
        + 0.50 * predictions["Gradient Boosting"]
    )

    rows = []
    for model_name, prediction in predictions.items():
        row = regression_metrics(y_test, prediction)
        row["Model"] = model_name
        row["Training Records"] = len(train_df)
        row["Test Records"] = len(test_df)
        rows.append(row)

    metrics = pd.DataFrame(rows)[
        [
            "Model",
            "R²",
            "MAE",
            "RMSE",
            "MAPE",
            "sMAPE",
            "Training Records",
            "Test Records",
        ]
    ]

    # Rank only by error metrics; higher R² is better.
    rank_df = metrics.copy()
    for col in ["MAE", "RMSE", "MAPE", "sMAPE"]:
        rank_df[col + "_rank"] = rank_df[col].rank(method="average")
    rank_df["R2_rank"] = rank_df["R²"].rank(ascending=False, method="average")
    rank_cols = ["MAE_rank", "RMSE_rank", "MAPE_rank", "sMAPE_rank", "R2_rank"]
    rank_df["Average Rank"] = rank_df[rank_cols].mean(axis=1)
    rank_df = rank_df.sort_values(["Average Rank", "RMSE"])
    rank_df["Overall Rank"] = np.arange(1, len(rank_df) + 1)

    metrics = metrics.merge(
        rank_df[["Model", "Average Rank", "Overall Rank"]],
        on="Model",
        how="left",
    ).sort_values("Overall Rank")

    return {
        "metrics": metrics.reset_index(drop=True),
        "models": fitted_models,
        "predictions": predictions,
        "y_test": y_test,
        "all_features": all_features,
        "selected_features": selected_features,
        "naive_value": naive_value,
        "train_df": train_df,
        "test_df": test_df,
    }


# Backward-compatible Dataset 2 wrapper used by the rest of the dashboard.
def train_dataset2_models(df_for_training):
    return train_transaction_models(
        df_for_training,
        "Dataset 2 — Global Superstore",
    )


# ============================================================
# 10. DATASET 3 ML — CUSTOMER SEGMENTATION
# ============================================================
# ============================================================

@st.cache_data(show_spinner=True)
def run_dataset3_clustering(df):
    working = df.copy()

    zip_col = first_existing(
        working,
        [
            "customer_zip_code_prefix",
            "zip_prefix",
        ]
    )

    state_col = first_existing(
        working,
        [
            "customer_state",
            "state",
        ]
    )

    if zip_col is None:
        return None

    data = pd.DataFrame()

    data["zip_prefix"] = pd.to_numeric(
        working[zip_col],
        errors="coerce"
    )

    data = data.dropna()

    if len(data) < 10:
        return None

    # Limit only the clustering computation, not dashboard data.
    sample = data.copy()

    if len(sample) > 20000:
        sample = sample.sample(
            20000,
            random_state=42
        )

    X = StandardScaler().fit_transform(
        sample[["zip_prefix"]]
    )

    kmeans = KMeans(
        n_clusters=4,
        random_state=42,
        n_init=10,
    )

    labels = kmeans.fit_predict(X)

    sample["Cluster"] = labels + 1

    counts = (
        sample["Cluster"]
        .value_counts()
        .sort_index()
        .reset_index()
    )

    counts.columns = [
        "Cluster",
        "Customers",
    ]

    return {
        "data": sample,
        "counts": counts,
        "k": 4,
        "zip_col": zip_col,
        "state_col": state_col,
    }


# ============================================================
# 11. MODEL / ALGORITHM CATALOGS
# ============================================================

def dataset1_algorithm_table():
    return pd.DataFrame(
        [
            ["Naive Model", "Baseline Regression", "Reference prediction", "Predicts the training-set mean profit.", "Implemented + evaluated"],
            ["Ridge Regression", "Regularized Regression", "Linear baseline", "Controls coefficient magnitude and provides a regularized linear benchmark.", "Implemented + evaluated"],
            ["Random Forest", "Ensemble Regression", "Non-linear baseline", "Captures non-linear relationships using all candidate features.", "Implemented + evaluated"],
            ["Random Forest Selected", "Ensemble Regression", "Feature-selected model", "Random Forest trained on the project-selected feature subset.", "Implemented + evaluated"],
            ["Gradient Boosting", "Boosting Regression", "Strong non-linear baseline", "Sequentially reduces residual prediction error.", "Implemented + evaluated"],
            ["Proposed Hybrid", "Hybrid Ensemble", "Candidate final model", "50% RF Selected + 50% Gradient Boosting prediction average.", "Implemented + evaluated"],
            ["Mutual Information + RF Importance", "Feature Selection", "Hybrid feature selection", "Combines information-based and model-based relevance.", "Implemented"],
            ["Permutation Importance", "Explainable AI", "Model explanation", "Measures prediction degradation when features are permuted.", "Implemented"],
            ["One-Hot Encoding", "Preprocessing", "Categorical features", "Converts business categories into model-ready variables.", "Implemented"],
        ],
        columns=["Algorithm / Method", "Type", "Project Role", "Contribution", "Status"],
    )


def dataset2_algorithm_table():
    return pd.DataFrame(
        [
            ["Naive Model", "Baseline Regression", "Reference prediction", "Uses the training-set mean profit as the prediction.", "Implemented + evaluated"],
            ["Ridge Regression", "Regularized Regression", "Linear baseline", "Provides a regularized linear benchmark.", "Implemented + evaluated"],
            ["Random Forest", "Ensemble Regression", "Non-linear baseline", "Uses all available candidate Global Superstore features.", "Implemented + evaluated"],
            ["Random Forest Selected", "Ensemble Regression", "Feature-selected model", "Uses the reduced Global Superstore feature subset.", "Implemented + evaluated"],
            ["Gradient Boosting", "Boosting Regression", "Strong non-linear baseline", "Learns residual corrections sequentially.", "Implemented + evaluated"],
            ["Proposed Hybrid", "Hybrid Ensemble", "Candidate final model", "50% RF Selected + 50% Gradient Boosting prediction average.", "Implemented + evaluated"],
            ["One-Hot Encoding", "Preprocessing", "Categorical features", "Encodes market, category, subcategory, segment and regional variables.", "Implemented"],
            ["Train-Test Split", "Evaluation", "Model validation", "Uses a held-out test set for unbiased model comparison.", "Implemented"],
        ],
        columns=["Algorithm / Method", "Type", "Project Role", "Contribution", "Status"],
    )


def dataset3_algorithm_table():
    return pd.DataFrame(
        [
            [
                "K-Means Clustering",
                "Unsupervised ML",
                "Geographic customer segmentation",
                "Groups customer records into four exploratory ZIP-prefix clusters.",
                "Implemented in dashboard",
            ],
            [
                "StandardScaler",
                "Preprocessing",
                "Clustering preparation",
                "Standardizes ZIP-prefix values before clustering.",
                "Implemented",
            ],
        ],
        columns=[
            "Algorithm / Method",
            "Type",
            "Project Role",
            "Contribution",
            "Status",
        ]
    )


# ============================================================
# 12. SIDEBAR NAVIGATION
# ============================================================

st.sidebar.title("📊 Navigation")

page = st.sidebar.radio(
    "Section",
    [
        "🏠 Overview",
        "📈 Analytics",
        "🤖 Profit Prediction",
        "🔍 Explainable AI",
        "💡 Recommendations",
        "🧠 ML Models & Algorithms",
        "💬 Ask the AI",
        "📊 Comparative Analysis",
    ],
    key="main_navigation",
)


# ============================================================
# 13. DATASET SELECTOR — IMMEDIATELY AFTER NAVIGATION
# ============================================================

st.sidebar.markdown("---")
st.sidebar.subheader("📚 Dataset")

dataset_name = st.sidebar.selectbox(
    "Select Dataset",
    list(DATASET_CONFIG.keys()),
    key="active_dataset",
)


# Upload controls are below the dataset selector.
upload_section()


# ============================================================
# 14. LOAD ACTIVE DATASET
# ============================================================

df, source = load_dataset(dataset_name)

if df is None:
    st.error(
        f"""
        ## {dataset_name} is not available.

        Please use **📤 Upload / Replace All 3 Datasets**
        in the sidebar or place the dataset in the project's
        `datasets` folder.
        """
    )
    st.stop()


# Dataset-specific filters.
filtered_df = render_filters(
    df,
    dataset_name
)


# ============================================================
# 15. ACTIVE DATASET HEADER
# ============================================================

st.markdown(
    """
    <div class="hero">
        <h1>AI-Assisted Business Decision Support System</h1>
        <p>
            Interactive business analytics • machine learning •
            explainable AI • decision support
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


st.markdown(
    f"""
    <div class="dataset-card">
        <b>Active Dataset:</b> {dataset_name}<br>
        <span class="small-muted">
            {DATASET_CONFIG[dataset_name]["description"]}
        </span><br>
        <span class="small-muted">
            Source: {source}
        </span>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# PAGE 1 — OVERVIEW
# ============================================================

if page == "🏠 Overview":

    st.header(
        f"Business Overview — {dataset_name}"
    )

    if filtered_df.empty:
        st.warning("No records match the selected filters.")
        st.stop()

    if DATASET_CONFIG[dataset_name]["type"] == "transaction":

        sales = (
            filtered_df["sales"].sum()
            if has_col(filtered_df, "sales")
            else 0
        )

        profit = (
            filtered_df["profit"].sum()
            if has_col(filtered_df, "profit")
            else 0
        )

        quantity = (
            filtered_df["quantity"].sum()
            if has_col(filtered_df, "quantity")
            else 0
        )

        avg_discount = (
            filtered_df["discount"].mean()
            if has_col(filtered_df, "discount")
            else np.nan
        )

        if has_col(filtered_df, "profit_margin"):
            avg_margin = filtered_df["profit_margin"].mean()
        elif sales != 0 and has_col(filtered_df, "profit"):
            avg_margin = (profit / sales) * 100
        else:
            avg_margin = np.nan

        c1, c2, c3, c4, c5 = st.columns(5)

        c1.metric("Total Sales", money(sales))
        c2.metric("Total Profit", money(profit))
        c3.metric("Total Quantity", f"{quantity:,.0f}")
        c4.metric(
            "Average Discount",
            f"{avg_discount:.2%}" if pd.notna(avg_discount) else "N/A"
        )
        c5.metric(
            "Average Profit Margin",
            f"{avg_margin:.2f}%" if pd.notna(avg_margin) else "N/A"
        )

        st.markdown("---")

        if has_col(filtered_df, "category"):

            category = (
                filtered_df
                .groupby("category")
                .agg(
                    Sales=("sales", "sum"),
                    Profit=("profit", "sum"),
                    Quantity=("quantity", "sum"),
                )
                .reset_index()
            )

            a, b = st.columns(2)

            with a:
                st.plotly_chart(
                    chart_layout(
                        px.bar(
                            category,
                            x="category",
                            y="Sales",
                            title="Sales by Category",
                        )
                    ),
                    use_container_width=True,
                )

            with b:
                st.plotly_chart(
                    chart_layout(
                        px.bar(
                            category,
                            x="category",
                            y="Profit",
                            title="Profit by Category",
                        )
                    ),
                    use_container_width=True,
                )

            st.subheader("Category Summary")
            show_table(
                category,
                {
                    "Sales": "{:,.2f}",
                    "Profit": "{:,.2f}",
                    "Quantity": "{:,.0f}",
                }
            )

    else:

        st.subheader("Customer Dataset Overview")

        unique_customers = (
            filtered_df["customer_unique_id"].nunique()
            if has_col(filtered_df, "customer_unique_id")
            else np.nan
        )

        missing = int(
            filtered_df.isna().sum().sum()
        )

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Customer Records",
            f"{len(filtered_df):,}"
        )

        c2.metric(
            "Unique Customers",
            f"{unique_customers:,.0f}"
            if pd.notna(unique_customers)
            else "N/A"
        )

        c3.metric(
            "Columns",
            f"{len(filtered_df.columns)}"
        )

        c4.metric(
            "Missing Cells",
            f"{missing:,}"
        )

        if has_col(filtered_df, "customer_state"):

            state = (
                filtered_df["customer_state"]
                .value_counts()
                .reset_index()
            )

            state.columns = [
                "State",
                "Customer Records",
            ]

            st.plotly_chart(
                chart_layout(
                    px.bar(
                        state.head(20),
                        x="State",
                        y="Customer Records",
                        title="Top States by Customer Records",
                    )
                ),
                use_container_width=True,
            )

            show_table(state.head(30))


# ============================================================
# PAGE 2 — ANALYTICS
# ============================================================

elif page == "📈 Analytics":

    st.header(
        f"Analytics — {dataset_name}"
    )

    if filtered_df.empty:
        st.warning("No records match the selected filters.")
        st.stop()

    if dataset_name == "Dataset 1 — Superstore":

        tabs = st.tabs(
            [
                "Category",
                "Region",
                "Yearly Trend",
                "Discount",
                "Top Products",
            ]
        )

        with tabs[0]:
            summary = (
                filtered_df
                .groupby("category")
                .agg(
                    Sales=("sales", "sum"),
                    Profit=("profit", "sum"),
                    Quantity=("quantity", "sum"),
                    Average_Discount=("discount", "mean"),
                )
                .reset_index()
            )

            a, b = st.columns(2)

            with a:
                st.plotly_chart(
                    chart_layout(
                        px.bar(
                            summary,
                            x="category",
                            y="Sales",
                            title="Sales by Category",
                        )
                    ),
                    use_container_width=True,
                )

            with b:
                st.plotly_chart(
                    chart_layout(
                        px.bar(
                            summary,
                            x="category",
                            y="Profit",
                            title="Profit by Category",
                        )
                    ),
                    use_container_width=True,
                )

            show_table(
                summary,
                {
                    "Sales": "{:,.2f}",
                    "Profit": "{:,.2f}",
                    "Quantity": "{:,.0f}",
                    "Average_Discount": "{:.2%}",
                }
            )

        with tabs[1]:
            summary = (
                filtered_df
                .groupby("region")
                .agg(
                    Sales=("sales", "sum"),
                    Profit=("profit", "sum"),
                    Quantity=("quantity", "sum"),
                )
                .reset_index()
            )

            a, b = st.columns(2)

            with a:
                st.plotly_chart(
                    chart_layout(
                        px.bar(
                            summary,
                            x="region",
                            y="Sales",
                            title="Sales by Region",
                        )
                    ),
                    use_container_width=True,
                )

            with b:
                st.plotly_chart(
                    chart_layout(
                        px.bar(
                            summary,
                            x="region",
                            y="Profit",
                            title="Profit by Region",
                        )
                    ),
                    use_container_width=True,
                )

            show_table(summary)

        with tabs[2]:
            if has_col(filtered_df, "order_year"):
                summary = (
                    filtered_df
                    .groupby("order_year")
                    .agg(
                        Sales=("sales", "sum"),
                        Profit=("profit", "sum"),
                    )
                    .reset_index()
                    .sort_values("order_year")
                )

                st.plotly_chart(
                    chart_layout(
                        px.line(
                            summary,
                            x="order_year",
                            y=["Sales", "Profit"],
                            markers=True,
                            title="Yearly Sales and Profit Trend",
                        )
                    ),
                    use_container_width=True,
                )

                show_table(summary)

        with tabs[3]:
            summary = (
                filtered_df
                .groupby("discount")
                .agg(
                    Sales=("sales", "sum"),
                    Profit=("profit", "sum"),
                    Quantity=("quantity", "sum"),
                )
                .reset_index()
                .sort_values("discount")
            )

            st.plotly_chart(
                chart_layout(
                    px.scatter(
                        filtered_df,
                        x="discount",
                        y="profit",
                        hover_data=["sales", "quantity"],
                        title="Observed Discount vs Profit",
                    ),
                    480,
                ),
                use_container_width=True,
            )

            show_table(summary)

            st.info(
                "Observed association is not proof that discount causes changes in profit."
            )

        with tabs[4]:
            summary = (
                filtered_df
                .groupby("product_name")
                .agg(
                    Sales=("sales", "sum"),
                    Profit=("profit", "sum"),
                )
                .reset_index()
                .sort_values("Profit", ascending=False)
                .head(10)
            )

            st.plotly_chart(
                chart_layout(
                    px.bar(
                        summary.sort_values("Profit"),
                        x="Profit",
                        y="product_name",
                        orientation="h",
                        title="Top 10 Products by Profit",
                    ),
                    500,
                ),
                use_container_width=True,
            )

            show_table(summary)

    elif dataset_name == "Dataset 2 — Global Superstore":

        tabs = st.tabs(
            [
                "Market",
                "Category",
                "Region",
                "Yearly Trend",
                "Top Countries",
            ]
        )

        with tabs[0]:
            if has_col(filtered_df, "market"):
                summary = (
                    filtered_df
                    .groupby("market")
                    .agg(
                        Sales=("sales", "sum"),
                        Profit=("profit", "sum"),
                        Quantity=("quantity", "sum"),
                    )
                    .reset_index()
                )

                a, b = st.columns(2)

                with a:
                    st.plotly_chart(
                        chart_layout(
                            px.bar(
                                summary,
                                x="market",
                                y="Sales",
                                title="Global Sales by Market",
                            )
                        ),
                        use_container_width=True,
                    )

                with b:
                    st.plotly_chart(
                        chart_layout(
                            px.bar(
                                summary,
                                x="market",
                                y="Profit",
                                title="Global Profit by Market",
                            )
                        ),
                        use_container_width=True,
                    )

                show_table(summary)

        with tabs[1]:
            if has_col(filtered_df, "category"):
                summary = (
                    filtered_df
                    .groupby("category")
                    .agg(
                        Sales=("sales", "sum"),
                        Profit=("profit", "sum"),
                        Quantity=("quantity", "sum"),
                    )
                    .reset_index()
                )

                a, b = st.columns(2)

                with a:
                    st.plotly_chart(
                        chart_layout(
                            px.bar(
                                summary,
                                x="category",
                                y="Sales",
                                title="Global Sales by Category",
                            )
                        ),
                        use_container_width=True,
                    )

                with b:
                    st.plotly_chart(
                        chart_layout(
                            px.bar(
                                summary,
                                x="category",
                                y="Profit",
                                title="Global Profit by Category",
                            )
                        ),
                        use_container_width=True,
                    )

                show_table(summary)

        with tabs[2]:
            if has_col(filtered_df, "region"):
                summary = (
                    filtered_df
                    .groupby("region")
                    .agg(
                        Sales=("sales", "sum"),
                        Profit=("profit", "sum"),
                    )
                    .reset_index()
                )

                st.plotly_chart(
                    chart_layout(
                        px.bar(
                            summary,
                            x="region",
                            y="Profit",
                            title="Global Profit by Region",
                        )
                    ),
                    use_container_width=True,
                )

                show_table(summary)

        with tabs[3]:
            if has_col(filtered_df, "order_year"):
                summary = (
                    filtered_df
                    .groupby("order_year")
                    .agg(
                        Sales=("sales", "sum"),
                        Profit=("profit", "sum"),
                    )
                    .reset_index()
                    .sort_values("order_year")
                )

                st.plotly_chart(
                    chart_layout(
                        px.line(
                            summary,
                            x="order_year",
                            y=["Sales", "Profit"],
                            markers=True,
                            title="Global Yearly Sales and Profit",
                        )
                    ),
                    use_container_width=True,
                )

                show_table(summary)

        with tabs[4]:

            country_col = first_existing(
                filtered_df,
                ["country", "countryregion"]
            )

            if country_col:

                summary = (
                    filtered_df
                    .groupby(country_col)
                    .agg(
                        Sales=("sales", "sum"),
                        Profit=("profit", "sum"),
                    )
                    .reset_index()
                    .sort_values(
                        "Profit",
                        ascending=False
                    )
                    .head(15)
                )

                st.plotly_chart(
                    chart_layout(
                        px.bar(
                            summary.sort_values("Profit"),
                            x="Profit",
                            y=country_col,
                            orientation="h",
                            title="Top 15 Countries by Profit",
                        )
                    ),
                    use_container_width=True,
                )

                show_table(summary)

    else:

        tabs = st.tabs(
            [
                "State Distribution",
                "City Distribution",
                "Customer Frequency",
            ]
        )

        with tabs[0]:

            if has_col(filtered_df, "customer_state"):

                state = (
                    filtered_df["customer_state"]
                    .value_counts()
                    .reset_index()
                )

                state.columns = [
                    "State",
                    "Customers",
                ]

                st.plotly_chart(
                    chart_layout(
                        px.bar(
                            state.head(20),
                            x="State",
                            y="Customers",
                            title="Top 20 Customer States",
                        )
                    ),
                    use_container_width=True,
                )

                show_table(state.head(30))

        with tabs[1]:

            if has_col(filtered_df, "customer_city"):

                city = (
                    filtered_df["customer_city"]
                    .astype(str)
                    .str.title()
                    .value_counts()
                    .reset_index()
                    .head(20)
                )

                city.columns = [
                    "City",
                    "Customers",
                ]

                st.plotly_chart(
                    chart_layout(
                        px.bar(
                            city.sort_values("Customers"),
                            x="Customers",
                            y="City",
                            orientation="h",
                            title="Top 20 Customer Cities",
                        )
                    ),
                    use_container_width=True,
                )

                show_table(city)

        with tabs[2]:

            if has_col(
                filtered_df,
                "customer_unique_id"
            ):

                frequency = (
                    filtered_df["customer_unique_id"]
                    .value_counts()
                    .reset_index()
                )

                frequency.columns = [
                    "Customer Unique ID",
                    "Record Frequency",
                ]

                st.plotly_chart(
                    chart_layout(
                        px.histogram(
                            frequency,
                            x="Record Frequency",
                            title="Customer Record Frequency",
                        )
                    ),
                    use_container_width=True,
                )

                show_table(
                    frequency.head(30)
                )


# ============================================================
# PAGE 3 — PROFIT PREDICTION
# ============================================================

elif page == "🤖 Profit Prediction":

    st.header(f"Profit Prediction — {dataset_name}")

    if DATASET_CONFIG[dataset_name]["type"] != "transaction":
        st.info(
            "Dataset 3 is a customer-location dataset and does not contain "
            "a transaction-level profit target. Profit regression is therefore "
            "not applied to Dataset 3."
        )

    else:
        ml_result = train_transaction_models(df, dataset_name)

        if ml_result is None:
            st.error("The active transaction dataset does not contain enough compatible data for regression.")
        else:
            st.info(
                "Six regression approaches are evaluated on the active transaction dataset: "
                "Naive, Ridge, Random Forest, Random Forest Selected, Gradient Boosting and the proposed hybrid candidate."
            )

            metrics = ml_result["metrics"]
            show_table(
                metrics,
                {
                    "R²": "{:.4f}",
                    "MAE": "{:,.2f}",
                    "RMSE": "{:,.2f}",
                    "MAPE": "{:.2f}%",
                    "sMAPE": "{:.2f}%",
                    "Average Rank": "{:.3f}",
                },
            )

            best = metrics.iloc[0]
            st.success(
                f"Best model by average rank: {best['Model']} | "
                f"RMSE = {best['RMSE']:,.2f}, MAE = {best['MAE']:,.2f}, R² = {best['R²']:.4f}."
            )

            fig = px.bar(
                metrics.sort_values("RMSE"),
                x="Model",
                y="RMSE",
                text="RMSE",
                title=f"{dataset_name} — Six-Model RMSE Comparison",
            )
            st.plotly_chart(chart_layout(fig, 500), use_container_width=True)

            st.caption(
                "MAPE and sMAPE can be unstable when profit is zero or close to zero and when profit values are negative; "
                "MAE, RMSE and R² are therefore also used for model interpretation."
            )

            st.markdown("---")
            st.subheader("Interactive Profit Prediction")

            prediction_model = st.selectbox(
                "Select a trained model",
                [
                    "Ridge Regression",
                    "Random Forest",
                    "RF Selected",
                    "Gradient Boosting",
                    "Proposed Hybrid",
                ],
                index=4,
                key=f"prediction_model_{dataset_name}",
            )

            if dataset_name == "Dataset 1 — Superstore":
                available = df
                selected_features = ml_result["selected_features"]
            else:
                available = df
                selected_features = ml_result["selected_features"]

            input_values = {}
            left, right = st.columns(2)

            with left:
                if "sales" in selected_features:
                    input_values["sales"] = st.number_input("Sales", min_value=0.0, value=500.0, step=50.0, key=f"sales_{dataset_name}")
                if "quantity" in selected_features:
                    input_values["quantity"] = st.number_input("Quantity", min_value=1, value=3, step=1, key=f"quantity_{dataset_name}")
                if "discount" in selected_features:
                    input_values["discount"] = st.number_input("Discount", min_value=0.0, max_value=1.0, value=0.10, step=0.05, key=f"discount_{dataset_name}")
                if "shipping_cost" in selected_features:
                    input_values["shipping_cost"] = st.number_input("Shipping Cost", min_value=0.0, value=20.0, step=5.0, key=f"shipcost_{dataset_name}")
                if "order_month" in selected_features:
                    input_values["order_month"] = st.selectbox("Order Month", list(range(1, 13)), index=1, key=f"month_{dataset_name}")

            with right:
                for feature, label in [
                    ("market", "Market"),
                    ("category", "Category"),
                    ("subcategory", "Subcategory"),
                    ("stateprovince", "State / Province"),
                    ("region", "Region"),
                ]:
                    if feature in selected_features and feature in available.columns:
                        vals = sorted(available[feature].dropna().astype(str).unique())
                        if vals:
                            input_values[feature] = st.selectbox(label, vals, key=f"{feature}_{dataset_name}")

            if st.button("🔮 Predict Profit", type="primary", use_container_width=True):
                try:
                    if prediction_model == "Proposed Hybrid":
                        rf = ml_result["models"]["RF Selected"]
                        gb = ml_result["models"]["Gradient Boosting"]
                        row = pd.DataFrame([input_values])
                        rf_pred = float(rf.predict(row[ml_result["selected_features"]])[0])
                        gb_pred = float(gb.predict(row[ml_result["selected_features"]])[0])
                        prediction = 0.50 * rf_pred + 0.50 * gb_pred
                    elif prediction_model == "RF Selected":
                        model = ml_result["models"]["RF Selected"]
                        row = pd.DataFrame([input_values])
                        prediction = float(model.predict(row[ml_result["selected_features"]])[0])
                    else:
                        model = ml_result["models"][prediction_model]
                        model_features = ml_result["all_features"]
                        row = pd.DataFrame([input_values])
                        for feature in model_features:
                            if feature not in row.columns:
                                if feature in available.columns and not available[feature].dropna().empty:
                                    row[feature] = available[feature].dropna().iloc[0]
                                else:
                                    row[feature] = 0 if feature in ["sales", "quantity", "discount", "shipping_cost", "order_year", "order_month", "shipping_days"] else "Unknown"
                        prediction = float(model.predict(row[model_features])[0])

                    st.success(f"Predicted Profit using {prediction_model}: {money(prediction)}")
                    st.caption("The prediction is a model estimate, not a guaranteed business outcome.")
                except Exception as error:
                    st.error("Prediction failed for the selected model.")
                    st.code(str(error))


# ============================================================
# PAGE 4 — EXPLAINABLE AI
# ============================================================

elif page == "🔍 Explainable AI":

    st.header(f"Explainable AI — {dataset_name}")

    if DATASET_CONFIG[dataset_name]["type"] == "transaction":
        ml_result = train_transaction_models(df, dataset_name)

        if ml_result is None:
            st.error("The active transaction dataset cannot be explained because the model pipeline could not be trained.")
        else:
            st.info(
                "XAI is now model-specific. Select a model and inspect its feature relevance. "
                "Permutation importance is computed safely on a limited held-out sample with a single worker."
            )

            explainable_models = [
                "Ridge Regression",
                "Random Forest",
                "RF Selected",
                "Gradient Boosting",
                "Proposed Hybrid",
            ]
            selected_model = st.selectbox(
                "Select model to explain",
                explainable_models,
                key=f"xai_model_{dataset_name}",
            )

            method = st.radio(
                "Explainability method",
                ["Model Feature Importance / Coefficients", "Permutation Importance"],
                horizontal=True,
                key=f"xai_method_{dataset_name}",
            )

            if method == "Model Feature Importance / Coefficients":
                model = ml_result["models"][selected_model]

                if selected_model == "Proposed Hybrid":
                    rf = ml_result["models"]["RF Selected"]
                    gb = ml_result["models"]["Gradient Boosting"]
                    pre = rf.named_steps["preprocessor"]
                    names = pre.get_feature_names_out()
                    rf_imp = rf.named_steps["model"].feature_importances_
                    gb_imp = gb.named_steps["model"].feature_importances_
                    values = 0.50 * rf_imp + 0.50 * gb_imp
                    title = "Proposed Hybrid — Average Component Feature Importance"
                elif selected_model == "Ridge Regression":
                    pre = model.named_steps["preprocessor"]
                    names = pre.get_feature_names_out()
                    values = np.abs(model.named_steps["model"].coef_)
                    title = "Ridge Regression — Absolute Coefficient Magnitude"
                else:
                    pre = model.named_steps["preprocessor"]
                    names = pre.get_feature_names_out()
                    values = model.named_steps["model"].feature_importances_
                    title = f"{selected_model} — Feature Importance"

                xai_df = pd.DataFrame({"Feature": names, "Importance": values})
                xai_df = xai_df.sort_values("Importance", ascending=False).head(20)
                show_table(xai_df, {"Importance": "{:.6f}"})
                st.plotly_chart(
                    chart_layout(
                        px.bar(
                            xai_df.sort_values("Importance"),
                            x="Importance",
                            y="Feature",
                            orientation="h",
                            title=title,
                        ),
                        600,
                    ),
                    use_container_width=True,
                )

            else:
                from sklearn.inspection import permutation_importance

                feature_key = (
                    ml_result["selected_features"]
                    if selected_model in ["RF Selected", "Gradient Boosting", "Proposed Hybrid"]
                    else ml_result["all_features"]
                )

                test_x = ml_result["test_df"][feature_key].copy()
                test_y = ml_result["y_test"]

                if len(test_x) > 1000:
                    sample_idx = test_x.sample(1000, random_state=42).index
                    test_x = test_x.loc[sample_idx]
                    test_y = test_y.loc[sample_idx]

                if selected_model == "Proposed Hybrid":
                    rf = ml_result["models"]["RF Selected"]
                    gb = ml_result["models"]["Gradient Boosting"]
                    base = test_x.copy()
                    base_pred = 0.50 * rf.predict(base) + 0.50 * gb.predict(base)
                    base_mae = mean_absolute_error(test_y, base_pred)
                    rows = []
                    rng = np.random.default_rng(42)
                    for feature in feature_key:
                        shuffled = base.copy()
                        shuffled[feature] = rng.permutation(shuffled[feature].values)
                        pred = 0.50 * rf.predict(shuffled) + 0.50 * gb.predict(shuffled)
                        rows.append({"Feature": feature, "Importance": mean_absolute_error(test_y, pred) - base_mae})
                    xai_df = pd.DataFrame(rows).sort_values("Importance", ascending=False)
                else:
                    model = ml_result["models"][selected_model]
                    perm = permutation_importance(
                        model,
                        test_x,
                        test_y,
                        n_repeats=5,
                        random_state=42,
                        n_jobs=1,
                        scoring="neg_mean_absolute_error",
                    )
                    xai_df = pd.DataFrame({"Feature": feature_key, "Importance": perm.importances_mean})
                    xai_df = xai_df.sort_values("Importance", ascending=False)

                show_table(xai_df.head(20), {"Importance": "{:.6f}"})
                st.plotly_chart(
                    chart_layout(
                        px.bar(
                            xai_df.head(20).sort_values("Importance"),
                            x="Importance",
                            y="Feature",
                            orientation="h",
                            title=f"{selected_model} — Permutation Importance",
                        ),
                        600,
                    ),
                    use_container_width=True,
                )

            st.warning(
                "XAI describes model reliance on inputs; feature importance is not causal evidence. "
                "For profit data, percentage-based metrics and importance values should be interpreted in context."
            )

    else:
        st.subheader("Customer Segmentation Explanation")
        clustering = run_dataset3_clustering(df)
        if clustering is None:
            st.warning("A suitable ZIP-prefix feature was not found.")
        else:
            st.write(
                "Dataset 3 uses exploratory geographic customer clustering. "
                "The clusters describe ZIP-prefix groupings and should not be interpreted as behavioral or profitability segments."
            )
            show_table(clustering["counts"])
            st.plotly_chart(
                chart_layout(
                    px.bar(
                        clustering["counts"],
                        x="Cluster",
                        y="Customers",
                        title="Olist Geographic Customer Clusters",
                    )
                ),
                use_container_width=True,
            )


# ============================================================
# PAGE 5 — RECOMMENDATIONS
# ============================================================

elif page == "💡 Recommendations":

    st.header(f"Recommendations — {dataset_name}")

    if dataset_name == "Dataset 1 — Superstore":
        # The old CSV artifact can contain blank cells. Generate live,
        # data-backed recommendations as a reliable dashboard fallback.
        recommendations = []

        if has_col(filtered_df, "category"):
            category = filtered_df.groupby("category").agg(Sales=("sales", "sum"), Profit=("profit", "sum")).reset_index()
            best = category.loc[category["Profit"].idxmax()]
            recommendations.append(("Business Analysis", f"{best['category']} has the highest observed category profit of {money(best['Profit'])}.", "Protect availability of the strongest category and investigate the products and discount levels driving its profitability."))

        if has_col(filtered_df, "region"):
            region = filtered_df.groupby("region").agg(Sales=("sales", "sum"), Profit=("profit", "sum")).reset_index()
            best = region.loc[region["Profit"].idxmax()]
            recommendations.append(("Regional Strategy", f"{best['region']} has the highest observed regional profit of {money(best['Profit'])}.", "Study the successful product mix and pricing pattern in the strongest region before expanding similar practices."))

        if has_col(filtered_df, "subcategory"):
            sub = filtered_df.groupby("subcategory").agg(Sales=("sales", "sum"), Profit=("profit", "sum")).reset_index()
            worst = sub.loc[sub["Profit"].idxmin()]
            recommendations.append(("Profitability Risk", f"{worst['subcategory']} has the lowest observed subcategory profit of {money(worst['Profit'])}.", "Review pricing, discounting, shipping and product-level costs for this subcategory."))

        if has_col(filtered_df, "discount"):
            tmp = filtered_df.copy()
            tmp["Discount Band"] = pd.cut(tmp["discount"], bins=[-0.001, 0.10, 0.20, 0.40, 1.01], labels=["0–10%", "10–20%", "20–40%", "40%+"])
            disc = tmp.groupby("Discount Band", observed=False).agg(Profit=("profit", "mean"), Transactions=("profit", "size")).reset_index()
            worst = disc.loc[disc["Profit"].idxmin()]
            recommendations.append(("Discount Policy", f"The {worst['Discount Band']} band has the lowest average transaction profit ({money(worst['Profit'])}).", "Treat high-discount transactions as a profitability-risk segment and validate margin before extending discounts."))

        if has_col(filtered_df, "product_name"):
            product = filtered_df.groupby("product_name").agg(Sales=("sales", "sum"), Profit=("profit", "sum")).reset_index().sort_values("Profit", ascending=False).head(1)
            if not product.empty:
                best = product.iloc[0]
                recommendations.append(("Product Strategy", f"{best['product_name']} is the highest-profit observed product at {money(best['Profit'])}.", "Use the strongest products as anchors for assortment, cross-selling and inventory decisions."))

        for i, (area, finding, recommendation) in enumerate(recommendations[:5], 1):
            with st.container(border=True):
                st.subheader(f"Recommendation {i}")
                st.write(f"**Area:** {area}")
                st.write(f"**Finding:** {finding}")
                st.write(f"**Recommendation:** {recommendation}")

        if not recommendations:
            st.warning("No recommendation could be generated from the active filtered data.")

    elif dataset_name == "Dataset 2 — Global Superstore":
        st.subheader("Global Superstore Decision Support")

        if has_col(filtered_df, "market"):
            market = filtered_df.groupby("market").agg(Sales=("sales", "sum"), Profit=("profit", "sum")).reset_index().sort_values("Profit", ascending=False)
            top = market.iloc[0]
            st.success(f"Highest observed market profit in the active filters: {top['market']} ({money(top['Profit'])}).")
            st.write("Investigate products, categories, discount and shipping patterns within high-performing markets before extending those practices to other markets.")
            show_table(market)

        if has_col(filtered_df, "category"):
            category = filtered_df.groupby("category").agg(Sales=("sales", "sum"), Profit=("profit", "sum")).reset_index().sort_values("Profit")
            st.subheader("Category Profitability Investigation")
            show_table(category)

        if has_col(filtered_df, "region"):
            region = filtered_df.groupby("region").agg(Sales=("sales", "sum"), Profit=("profit", "sum")).reset_index().sort_values("Profit", ascending=False)
            st.subheader("Regional Profitability Investigation")
            show_table(region)

    else:
        st.subheader("Olist Customer Decision Support")
        if has_col(filtered_df, "customer_state"):
            state = filtered_df["customer_state"].value_counts().reset_index()
            state.columns = ["State", "Customer Records"]
            top = state.iloc[0]
            st.success(f"Highest customer-record concentration: {top['State']} ({int(top['Customer Records']):,} records).")
            st.write("Use the geographic concentration to guide customer-service, regional operations and further e-commerce analysis.")
            show_table(state.head(20))


# PAGE 6 — ML MODELS & ALGORITHMS
# DATASET-SPECIFIC
# ============================================================

elif page == "🧠 ML Models & Algorithms":

    st.header(f"🧠 ML Models & Algorithms — {dataset_name}")

    if DATASET_CONFIG[dataset_name]["type"] == "transaction":
        ml_result = train_transaction_models(df, dataset_name)

        if ml_result is None:
            st.error("The active transaction dataset could not be used for regression.")
        else:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Predictive Models", "6")
            c2.metric("Feature Selection", "Hybrid / Selected")
            c3.metric("XAI", "Model-specific + Permutation")
            c4.metric("Target", "Profit")

            st.markdown("---")
            st.subheader("Algorithm Contribution Table")
            show_table(
                dataset1_algorithm_table()
                if dataset_name == "Dataset 1 — Superstore"
                else dataset2_algorithm_table()
            )

            st.markdown("---")
            st.subheader("Actual Model Evaluation")
            show_table(
                ml_result["metrics"],
                {
                    "R²": "{:.4f}",
                    "MAE": "{:,.2f}",
                    "RMSE": "{:,.2f}",
                    "MAPE": "{:.2f}%",
                    "sMAPE": "{:.2f}%",
                    "Average Rank": "{:.3f}",
                },
            )

            st.plotly_chart(
                chart_layout(
                    px.bar(
                        ml_result["metrics"].sort_values("RMSE"),
                        x="Model",
                        y="RMSE",
                        text="RMSE",
                        title=f"{dataset_name} — Six-Model RMSE Comparison",
                    ),
                    500,
                ),
                use_container_width=True,
            )

            st.subheader("Model Comparison by R²")
            st.plotly_chart(
                chart_layout(
                    px.bar(
                        ml_result["metrics"].sort_values("R²", ascending=False),
                        x="Model",
                        y="R²",
                        text="R²",
                        title=f"{dataset_name} — R² Comparison",
                    ),
                    500,
                ),
                use_container_width=True,
            )

            st.subheader("Feature Sets")
            c1, c2 = st.columns(2)
            with c1:
                st.write("**All-feature models:**")
                st.write(ml_result["all_features"])
            with c2:
                st.write("**Selected-feature models:**")
                st.write(ml_result["selected_features"])

            st.info(
                "The proposed hybrid shown here is a candidate ensemble formed from "
                "Random Forest Selected and Gradient Boosting. Its novelty must be justified "
                "against the literature before it is claimed as the final research contribution."
            )

    else:
        st.subheader("Dataset 3: Olist Customers — Customer Segmentation")
        st.write(
            "Dataset 3 does not contain transaction-level profit. Therefore the appropriate "
            "implemented ML task is exploratory geographic customer clustering rather than profit regression."
        )

        c1, c2, c3 = st.columns(3)
        c1.metric("ML Models", "1")
        c2.metric("Model", "K-Means")
        c3.metric("Clusters", "4")

        show_table(dataset3_algorithm_table())
        clustering = run_dataset3_clustering(df)
        if clustering is not None:
            show_table(clustering["counts"])
            st.plotly_chart(
                chart_layout(
                    px.bar(
                        clustering["counts"],
                        x="Cluster",
                        y="Customers",
                        title="Olist Geographic Customer Clusters",
                    )
                ),
                use_container_width=True,
            )
            st.caption("These are exploratory geographic clusters based on ZIP-prefix information; they are not profitability or purchasing-behavior clusters.")


# PAGE 7 — ASK THE AI
# DATASET-AWARE
# ============================================================

elif page == "💬 Ask the AI":

    st.header(f"💬 Ask the AI — {dataset_name}")
    st.write(
        "The question analyzer uses the selected dataset and can route questions "
        "to business analysis, ML model comparison, XAI or customer segmentation."
    )

    question = st.text_input(
        "Enter your business question",
        placeholder="Example: Which features are important for profit prediction?",
        key="ai_question",
    )

    st.markdown(
        "**Examples:** Which category has the highest profit? • Which features are important? • "
        "Compare the ML models • Explain the best model • Show the top products • Show customer states."
    )

    if st.button("🔎 Analyze Question", type="primary", use_container_width=True):
        if not question.strip():
            st.warning("Please enter a question.")
            st.stop()

        q = question.lower()

        # XAI routing comes before the generic 'model' branch.
        is_xai = (
            "xai" in q
            or "feature importance" in q
            or "features are important" in q
            or ("feature" in q and "important" in q)
            or "explain the model" in q
            or "model explanation" in q
            or "explainability" in q
        )

        if is_xai and DATASET_CONFIG[dataset_name]["type"] == "transaction":
            ml_result = train_transaction_models(df, dataset_name)
            if ml_result is not None:
                st.subheader(f"XAI Summary — {dataset_name}")
                selected = "Proposed Hybrid"
                rf = ml_result["models"]["RF Selected"]
                gb = ml_result["models"]["Gradient Boosting"]
                pre = rf.named_steps["preprocessor"]
                names = pre.get_feature_names_out()
                values = 0.50 * rf.named_steps["model"].feature_importances_ + 0.50 * gb.named_steps["model"].feature_importances_
                xai_df = pd.DataFrame({"Feature": names, "Importance": values}).sort_values("Importance", ascending=False).head(15)
                show_table(xai_df, {"Importance": "{:.6f}"})
                st.plotly_chart(
                    chart_layout(
                        px.bar(xai_df.sort_values("Importance"), x="Importance", y="Feature", orientation="h", title="Proposed Hybrid — Feature Importance")
                    ),
                    use_container_width=True,
                )
            else:
                st.warning("The transaction model pipeline could not be trained.")

        elif (
            "compare" in q
            or "comparison" in q
            or "models" in q
            or "algorithm" in q
            or "regression" in q
            or "machine learning" in q
        ) and DATASET_CONFIG[dataset_name]["type"] == "transaction":
            ml_result = train_transaction_models(df, dataset_name)
            if ml_result is not None:
                st.subheader(f"Six-Model Comparison — {dataset_name}")
                show_table(ml_result["metrics"], {"R²": "{:.4f}", "MAE": "{:,.2f}", "RMSE": "{:,.2f}", "MAPE": "{:.2f}%", "sMAPE": "{:.2f}%", "Average Rank": "{:.3f}"})

        elif dataset_name == "Dataset 3 — Olist Customers":
            if "state" in q or "states" in q or "customer distribution" in q:
                if has_col(filtered_df, "customer_state"):
                    state = filtered_df["customer_state"].value_counts().reset_index()
                    state.columns = ["State", "Customers"]
                    show_table(state.head(20))
                    st.plotly_chart(chart_layout(px.bar(state.head(20), x="State", y="Customers", title="Customer Distribution by State")), use_container_width=True)
            elif "city" in q:
                if has_col(filtered_df, "customer_city"):
                    city = filtered_df["customer_city"].astype(str).str.title().value_counts().reset_index().head(20)
                    city.columns = ["City", "Customers"]
                    show_table(city)
            elif "cluster" in q or "segment" in q:
                clustering = run_dataset3_clustering(df)
                if clustering is not None:
                    show_table(clustering["counts"])
            else:
                st.info("For Olist, ask about customer states, cities, clusters or algorithms.")

        elif DATASET_CONFIG[dataset_name]["type"] == "transaction":
            if "category" in q and "profit" in q and has_col(filtered_df, "category"):
                result = filtered_df.groupby("category").agg(Sales=("sales", "sum"), Profit=("profit", "sum"), Quantity=("quantity", "sum")).reset_index().sort_values("Profit", ascending=False)
                show_table(result)
                st.plotly_chart(chart_layout(px.bar(result, x="category", y="Profit", title="Profit by Category")), use_container_width=True)
            elif "market" in q and "profit" in q and has_col(filtered_df, "market"):
                result = filtered_df.groupby("market").agg(Sales=("sales", "sum"), Profit=("profit", "sum")).reset_index().sort_values("Profit", ascending=False)
                show_table(result)
            elif "region" in q and has_col(filtered_df, "region"):
                metric = "Profit" if "profit" in q else "Sales"
                result = filtered_df.groupby("region").agg(Sales=("sales", "sum"), Profit=("profit", "sum")).reset_index().sort_values(metric, ascending=False)
                show_table(result)
                st.plotly_chart(chart_layout(px.bar(result, x="region", y=metric, title=f"{metric} by Region")), use_container_width=True)
            elif "discount" in q and has_col(filtered_df, "discount") and has_col(filtered_df, "profit"):
                result = filtered_df.groupby("discount").agg(Sales=("sales", "sum"), Profit=("profit", "sum"), Quantity=("quantity", "sum")).reset_index().sort_values("discount")
                show_table(result)
                st.plotly_chart(chart_layout(px.scatter(filtered_df, x="discount", y="profit", title="Observed Discount vs Profit")), use_container_width=True)
            elif "product" in q and has_col(filtered_df, "product_name"):
                result = filtered_df.groupby("product_name").agg(Sales=("sales", "sum"), Profit=("profit", "sum")).reset_index().sort_values("Profit", ascending=False).head(10)
                show_table(result)
            elif ("trend" in q or "year" in q) and has_col(filtered_df, "order_year"):
                result = filtered_df.groupby("order_year").agg(Sales=("sales", "sum"), Profit=("profit", "sum")).reset_index().sort_values("order_year")
                st.plotly_chart(chart_layout(px.line(result, x="order_year", y=["Sales", "Profit"], markers=True, title="Yearly Sales and Profit")), use_container_width=True)
                show_table(result)
            else:
                st.info("Try a question about category, market, region, discount, product, yearly trend, model comparison or XAI.")


# ============================================================
# PAGE 8 — COMPARATIVE ANALYSIS
# ============================================================

elif page == "📊 Comparative Analysis":

    st.header(f"📊 Comparative Analysis — {dataset_name}")

    if DATASET_CONFIG[dataset_name]["type"] == "transaction":
        ml_result = train_transaction_models(df, dataset_name)
        if ml_result is None:
            st.error("Comparative analysis could not be generated for the active dataset.")
        else:
            st.subheader("Six-Model Regression Comparison")
            st.write(
                "The dashboard evaluates a naive baseline, Ridge Regression, Random Forest, "
                "Random Forest Selected, Gradient Boosting and the proposed hybrid candidate."
            )
            show_table(
                ml_result["metrics"],
                {
                    "R²": "{:.4f}",
                    "MAE": "{:,.2f}",
                    "RMSE": "{:,.2f}",
                    "MAPE": "{:.2f}%",
                    "sMAPE": "{:.2f}%",
                    "Average Rank": "{:.3f}",
                },
            )

            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(
                    chart_layout(
                        px.bar(
                            ml_result["metrics"].sort_values("RMSE"),
                            x="Model",
                            y="RMSE",
                            text="RMSE",
                            title="RMSE — Lower is Better",
                        ),
                        500,
                    ),
                    use_container_width=True,
                )
            with c2:
                st.plotly_chart(
                    chart_layout(
                        px.bar(
                            ml_result["metrics"].sort_values("R²", ascending=False),
                            x="Model",
                            y="R²",
                            text="R²",
                            title="R² — Higher is Better",
                        ),
                        500,
                    ),
                    use_container_width=True,
                )

            best = ml_result["metrics"].iloc[0]
            st.success(
                f"Overall ranking leader: {best['Model']}. "
                "Ranking uses MAE, RMSE, MAPE, sMAPE and R²; RMSE is used as a practical tie-breaker."
            )
            st.warning(
                "The hybrid shown here is an implemented candidate. Do not claim that the RF+GB average itself is novel without the literature/novelty check."
            )

    else:
        st.subheader("Dataset 3 — Customer Segmentation Comparison")
        st.info(
            "Dataset 3 has no transaction-level profit target, so regression metrics such as MAE and RMSE are not appropriate. "
            "The comparative view therefore reports the implemented K-Means clustering configuration."
        )
        clustering = run_dataset3_clustering(df)
        if clustering is not None:
            show_table(clustering["counts"])
            st.plotly_chart(
                chart_layout(
                    px.bar(
                        clustering["counts"],
                        x="Cluster",
                        y="Customers",
                        title="Dataset 3 — Geographic Customer Cluster Comparison",
                    )
                ),
                use_container_width=True,
            )


# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "AI-Assisted Data Visualization for Business Decision Support Systems "
    "| Data Science and Visualization Project"
)