# AI-Assisted Data Visualization for Business Decision Support Systems

## Project Overview

This project develops an AI-assisted business analytics and decision-support platform that combines data preprocessing, exploratory data analysis, machine learning, explainable AI, interactive visualization, and business recommendations.

The system analyzes retail transaction datasets to support profit prediction and uses customer geographic information to explore customer segmentation.

## Objectives

- Analyze and visualize business data through interactive dashboards.
- Perform data cleaning, preprocessing, and feature engineering.
- Compare machine-learning models for profit prediction.
- Evaluate multiple hybrid-model candidates.
- Apply explainable AI techniques to interpret model predictions.
- Generate data-driven business insights and recommendations.
- Explore geographic customer segmentation using the available customer-location data.

## Datasets

1. **Superstore Sales:** Retail transaction data containing sales, profit, discounts, products, regions, and dates.
2. **Global Superstore:** Global retail transaction data for comparative profit analysis.
3. **Brazilian E-Commerce (Olist):** Customer location data used for geographic segmentation.

Dataset availability and permitted redistribution depend on the original sources.

## Technologies Used

- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- Streamlit
- Explainable AI and feature-importance techniques
- Git and GitHub

## Project Structure

```text
AI-Assisted-Data-Visualization/
├── Review 1/
├── Review 2/
├── Review 3/
├── dashboard/
├── datasets/
├── docs/
├── models/
├── notebooks/
├── outputs/
│   ├── cleaned/
│   ├── figures/
│   ├── model_metrics/
│   ├── reports/
│   └── results/
├── requirements/
├── src/
├── .gitignore
└── README.md
```

## Implementation Workflow

1. Dataset loading and validation
2. Data cleaning and preprocessing
3. Exploratory data analysis and visualization
4. Feature engineering and feature selection
5. Baseline machine-learning model training
6. Comparative evaluation of hybrid-model candidates
7. Model explainability and feature-importance analysis
8. Business insights and recommendations
9. Interactive dashboard development
10. Results analysis and documentation

## Running the Dashboard

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd YOUR_PROJECT_FOLDER
```

### 2. Install dependencies

```bash
pip install -r requirements/requirements.txt
```

### 3. Launch the application

```bash
streamlit run dashboard/app.py
```

The dashboard should open in your browser at the local URL displayed in the terminal.

## Results and Evaluation

The project includes saved exploratory-analysis figures, model-comparison tables, regression metrics, clustering evaluations, feature-importance results, and business-summary outputs under the `outputs/` directory.

Model performance is evaluated using suitable metrics such as MAE, RMSE, R², and clustering evaluation measures. The best-performing model is determined from the measured results rather than assumed in advance.

## Project Documentation

The `Review 1`, `Review 2`, and `Review 3` folders organize review presentations and supporting documents. The `docs/` folder contains additional project documentation, while the `outputs/` folder stores generated analytical results.

## Academic Project

**Domain:** Business Analytics and Decision Support Systems

**Project Title:** AI-Assisted Data Visualization for Business Decision Support Systems

**Institution:** KLH University

**Academic Year:** 2026–2027

## Disclaimer

This project is intended for academic and research purposes. Predictions and recommendations should be interpreted in the context of the datasets, experimental setup, and model limitations. The reported results do not imply causal relationships or guaranteed business outcomes.
