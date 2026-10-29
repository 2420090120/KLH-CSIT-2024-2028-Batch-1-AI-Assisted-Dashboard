# ============================================================
# AI-ASSISTED BUSINESS DECISION SUPPORT SYSTEM
# Project: AI-Assisted Data Visualization for Business
# Decision Support Systems
#
# Dataset: Superstore Sales Dataset
#
# Integrated Components:
# 1. Business Analytics
# 2. Data Visualization
# 3. Explainable AI (XAI)
# 4. Business Recommendations
# 5. Natural-Language Question Interface
# ============================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


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

XAI_FILE = os.path.join(
    BASE_DIR,
    "outputs",
    "results",
    "xai_feature_importance.csv"
)

RECOMMENDATION_FILE = os.path.join(
    BASE_DIR,
    "outputs",
    "results",
    "business_recommendations.csv"
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
# 2. LOAD PROJECT DATA
# ============================================================

print("=" * 75)
print("AI-ASSISTED BUSINESS DECISION SUPPORT SYSTEM")
print("=" * 75)


df = pd.read_csv(
    DATA_FILE
)


xai_df = pd.read_csv(
    XAI_FILE
)


recommendation_df = pd.read_csv(
    RECOMMENDATION_FILE
)


print("\nAll project data loaded successfully.")

print(
    "Business records:",
    len(df)
)

print(
    "XAI features:",
    len(xai_df)
)

print(
    "Recommendations:",
    len(recommendation_df)
)


# ============================================================
# 3. BUSINESS SUMMARY
# ============================================================

def show_business_summary():

    total_sales = df["sales"].sum()

    total_profit = df["profit"].sum()

    total_quantity = df["quantity"].sum()

    average_discount = df["discount"].mean()

    overall_profit_margin = (
        total_profit
        /
        total_sales
    ) * 100


    print("\n")
    print("=" * 75)
    print("BUSINESS SUMMARY")
    print("=" * 75)

    print(
        f"Total Sales      : {total_sales:,.2f}"
    )

    print(
        f"Total Profit     : {total_profit:,.2f}"
    )

    print(
        f"Total Quantity   : {total_quantity:,}"
    )

    print(
        f"Average Discount : {average_discount:.2%}"
    )

    print(
        f"Profit Margin    : {overall_profit_margin:.2f}%"
    )


# ============================================================
# 4. CATEGORY ANALYSIS
# ============================================================

def category_analysis():

    summary = (

        df.groupby(
            "category"
        )

        .agg(

            Sales=(
                "sales",
                "sum"
            ),

            Profit=(
                "profit",
                "sum"
            ),

            Quantity=(
                "quantity",
                "sum"
            ),

            Average_Discount=(
                "discount",
                "mean"
            )

        )

        .reset_index()

        .sort_values(
            "Profit",
            ascending=False
        )

    )


    print("\n")
    print("=" * 75)
    print("CATEGORY PERFORMANCE")
    print("=" * 75)

    print(
        summary.to_string(
            index=False
        )
    )


    # Visualization

    plt.figure(
        figsize=(9, 6)
    )

    plt.bar(
        summary["category"],
        summary["Profit"]
    )

    plt.xlabel(
        "Category"
    )

    plt.ylabel(
        "Total Profit"
    )

    plt.title(
        "AI Decision Support - Profit by Category"
    )

    plt.tight_layout()


    filename = os.path.join(
        FIGURE_DIR,
        "30_dss_category_profit.png"
    )

    plt.savefig(
        filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


    print(
        "\nVisualization saved:"
    )

    print(
        filename
    )


# ============================================================
# 5. REGIONAL ANALYSIS
# ============================================================

def region_analysis():

    summary = (

        df.groupby(
            "region"
        )

        .agg(

            Sales=(
                "sales",
                "sum"
            ),

            Profit=(
                "profit",
                "sum"
            ),

            Quantity=(
                "quantity",
                "sum"
            )

        )

        .reset_index()

        .sort_values(
            "Profit",
            ascending=False
        )

    )


    print("\n")
    print("=" * 75)
    print("REGIONAL PERFORMANCE")
    print("=" * 75)

    print(
        summary.to_string(
            index=False
        )
    )


    plt.figure(
        figsize=(9, 6)
    )

    plt.bar(
        summary["region"],
        summary["Profit"]
    )

    plt.xlabel(
        "Region"
    )

    plt.ylabel(
        "Total Profit"
    )

    plt.title(
        "AI Decision Support - Profit by Region"
    )

    plt.tight_layout()


    filename = os.path.join(
        FIGURE_DIR,
        "31_dss_region_profit.png"
    )

    plt.savefig(
        filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


    print(
        "\nVisualization saved:"
    )

    print(
        filename
    )


# ============================================================
# 6. SUBCATEGORY ANALYSIS
# ============================================================

def subcategory_analysis():

    summary = (

        df.groupby(
            "subcategory"
        )

        .agg(

            Sales=(
                "sales",
                "sum"
            ),

            Profit=(
                "profit",
                "sum"
            ),

            Quantity=(
                "quantity",
                "sum"
            )

        )

        .reset_index()

        .sort_values(
            "Profit"
        )

    )


    print("\n")
    print("=" * 75)
    print("SUBCATEGORY PROFITABILITY")
    print("=" * 75)

    print(
        summary.to_string(
            index=False
        )
    )


    plt.figure(
        figsize=(11, 7)
    )

    plt.barh(
        summary["subcategory"],
        summary["Profit"]
    )

    plt.xlabel(
        "Total Profit"
    )

    plt.ylabel(
        "Subcategory"
    )

    plt.title(
        "AI Decision Support - Subcategory Profitability"
    )

    plt.tight_layout()


    filename = os.path.join(
        FIGURE_DIR,
        "32_dss_subcategory_profit.png"
    )

    plt.savefig(
        filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


    print(
        "\nVisualization saved:"
    )

    print(
        filename
    )


# ============================================================
# 7. YEARLY SALES AND PROFIT TREND
# ============================================================

def sales_trend_analysis():

    summary = (

        df.groupby(
            "order_year"
        )

        .agg(

            Sales=(
                "sales",
                "sum"
            ),

            Profit=(
                "profit",
                "sum"
            )

        )

        .reset_index()

        .sort_values(
            "order_year"
        )

    )


    print("\n")
    print("=" * 75)
    print("YEARLY SALES AND PROFIT TREND")
    print("=" * 75)

    print(
        summary.to_string(
            index=False
        )
    )


    plt.figure(
        figsize=(10, 6)
    )


    plt.plot(
        summary["order_year"],
        summary["Sales"],
        marker="o",
        label="Sales"
    )


    plt.plot(
        summary["order_year"],
        summary["Profit"],
        marker="o",
        label="Profit"
    )


    plt.xlabel(
        "Year"
    )

    plt.ylabel(
        "Amount"
    )

    plt.title(
        "AI Decision Support - Yearly Business Trend"
    )

    plt.legend()

    plt.tight_layout()


    filename = os.path.join(
        FIGURE_DIR,
        "33_dss_yearly_trend.png"
    )

    plt.savefig(
        filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


    print(
        "\nVisualization saved:"
    )

    print(
        filename
    )


# ============================================================
# 8. DISCOUNT ANALYSIS
# ============================================================

def discount_analysis():

    summary = (

        df.groupby(
            "discount"
        )

        .agg(

            Sales=(
                "sales",
                "sum"
            ),

            Profit=(
                "profit",
                "sum"
            ),

            Quantity=(
                "quantity",
                "sum"
            )

        )

        .reset_index()

        .sort_values(
            "discount"
        )

    )


    print("\n")
    print("=" * 75)
    print("DISCOUNT AND PROFIT ANALYSIS")
    print("=" * 75)

    print(
        summary.to_string(
            index=False
        )
    )


    plt.figure(
        figsize=(10, 6)
    )


    plt.scatter(
        summary["discount"],
        summary["Profit"]
    )


    plt.xlabel(
        "Discount"
    )

    plt.ylabel(
        "Total Profit"
    )

    plt.title(
        "AI Decision Support - Discount vs Profit"
    )

    plt.tight_layout()


    filename = os.path.join(
        FIGURE_DIR,
        "34_dss_discount_profit.png"
    )

    plt.savefig(
        filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


    print(
        "\nVisualization saved:"
    )

    print(
        filename
    )


# ============================================================
# 9. TOP PRODUCTS
# ============================================================

def top_products():

    summary = (

        df.groupby(
            "product_name"
        )

        .agg(

            Sales=(
                "sales",
                "sum"
            ),

            Profit=(
                "profit",
                "sum"
            )

        )

        .reset_index()

        .sort_values(
            "Profit",
            ascending=False
        )

        .head(10)

    )


    print("\n")
    print("=" * 75)
    print("TOP 10 PRODUCTS BY PROFIT")
    print("=" * 75)

    print(
        summary.to_string(
            index=False
        )
    )


    plt.figure(
        figsize=(12, 7)
    )


    plt.barh(
        summary["product_name"],
        summary["Profit"]
    )


    plt.xlabel(
        "Total Profit"
    )

    plt.ylabel(
        "Product"
    )

    plt.title(
        "Top 10 Products by Profit"
    )


    plt.gca().invert_yaxis()

    plt.tight_layout()


    filename = os.path.join(
        FIGURE_DIR,
        "35_dss_top_products.png"
    )

    plt.savefig(
        filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


    print(
        "\nVisualization saved:"
    )

    print(
        filename
    )


# ============================================================
# 10. XAI FEATURE IMPORTANCE
# ============================================================

def show_xai_insights():

    print("\n")
    print("=" * 75)
    print("EXPLAINABLE AI - FEATURE IMPORTANCE")
    print("=" * 75)


    display_df = xai_df[
        [
            "rank",
            "feature",
            "importance_mean",
            "importance_std"
        ]
    ]


    print(
        display_df.to_string(
            index=False
        )
    )


    top_feature = (
        xai_df
        .sort_values(
            "importance_mean",
            ascending=False
        )
        .iloc[0]
    )


    print("\nKey XAI observation:")

    print(
        f"'{top_feature['feature']}' has the "
        f"highest permutation importance."
    )


    # Visualization

    plot_data = (
        xai_df
        .sort_values(
            "importance_mean"
        )
    )


    plt.figure(
        figsize=(10, 6)
    )


    plt.barh(
        plot_data["feature"],
        plot_data["importance_mean"]
    )


    plt.xlabel(
        "Permutation Importance"
    )

    plt.ylabel(
        "Feature"
    )

    plt.title(
        "AI Decision Support - XAI Feature Importance"
    )

    plt.tight_layout()


    filename = os.path.join(
        FIGURE_DIR,
        "36_dss_xai_feature_importance.png"
    )


    plt.savefig(
        filename,
        dpi=300,
        bbox_inches="tight"
    )


    plt.close()


    print(
        "\nXAI visualization saved:"
    )

    print(
        filename
    )


# ============================================================
# 11. BUSINESS RECOMMENDATIONS
# ============================================================

def show_recommendations():

    print("\n")
    print("=" * 75)
    print("AI-ASSISTED BUSINESS RECOMMENDATIONS")
    print("=" * 75)


    for index, row in recommendation_df.iterrows():

        print(
            f"\nRecommendation {index + 1}"
        )

        print(
            f"Area: {row['Area']}"
        )

        print(
            f"Finding: {row['Finding']}"
        )

        print(
            f"Recommendation: "
            f"{row['Recommendation']}"
        )


# ============================================================
# 12. BUSINESS QUESTION: SALES
# ============================================================

def sales_analysis():

    summary = (

        df.groupby(
            "category"
        )

        .agg(
            Sales=("sales", "sum")
        )

        .reset_index()

        .sort_values(
            "Sales",
            ascending=False
        )

    )


    print("\n")
    print("=" * 75)
    print("SALES BY CATEGORY")
    print("=" * 75)

    print(
        summary.to_string(
            index=False
        )
    )


    plt.figure(
        figsize=(9, 6)
    )


    plt.bar(
        summary["category"],
        summary["Sales"]
    )


    plt.xlabel(
        "Category"
    )

    plt.ylabel(
        "Total Sales"
    )

    plt.title(
        "AI Decision Support - Sales by Category"
    )

    plt.tight_layout()


    filename = os.path.join(
        FIGURE_DIR,
        "37_dss_sales_by_category.png"
    )


    plt.savefig(
        filename,
        dpi=300,
        bbox_inches="tight"
    )


    plt.close()


    print(
        "\nVisualization saved:"
    )

    print(
        filename
    )


# ============================================================
# 13. NATURAL-LANGUAGE QUESTION ANALYZER
# ============================================================

def analyze_question(question):

    question = (
        question
        .lower()
        .strip()
    )


    # --------------------------------------------------------
    # XAI / FEATURE IMPORTANCE
    # --------------------------------------------------------

    if (
        "xai" in question
        or "feature importance" in question
        or "features are important" in question
        or "which features" in question
        or (
            "feature" in question
            and "important" in question
        )
        or "explain the model" in question
        or "model explanation" in question
    ):

        show_xai_insights()

        return


    # --------------------------------------------------------
    # BUSINESS RECOMMENDATIONS
    # --------------------------------------------------------

    if (
        "recommendation" in question
        or "recommendations" in question
        or "advice" in question
        or "what should we do" in question
        or "business advice" in question
    ):

        show_recommendations()

        return


    # --------------------------------------------------------
    # CATEGORY + PROFIT
    # --------------------------------------------------------

    if (
        "category" in question
        and "profit" in question
    ):

        category_analysis()

        return


    # --------------------------------------------------------
    # CATEGORY + SALES
    # --------------------------------------------------------

    if (
        "category" in question
        and "sales" in question
    ):

        sales_analysis()

        return


    # --------------------------------------------------------
    # REGION + PROFIT/SALES
    # --------------------------------------------------------

    if (
        "region" in question
        and (
            "profit" in question
            or "sales" in question
        )
    ):

        region_analysis()

        return


    # --------------------------------------------------------
    # SUBCATEGORY
    # --------------------------------------------------------

    if (
        "subcategory" in question
        or "sub-category" in question
        or "sub category" in question
    ):

        subcategory_analysis()

        return


    # --------------------------------------------------------
    # DISCOUNT
    # --------------------------------------------------------

    if "discount" in question:

        discount_analysis()

        return


    # --------------------------------------------------------
    # PRODUCT
    # --------------------------------------------------------

    if (
        "product" in question
        or "products" in question
    ):

        top_products()

        return


    # --------------------------------------------------------
    # TREND / YEAR
    # --------------------------------------------------------

    if (
        "trend" in question
        or "yearly" in question
        or "year" in question
        or "over time" in question
        or "time trend" in question
    ):

        sales_trend_analysis()

        return


    # --------------------------------------------------------
    # BUSINESS SUMMARY
    # --------------------------------------------------------

    if (
        "summary" in question
        or "overview" in question
        or "overall" in question
        or "total sales" in question
        or "total profit" in question
    ):

        show_business_summary()

        return


    # --------------------------------------------------------
    # UNKNOWN QUESTION
    # --------------------------------------------------------

    print("\nI could not identify the requested analysis.")

    print("\nTry questions such as:")

    print(
        "1. Which category has the highest profit?"
    )

    print(
        "2. Which category has the highest sales?"
    )

    print(
        "3. Which region has the highest profit?"
    )

    print(
        "4. Show subcategory profitability."
    )

    print(
        "5. Show yearly sales trend."
    )

    print(
        "6. Does discount affect profit?"
    )

    print(
        "7. Show top products."
    )

    print(
        "8. Which features are important?"
    )

    print(
        "9. Give business recommendations."
    )


# ============================================================
# 14. SAVE DSS SUMMARY
# ============================================================

def save_summary():

    total_sales = df[
        "sales"
    ].sum()

    total_profit = df[
        "profit"
    ].sum()

    total_quantity = df[
        "quantity"
    ].sum()

    average_discount = df[
        "discount"
    ].mean()

    profit_margin = (
        total_profit
        /
        total_sales
    ) * 100


    summary_df = pd.DataFrame({

        "metric": [

            "Total Sales",

            "Total Profit",

            "Total Quantity",

            "Average Discount",

            "Overall Profit Margin"

        ],

        "value": [

            total_sales,

            total_profit,

            total_quantity,

            average_discount,

            profit_margin

        ]

    })


    output_file = os.path.join(

        OUTPUT_DIR,

        "dss_business_summary.csv"

    )


    summary_df.to_csv(

        output_file,

        index=False

    )


    print(
        "\nDSS summary saved to:"
    )

    print(
        output_file
    )


# ============================================================
# 15. MAIN MENU
# ============================================================

def main():

    save_summary()


    while True:

        print("\n")
        print("=" * 75)
        print("AI DECISION SUPPORT MENU")
        print("=" * 75)

        print(
            "1. Business Summary"
        )

        print(
            "2. Category Analysis"
        )

        print(
            "3. Regional Analysis"
        )

        print(
            "4. Subcategory Analysis"
        )

        print(
            "5. Yearly Sales Trend"
        )

        print(
            "6. Discount Analysis"
        )

        print(
            "7. Top Products"
        )

        print(
            "8. XAI Feature Importance"
        )

        print(
            "9. Business Recommendations"
        )

        print(
            "10. Ask Natural-Language Question"
        )

        print(
            "0. Exit"
        )


        choice = input(
            "\nEnter your choice: "
        ).strip()


        # ----------------------------------------------------
        # MENU OPTIONS
        # ----------------------------------------------------

        if choice == "1":

            show_business_summary()


        elif choice == "2":

            category_analysis()


        elif choice == "3":

            region_analysis()


        elif choice == "4":

            subcategory_analysis()


        elif choice == "5":

            sales_trend_analysis()


        elif choice == "6":

            discount_analysis()


        elif choice == "7":

            top_products()


        elif choice == "8":

            show_xai_insights()


        elif choice == "9":

            show_recommendations()


        elif choice == "10":

            question = input(
                "\nEnter your business question: "
            )

            analyze_question(
                question
            )


        elif choice == "0":

            print(
                "\nExiting AI Decision Support System."
            )

            break


        else:

            print(
                "\nInvalid choice."
            )

            print(
                "Please enter a number from 0 to 10."
            )


# ============================================================
# 16. PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()