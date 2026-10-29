# ============================================================
# AI-ASSISTED BUSINESS RECOMMENDATIONS
# Project: AI-Assisted Data Visualization for Business
# Decision Support Systems
#
# Dataset: Superstore Sales Dataset
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
print("AI-ASSISTED BUSINESS RECOMMENDATION ENGINE")
print("=" * 70)


# ============================================================
# 3. LOAD DATA
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
# 4. LOAD XAI RESULTS
# ============================================================

xai = pd.read_csv(
    XAI_FILE
)

print("\nXAI results loaded successfully.")


# ============================================================
# 5. BASIC BUSINESS METRICS
# ============================================================

total_sales = df["sales"].sum()

total_profit = df["profit"].sum()

total_quantity = df["quantity"].sum()

average_discount = df["discount"].mean()

average_profit_margin = df[
    "profit_margin"
].mean()


print("\n")
print("=" * 70)
print("OVERALL BUSINESS METRICS")
print("=" * 70)

print(
    "Total Sales:",
    round(total_sales, 2)
)

print(
    "Total Profit:",
    round(total_profit, 2)
)

print(
    "Total Quantity:",
    total_quantity
)

print(
    "Average Discount:",
    round(
        average_discount,
        4
    )
)

print(
    "Average Profit Margin:",
    round(
        average_profit_margin,
        4
    )
)


# ============================================================
# 6. CATEGORY ANALYSIS
# ============================================================

category_summary = (

    df.groupby(
        "category"
    )

    .agg(

        sales=("sales", "sum"),

        profit=("profit", "sum"),

        quantity=("quantity", "sum"),

        average_discount=(
            "discount",
            "mean"
        )

    )

    .reset_index()

)


category_summary[
    "profit_margin"
] = (

    category_summary[
        "profit"
    ]

    /

    category_summary[
        "sales"
    ]

)


category_summary = category_summary.sort_values(
    "profit",
    ascending=False
)


# ============================================================
# 7. REGION ANALYSIS
# ============================================================

region_summary = (

    df.groupby(
        "region"
    )

    .agg(

        sales=("sales", "sum"),

        profit=("profit", "sum"),

        quantity=("quantity", "sum"),

        average_discount=(
            "discount",
            "mean"
        )

    )

    .reset_index()

)


region_summary[
    "profit_margin"
] = (

    region_summary[
        "profit"
    ]

    /

    region_summary[
        "sales"
    ]

)


region_summary = region_summary.sort_values(
    "profit",
    ascending=False
)


# ============================================================
# 8. SUBCATEGORY ANALYSIS
# ============================================================

subcategory_summary = (

    df.groupby(
        "subcategory"
    )

    .agg(

        sales=("sales", "sum"),

        profit=("profit", "sum"),

        quantity=("quantity", "sum"),

        average_discount=(
            "discount",
            "mean"
        )

    )

    .reset_index()

)


subcategory_summary[
    "profit_margin"
] = (

    subcategory_summary[
        "profit"
    ]

    /

    subcategory_summary[
        "sales"
    ]

)


subcategory_summary = subcategory_summary.sort_values(
    "profit",
    ascending=False
)


# ============================================================
# 9. LOSS-MAKING SUBCATEGORIES
# ============================================================

loss_subcategories = (

    subcategory_summary[
        subcategory_summary[
            "profit"
        ] < 0
    ]

    .sort_values(
        "profit"
    )

)


# ============================================================
# 10. HIGH-DISCOUNT ANALYSIS
# ============================================================

# Calculate the average profit for different
# discount ranges.

df["discount_group"] = pd.cut(

    df["discount"],

    bins=[
        -0.01,
        0.10,
        0.20,
        0.40,
        0.60,
        1.00
    ],

    labels=[
        "0-10%",
        "10-20%",
        "20-40%",
        "40-60%",
        "60%+"
    ]

)


discount_summary = (

    df.groupby(
        "discount_group",
        observed=False
    )

    .agg(

        sales=("sales", "sum"),

        profit=("profit", "sum"),

        average_profit=(
            "profit",
            "mean"
        ),

        order_count=(
            "profit",
            "count"
        )

    )

    .reset_index()

)


# ============================================================
# 11. MONTHLY ANALYSIS
# ============================================================

monthly_summary = (

    df.groupby(
        "order_month"
    )

    .agg(

        sales=("sales", "sum"),

        profit=("profit", "sum")

    )

    .reset_index()

    .sort_values(
        "order_month"
    )

)


# ============================================================
# 12. STATE ANALYSIS
# ============================================================

state_summary = (

    df.groupby(
        "stateprovince"
    )

    .agg(

        sales=("sales", "sum"),

        profit=("profit", "sum")

    )

    .reset_index()

)


state_summary = state_summary.sort_values(
    "profit",
    ascending=False
)


# ============================================================
# 13. PRODUCT ANALYSIS
# ============================================================

product_summary = (

    df.groupby(
        "product_name"
    )

    .agg(

        sales=("sales", "sum"),

        profit=("profit", "sum"),

        quantity=("quantity", "sum"),

        average_discount=(
            "discount",
            "mean"
        )

    )

    .reset_index()

)


# ============================================================
# 14. GENERATE RECOMMENDATIONS
# ============================================================

recommendations = []


# ------------------------------------------------------------
# Recommendation 1: Top category
# ------------------------------------------------------------

top_category = category_summary.iloc[0]

recommendations.append({

    "Area":
        "Category Performance",

    "Finding":
        (
            f"{top_category['category']} "
            f"has the highest total profit of "
            f"{top_category['profit']:.2f}."
        ),

    "Recommendation":
        (
            f"Analyze the successful products and "
            f"pricing patterns within the "
            f"{top_category['category']} category "
            f"and consider maintaining or expanding "
            f"its successful business practices."
        )

})


# ------------------------------------------------------------
# Recommendation 2: Lowest-profit category
# ------------------------------------------------------------

lowest_category = category_summary.iloc[-1]

recommendations.append({

    "Area":
        "Category Risk",

    "Finding":
        (
            f"{lowest_category['category']} "
            f"has the lowest total profit of "
            f"{lowest_category['profit']:.2f}."
        ),

    "Recommendation":
        (
            f"Investigate pricing, discount levels, "
            f"product mix, and operating costs within "
            f"the {lowest_category['category']} category "
            f"before making pricing or inventory decisions."
        )

})


# ------------------------------------------------------------
# Recommendation 3: Top region
# ------------------------------------------------------------

top_region = region_summary.iloc[0]

recommendations.append({

    "Area":
        "Regional Performance",

    "Finding":
        (
            f"{top_region['region']} region has the "
            f"highest total profit of "
            f"{top_region['profit']:.2f}."
        ),

    "Recommendation":
        (
            f"Examine the products, customers, and "
            f"sales patterns contributing to the "
            f"{top_region['region']} region's performance "
            f"for potential practices that can be "
            f"evaluated in other regions."
        )

})


# ------------------------------------------------------------
# Recommendation 4: Loss-making subcategories
# ------------------------------------------------------------

if len(loss_subcategories) > 0:

    loss_names = loss_subcategories[
        "subcategory"
    ].head(5).tolist()

    recommendations.append({

        "Area":
            "Loss-Making Subcategories",

        "Finding":
            (
                "The following subcategories have "
                "negative total profit: "
                + ", ".join(loss_names)
                + "."
            ),

        "Recommendation":
            (
                "Review pricing, discounting, product "
                "costs, and sales volume for these "
                "subcategories before increasing their "
                "sales or inventory."
            )

    })


# ------------------------------------------------------------
# Recommendation 5: Discount analysis
# ------------------------------------------------------------

highest_discount_group = (
    discount_summary
    .sort_values(
        "average_profit"
    )
    .iloc[0]
)


recommendations.append({

    "Area":
        "Discount Strategy",

    "Finding":
        (
            f"The {highest_discount_group['discount_group']} "
            f"discount range has the lowest average "
            f"profit per transaction at "
            f"{highest_discount_group['average_profit']:.2f}."
        ),

    "Recommendation":
        (
            "Review high-discount transactions carefully "
            "and evaluate whether discounts are generating "
            "sufficient sales volume to justify their "
            "effect on profitability."
        )

})


# ------------------------------------------------------------
# Recommendation 6: XAI finding
# ------------------------------------------------------------

top_xai_feature = xai.iloc[0]

recommendations.append({

    "Area":
        "AI Model Insight",

    "Finding":
        (
            f"The XAI analysis identified "
            f"{top_xai_feature['feature']} as the most "
            f"important feature according to permutation "
            f"importance."
        ),

    "Recommendation":
        (
            f"Include {top_xai_feature['feature']} prominently "
            "in business dashboards and decision-support "
            "analysis because the trained model relies "
            "strongly on this feature when predicting profit."
        )

})


# ============================================================
# 15. CREATE RECOMMENDATION DATAFRAME
# ============================================================

recommendation_df = pd.DataFrame(
    recommendations
)


# ============================================================
# 16. DISPLAY RECOMMENDATIONS
# ============================================================

print("\n")
print("=" * 70)
print("AI-ASSISTED BUSINESS RECOMMENDATIONS")
print("=" * 70)


for index, row in recommendation_df.iterrows():

    print(
        f"\nRecommendation {index + 1}"
    )

    print(
        "Area:",
        row["Area"]
    )

    print(
        "Finding:",
        row["Finding"]
    )

    print(
        "Recommendation:",
        row["Recommendation"]
    )


# ============================================================
# 17. SAVE RECOMMENDATIONS
# ============================================================

recommendation_file = os.path.join(

    OUTPUT_DIR,

    "business_recommendations.csv"

)


recommendation_df.to_csv(

    recommendation_file,

    index=False

)


print("\nRecommendations saved to:")

print(
    recommendation_file
)


# ============================================================
# 18. SAVE BUSINESS SUMMARIES
# ============================================================

category_file = os.path.join(

    OUTPUT_DIR,

    "category_business_summary.csv"

)

region_file = os.path.join(

    OUTPUT_DIR,

    "region_business_summary.csv"

)

subcategory_file = os.path.join(

    OUTPUT_DIR,

    "subcategory_business_summary.csv"

)

discount_file = os.path.join(

    OUTPUT_DIR,

    "discount_business_summary.csv"

)


category_summary.to_csv(
    category_file,
    index=False
)

region_summary.to_csv(
    region_file,
    index=False
)

subcategory_summary.to_csv(
    subcategory_file,
    index=False
)

discount_summary.to_csv(
    discount_file,
    index=False
)


print("\nBusiness summary files saved.")


# ============================================================
# 19. CREATE CATEGORY PROFIT CHART
# ============================================================

category_plot = category_summary.sort_values(
    "profit"
)


plt.figure(
    figsize=(9, 6)
)


plt.bar(

    category_plot[
        "category"
    ],

    category_plot[
        "profit"
    ]

)


plt.xlabel(
    "Category"
)

plt.ylabel(
    "Total Profit"
)

plt.title(
    "Profit by Category"
)


plt.tight_layout()


category_figure = os.path.join(

    FIGURE_DIR,

    "27_business_profit_by_category.png"

)


plt.savefig(

    category_figure,

    dpi=300,

    bbox_inches="tight"

)


plt.close()


print(
    "Category profit chart saved to:"
)

print(
    category_figure
)


# ============================================================
# 20. CREATE DISCOUNT VS PROFIT CHART
# ============================================================

plt.figure(
    figsize=(9, 6)
)


plt.bar(

    discount_summary[
        "discount_group"
    ].astype(str),

    discount_summary[
        "average_profit"
    ]

)


plt.xlabel(
    "Discount Range"
)

plt.ylabel(
    "Average Profit per Transaction"
)

plt.title(
    "Discount Level vs Average Profit"
)


plt.tight_layout()


discount_figure = os.path.join(

    FIGURE_DIR,

    "28_discount_vs_profit.png"

)


plt.savefig(

    discount_figure,

    dpi=300,

    bbox_inches="tight"

)


plt.close()


print(
    "Discount-profit chart saved to:"
)

print(
    discount_figure
)


# ============================================================
# 21. CREATE REGIONAL PROFIT CHART
# ============================================================

region_plot = region_summary.sort_values(
    "profit"
)


plt.figure(
    figsize=(9, 6)
)


plt.bar(

    region_plot[
        "region"
    ],

    region_plot[
        "profit"
    ]

)


plt.xlabel(
    "Region"
)

plt.ylabel(
    "Total Profit"
)

plt.title(
    "Profit by Region"
)


plt.tight_layout()


region_figure = os.path.join(

    FIGURE_DIR,

    "29_business_profit_by_region.png"

)


plt.savefig(

    region_figure,

    dpi=300,

    bbox_inches="tight"

)


plt.close()


print(
    "Regional profit chart saved to:"
)

print(
    region_figure
)


# ============================================================
# 22. FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("BUSINESS RECOMMENDATION ENGINE COMPLETE")
print("=" * 70)

print(
    "Recommendations generated:",
    len(recommendation_df)
)

print(
    "Categories analyzed:",
    len(category_summary)
)

print(
    "Regions analyzed:",
    len(region_summary)
)

print(
    "Subcategories analyzed:",
    len(subcategory_summary)
)


print("\nGenerated files:")

print(
    " - business_recommendations.csv"
)

print(
    " - category_business_summary.csv"
)

print(
    " - region_business_summary.csv"
)

print(
    " - subcategory_business_summary.csv"
)

print(
    " - discount_business_summary.csv"
)

print(
    " - 27_business_profit_by_category.png"
)

print(
    " - 28_discount_vs_profit.png"
)

print(
    " - 29_business_profit_by_region.png"
)


print("\nNext stage:")
print(
    "Integrate AI recommendations with the AI visualization engine."
)


print("=" * 70)