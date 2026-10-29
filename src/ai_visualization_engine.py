import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# AI-ASSISTED DATA VISUALIZATION ENGINE
# Prototype - Business Decision Support
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "outputs" / "cleaned" / "superstore_cleaned.csv"
FIG_DIR = BASE_DIR / "outputs" / "figures"

FIG_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# LOAD CLEAN DATA
# ------------------------------------------------------------

df = pd.read_csv(DATA_FILE)

print("=" * 70)
print("AI-ASSISTED DATA VISUALIZATION ENGINE")
print("=" * 70)

print("\nDataset loaded successfully")
print("Rows:", len(df))
print("Columns:", len(df.columns))


# ------------------------------------------------------------
# BUSINESS QUESTION PROCESSOR
# ------------------------------------------------------------

def analyze_question(question):

    q = question.lower().strip()

    print("\n" + "-" * 70)
    print("USER QUESTION")
    print(question)
    print("-" * 70)

    # ========================================================
    # 1. PROFIT BY CATEGORY
    # ========================================================

    if "profit" in q and "category" in q:

        result = (
            df.groupby("category")["profit"]
            .sum()
            .sort_values(ascending=False)
        )

        best = result.index[0]
        value = result.iloc[0]

        plt.figure(figsize=(9, 5))
        result.sort_values().plot(kind="barh")
        plt.title("Profit by Product Category")
        plt.xlabel("Total Profit")
        plt.ylabel("Category")
        plt.tight_layout()

        output = FIG_DIR / "ai_profit_by_category.png"
        plt.savefig(output, dpi=150)
        plt.close()

        print("\nANALYSIS TYPE: Profit by Category")
        print("\nResult:")
        print(result)

        print("\nBUSINESS INSIGHT:")
        print(
            f"{best} is the most profitable category "
            f"with total profit of {value:,.2f}."
        )

        print("\nVISUALIZATION:")
        print(output)

    # ========================================================
    # 2. SALES BY CATEGORY
    # ========================================================

    elif "sales" in q and "category" in q:

        result = (
            df.groupby("category")["sales"]
            .sum()
            .sort_values(ascending=False)
        )

        best = result.index[0]
        value = result.iloc[0]

        plt.figure(figsize=(9, 5))
        result.sort_values().plot(kind="barh")
        plt.title("Sales by Product Category")
        plt.xlabel("Total Sales")
        plt.ylabel("Category")
        plt.tight_layout()

        output = FIG_DIR / "ai_sales_by_category.png"
        plt.savefig(output, dpi=150)
        plt.close()

        print("\nANALYSIS TYPE: Sales by Category")
        print("\nResult:")
        print(result)

        print("\nBUSINESS INSIGHT:")
        print(
            f"{best} generates the highest sales "
            f"with total sales of {value:,.2f}."
        )

        print("\nVISUALIZATION:")
        print(output)

    # ========================================================
    # 3. PROFIT BY REGION
    # ========================================================

    elif "profit" in q and "region" in q:

        result = (
            df.groupby("region")["profit"]
            .sum()
            .sort_values(ascending=False)
        )

        best = result.index[0]
        value = result.iloc[0]

        plt.figure(figsize=(9, 5))
        result.sort_values().plot(kind="barh")
        plt.title("Profit by Region")
        plt.xlabel("Total Profit")
        plt.ylabel("Region")
        plt.tight_layout()

        output = FIG_DIR / "ai_profit_by_region.png"
        plt.savefig(output, dpi=150)
        plt.close()

        print("\nANALYSIS TYPE: Profit by Region")
        print("\nResult:")
        print(result)

        print("\nBUSINESS INSIGHT:")
        print(
            f"{best} is the highest-profit region "
            f"with total profit of {value:,.2f}."
        )

        print("\nVISUALIZATION:")
        print(output)

    # ========================================================
    # 4. SALES BY REGION
    # ========================================================

    elif "sales" in q and "region" in q:

        result = (
            df.groupby("region")["sales"]
            .sum()
            .sort_values(ascending=False)
        )

        best = result.index[0]
        value = result.iloc[0]

        plt.figure(figsize=(9, 5))
        result.sort_values().plot(kind="barh")
        plt.title("Sales by Region")
        plt.xlabel("Total Sales")
        plt.ylabel("Region")
        plt.tight_layout()

        output = FIG_DIR / "ai_sales_by_region.png"
        plt.savefig(output, dpi=150)
        plt.close()

        print("\nANALYSIS TYPE: Sales by Region")
        print("\nResult:")
        print(result)

        print("\nBUSINESS INSIGHT:")
        print(
            f"{best} generates the highest regional sales "
            f"with total sales of {value:,.2f}."
        )

        print("\nVISUALIZATION:")
        print(output)

    # ========================================================
    # 5. TOP PRODUCTS
    # ========================================================

    elif "top" in q and "product" in q:

        result = (
            df.groupby("product_name")["sales"]
            .sum()
            .sort_values(ascending=False)
            .head(10)
        )

        plt.figure(figsize=(10, 6))
        result.sort_values().plot(kind="barh")
        plt.title("Top 10 Products by Sales")
        plt.xlabel("Total Sales")
        plt.ylabel("Product")
        plt.tight_layout()

        output = FIG_DIR / "ai_top_products.png"
        plt.savefig(output, dpi=150)
        plt.close()

        print("\nANALYSIS TYPE: Top Products by Sales")
        print("\nResult:")
        print(result)

        print("\nBUSINESS INSIGHT:")
        print(
            f"The highest-selling product is "
            f"'{result.index[0]}' with sales of "
            f"{result.iloc[0]:,.2f}."
        )

        print("\nVISUALIZATION:")
        print(output)

    # ========================================================
    # 6. YEARLY SALES TREND
    # ========================================================

    elif "sales" in q and ("year" in q or "trend" in q):

        result = (
            df.groupby("order_year")["sales"]
            .sum()
        )

        best_year = result.idxmax()

        plt.figure(figsize=(9, 5))
        result.plot(kind="line", marker="o")
        plt.title("Yearly Sales Trend")
        plt.xlabel("Year")
        plt.ylabel("Total Sales")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        output = FIG_DIR / "ai_yearly_sales_trend.png"
        plt.savefig(output, dpi=150)
        plt.close()

        print("\nANALYSIS TYPE: Yearly Sales Trend")
        print("\nResult:")
        print(result)

        print("\nBUSINESS INSIGHT:")
        print(
            f"The highest sales occurred in {best_year} "
            f"with total sales of {result.loc[best_year]:,.2f}."
        )

        print("\nVISUALIZATION:")
        print(output)

    else:

        print("\nSorry, this prototype does not recognize that question yet.")
        print("\nSupported questions include:")
        print("- Which category has the highest profit?")
        print("- Which category has the highest sales?")
        print("- Which region has the highest profit?")
        print("- Which region has the highest sales?")
        print("- What are the top products by sales?")
        print("- What is the yearly sales trend?")


# ------------------------------------------------------------
# TEST QUESTIONS
# ------------------------------------------------------------

questions = [
    "Which category has the highest profit?",
    "Which region has the highest sales?",
    "What are the top products by sales?"
]

for question in questions:
    analyze_question(question)

print("\n" + "=" * 70)
print("AI VISUALIZATION ENGINE TEST COMPLETE")
print("=" * 70)
