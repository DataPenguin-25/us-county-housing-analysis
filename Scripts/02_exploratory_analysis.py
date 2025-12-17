"""
02_exploratory_analysis.py
US County-Level Housing & Employment Analysis

This script performs exploratory data analysis on ACS housing data:
- Data cleaning and validation
- Correlation analysis
- Scatter plots and visualizations
- Rent tier categorization

Author: Joseph Adamski
Project: CareerFoundry Data Analytics - Achievement 6
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set visualization style
sns.set(style="whitegrid")

# =============================================================================
# LOAD DATA
# =============================================================================

df = pd.read_csv("../Data/Prepared/acs_2022_county_data_clean.csv")

print("Original Data Shape:", df.shape)

# =============================================================================
# DATA CLEANING
# Convert columns to numeric and remove invalid values
# =============================================================================

num_cols = [
    "median_gross_rent",
    "median_household_income",
    "median_home_value",
    "total_housing_units",
    "vacant_housing_units",
    "rental_vacancy_rate"
]

df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce")

# Remove corrupted and impossible values
# Rent: $0-$10,000, Income: $0-$300,000, Home Value: $0-$3,000,000
df = df[
    (df["median_gross_rent"] > 0) &
    (df["median_gross_rent"] < 10000) &
    (df["median_household_income"] > 0) &
    (df["median_household_income"] < 300000) &
    (df["median_home_value"] > 0) &
    (df["median_home_value"] < 3000000)
].copy()

print("Cleaned Data Shape:", df.shape)

# =============================================================================
# FEATURE ENGINEERING
# Create rent-to-income ratio (annual rent / annual income)
# =============================================================================

df["rent_to_income_ratio"] = (df["median_gross_rent"] * 12) / df["median_household_income"]

# =============================================================================
# DESCRIPTIVE STATISTICS
# =============================================================================

print("\nDescriptive Statistics:")
print(df[['median_gross_rent', 'median_household_income', 'median_home_value']].describe())

# =============================================================================
# CORRELATION ANALYSIS
# =============================================================================

# Select columns for correlation
eda_df = df[[
    "median_gross_rent",
    "median_household_income",
    "median_home_value",
    "rental_vacancy_rate",
    "rent_to_income_ratio"
]]

# Calculate correlation matrix
corr = eda_df.corr()

print("\nCorrelation Matrix:")
print(corr)

# =============================================================================
# VISUALIZATION 1: Correlation Heatmap
# =============================================================================

plt.figure(figsize=(10, 7))
sns.heatmap(corr, annot=True, cmap="coolwarm", linewidths=0.5)
plt.title("Correlation Heatmap: Housing Variables")
plt.tight_layout()
plt.savefig("../Outputs/correlation_heatmap.png", dpi=150)
plt.show()

# =============================================================================
# VISUALIZATION 2: Income vs Rent Scatter Plot
# =============================================================================

plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=df,
    x="median_household_income",
    y="median_gross_rent",
    alpha=0.4
)
plt.title("Median Household Income vs Median Gross Rent")
plt.xlabel("Median Household Income ($)")
plt.ylabel("Median Gross Rent ($)")
plt.tight_layout()
plt.savefig("../Outputs/income_vs_rent_scatter.png", dpi=150)
plt.show()

# =============================================================================
# VISUALIZATION 3: Home Value vs Rent Scatter Plot
# =============================================================================

plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=df,
    x="median_home_value",
    y="median_gross_rent",
    alpha=0.4
)
plt.title("Median Home Value vs Median Gross Rent")
plt.xlabel("Median Home Value ($)")
plt.ylabel("Median Gross Rent ($)")
plt.tight_layout()
plt.savefig("../Outputs/home_value_vs_rent_scatter.png", dpi=150)
plt.show()

# =============================================================================
# VISUALIZATION 4: Pair Plot
# =============================================================================

sns.pairplot(eda_df, diag_kind="hist", plot_kws={"alpha": 0.4})
plt.savefig("../Outputs/pairplot.png", dpi=150)
plt.show()

# =============================================================================
# CATEGORICAL ANALYSIS: Rent Tiers
# =============================================================================

def rent_tier(x):
    """Categorize counties by median rent level"""
    if x < 800:
        return "Low rent"
    elif x < 1500:
        return "Medium rent"
    else:
        return "High rent"

df["rent_tier"] = df["median_gross_rent"].apply(rent_tier)

print("\nRent Tier Distribution:")
print(df["rent_tier"].value_counts())

# =============================================================================
# VISUALIZATION 5: Rent Burden by Tier
# =============================================================================

plt.figure(figsize=(8, 6))
sns.boxplot(
    data=df,
    x="rent_tier",
    y="rent_to_income_ratio"
)
plt.title("Rent-to-Income Ratio by Rent Tier")
plt.xlabel("Rent Tier")
plt.ylabel("Rent-to-Income Ratio")
plt.tight_layout()
plt.savefig("../Outputs/rent_burden_by_tier.png", dpi=150)
plt.show()

# =============================================================================
# KEY FINDINGS
# =============================================================================

print("\n" + "="*60)
print("KEY CORRELATION FINDINGS")
print("="*60)
print(f"Rent vs Income:     r = {corr.loc['median_gross_rent', 'median_household_income']:.2f}")
print(f"Rent vs Home Value: r = {corr.loc['median_gross_rent', 'median_home_value']:.2f}")
print(f"Rent vs Vacancy:    r = {corr.loc['median_gross_rent', 'rental_vacancy_rate']:.2f}")

print("\nExploratory analysis complete!")
