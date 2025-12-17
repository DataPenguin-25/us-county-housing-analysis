"""
05_clustering_analysis.py
US County-Level Housing & Employment Analysis

This script applies K-Means clustering to identify distinct
housing market segments across US counties.

Author: Joseph Adamski
Project: CareerFoundry Data Analytics - Achievement 6
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Set visualization style
sns.set(style="whitegrid")

# =============================================================================
# LOAD DATA
# =============================================================================

df = pd.read_csv("../Data/Prepared/acs_2022_county_data_clean.csv")

print("Original Data Shape:", df.shape)

# =============================================================================
# DATA CLEANING
# Replace sentinel values and drop missing data
# =============================================================================

# ACS uses -666666666 as a sentinel for missing values
df = df.replace(-666666666, np.nan)

# Drop rows with missing values in key columns
df = df.dropna(subset=[
    'median_gross_rent',
    'median_household_income',
    'median_home_value',
    'rental_vacancy_rate'
])

print("Cleaned Data Shape:", df.shape)

# =============================================================================
# SELECT FEATURES FOR CLUSTERING
# =============================================================================

# Select numeric columns for clustering
numeric_cols = df.columns[df.dtypes != 'object'].tolist()

# Remove ID columns that should not be used for clustering
cols_to_remove = ["fips", "state", "county"]
for col in cols_to_remove:
    if col in numeric_cols:
        numeric_cols.remove(col)

print("Features for clustering:", numeric_cols)

# =============================================================================
# STANDARDIZE DATA
# K-Means requires standardized features for equal weighting
# =============================================================================

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[numeric_cols])

scaled_df = pd.DataFrame(X_scaled, columns=numeric_cols)

print("\nScaled data statistics (should be mean≈0, std≈1):")
print(scaled_df.describe().loc[['mean', 'std']])

# =============================================================================
# ELBOW METHOD: Find Optimal Number of Clusters
# =============================================================================

inertia = []
K = range(1, 11)

for k in K:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(scaled_df)
    inertia.append(km.inertia_)

# Plot elbow curve
plt.figure(figsize=(7, 5))
plt.plot(K, inertia, marker="o")
plt.xticks(K)
plt.xlabel("Number of clusters (k)")
plt.ylabel("Inertia (within-cluster sum of squares)")
plt.title("Elbow Curve for K-Means Clustering")
plt.tight_layout()
plt.savefig("../Outputs/elbow_curve.png", dpi=150)
plt.show()

print("\nElbow curve suggests k=3 is optimal (diminishing returns after)")

# =============================================================================
# FIT K-MEANS WITH K=3
# =============================================================================

k = 3
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
df["cluster"] = kmeans.fit_predict(scaled_df)

print("\nCluster Distribution:")
print(df["cluster"].value_counts().sort_index())

# =============================================================================
# VISUALIZATION: Cluster Scatter Plots
# =============================================================================

def plot_clusters(x_col, y_col, title_suffix=""):
    """Create scatter plot colored by cluster"""
    if x_col not in df.columns or y_col not in df.columns:
        print(f"Skipping: {x_col} or {y_col} not in dataframe")
        return
    
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=df,
        x=x_col,
        y=y_col,
        hue="cluster",
        palette="viridis",
        alpha=0.7
    )
    plt.title(f"County Clusters: {y_col} vs {x_col}")
    plt.legend(title="Cluster")
    plt.tight_layout()
    plt.savefig(f"../Outputs/cluster_{title_suffix}.png", dpi=150)
    plt.show()

# Plot key relationships by cluster
plot_clusters("median_household_income", "median_gross_rent", "income_rent")
plot_clusters("median_household_income", "rental_vacancy_rate", "income_vacancy")

# =============================================================================
# CLUSTER PROFILING
# =============================================================================

print("\n" + "="*60)
print("CLUSTER PROFILES (Median Values)")
print("="*60)

cluster_medians = df.groupby("cluster")[numeric_cols].median()
print(cluster_medians.to_string())

# =============================================================================
# CLUSTER INTERPRETATION
# =============================================================================

print("\n" + "="*60)
print("CLUSTER INTERPRETATION")
print("="*60)

# Analyze each cluster
for cluster_id in sorted(df["cluster"].unique()):
    cluster_data = df[df["cluster"] == cluster_id]
    med_income = cluster_data["median_household_income"].median()
    med_rent = cluster_data["median_gross_rent"].median()
    med_vacancy = cluster_data["rental_vacancy_rate"].median()
    
    print(f"\nCluster {cluster_id} ({len(cluster_data)} counties):")
    print(f"  Median Income:  ${med_income:,.0f}")
    print(f"  Median Rent:    ${med_rent:,.0f}")
    print(f"  Vacancy Rate:   {med_vacancy:.1f}%")
    
    # Characterize cluster
    if med_income > 70000 and med_rent > 1000:
        print("  → AFFLUENT/TIGHT MARKETS")
    elif med_income < 50000 and med_rent < 800:
        print("  → DISTRESSED/LOW-COST MARKETS")
    else:
        print("  → MIDDLE-TIER MARKETS")

print("\n" + "="*60)

# =============================================================================
# SAVE CLUSTERED DATA
# =============================================================================

df.to_csv("../Data/Prepared/acs_2022_with_clusters.csv", index=False)
print("\nSaved clustered data to: acs_2022_with_clusters.csv")

print("\nClustering analysis complete!")
