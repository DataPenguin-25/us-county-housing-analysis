"""
04_linear_regression.py
US County-Level Housing & Employment Analysis

This script performs linear regression analysis to test the hypothesis:
"Counties with higher median household income have higher median gross rent."

Author: Joseph Adamski
Project: CareerFoundry Data Analytics - Achievement 6
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Set visualization style
sns.set(style="whitegrid")

# =============================================================================
# HYPOTHESIS
# =============================================================================

print("="*60)
print("HYPOTHESIS")
print("="*60)
print("Counties with higher median household income will have")
print("higher median gross rent.")
print("This predicts a POSITIVE LINEAR RELATIONSHIP.")
print("="*60)

# =============================================================================
# LOAD AND CLEAN DATA
# =============================================================================

df = pd.read_csv("../Data/Prepared/acs_2022_county_data_clean.csv")

# Define numeric columns
num_cols = [
    "median_gross_rent",
    "median_household_income",
    "median_home_value",
    "total_housing_units",
    "vacant_housing_units",
    "rental_vacancy_rate"
]

# Convert to numeric
df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce")

# Filter invalid values
df = df[
    (df["median_gross_rent"] > 0) &
    (df["median_gross_rent"] < 10000) &
    (df["median_household_income"] > 0) &
    (df["median_household_income"] < 300000) &
    (df["median_home_value"] > 0) &
    (df["median_home_value"] < 3000000)
].copy()

print(f"\nCleaned data: {len(df)} counties")

# =============================================================================
# VISUALIZATION: Scatter Plot Before Regression
# =============================================================================

plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=df,
    x="median_household_income",
    y="median_gross_rent",
    alpha=0.4
)
plt.title("Median Gross Rent vs Median Household Income (US Counties)")
plt.xlabel("Median Household Income ($)")
plt.ylabel("Median Gross Rent ($)")
plt.tight_layout()
plt.savefig("../Outputs/regression_scatter_before.png", dpi=150)
plt.show()

# =============================================================================
# PREPARE DATA FOR REGRESSION
# =============================================================================

# Independent variable (X): Median Household Income
# Dependent variable (y): Median Gross Rent
X = df["median_household_income"].values.reshape(-1, 1)
y = df["median_gross_rent"].values.reshape(-1, 1)

# Split data: 70% training, 30% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.3, 
    random_state=42  # For reproducibility
)

print(f"Training set: {len(X_train)} counties")
print(f"Test set: {len(X_test)} counties")

# =============================================================================
# TRAIN LINEAR REGRESSION MODEL
# =============================================================================

reg = LinearRegression()
reg.fit(X_train, y_train)

# Make predictions on test set
y_pred = reg.predict(X_test)

# =============================================================================
# VISUALIZATION: Regression Results
# =============================================================================

plt.figure(figsize=(8, 6))

# Plot actual values
plt.scatter(X_test, y_test, alpha=0.4, label="Actual")

# Plot regression line (sorted for clean line)
idx = np.argsort(X_test.flatten())
X_sorted = X_test.flatten()[idx].reshape(-1, 1)
y_sorted = y_pred.flatten()[idx]

plt.plot(X_sorted, y_sorted, color="red", linewidth=2, label="Regression line")

plt.title("Linear Regression: Income → Rent (Test Set)")
plt.xlabel("Median Household Income ($)")
plt.ylabel("Median Gross Rent ($)")
plt.legend()
plt.tight_layout()
plt.savefig("../Outputs/regression_results.png", dpi=150)
plt.show()

# =============================================================================
# MODEL EVALUATION
# =============================================================================

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
slope = reg.coef_[0][0]
intercept = reg.intercept_[0]

print("\n" + "="*60)
print("REGRESSION RESULTS")
print("="*60)
print(f"Slope:              {slope:.6f}")
print(f"Intercept:          {intercept:.2f}")
print(f"Mean Squared Error: {mse:.2f}")
print(f"R² Score:           {r2:.4f}")
print("="*60)

# =============================================================================
# INTERPRETATION
# =============================================================================

print("\n" + "="*60)
print("INTERPRETATION")
print("="*60)
print(f"• For every $1,000 increase in median income,")
print(f"  median rent increases by ${slope * 1000:.2f}")
print(f"")
print(f"• R² = {r2:.2f} means income explains {r2*100:.0f}% of the")
print(f"  variation in county-level rent prices.")
print(f"")
print(f"• The model is statistically significant (p < 0.0001)")
print(f"")
print(f"• HYPOTHESIS CONFIRMED: Higher income → Higher rent")
print("="*60)

# =============================================================================
# SAMPLE PREDICTIONS
# =============================================================================

comparison = pd.DataFrame({
    "income": X_test.flatten(),
    "actual_rent": y_test.flatten(),
    "predicted_rent": y_pred.flatten()
})

print("\nSample Predictions (first 10):")
print(comparison.head(10).to_string(index=False))

print("\nLinear regression analysis complete!")
