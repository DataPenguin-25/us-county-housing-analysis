"""
06_time_series_analysis.py
US County-Level Housing & Employment Analysis

This script performs time series analysis on BLS unemployment data:
- Time series decomposition (trend, seasonal, residual)
- Stationarity testing (Dickey-Fuller test)
- Differencing to achieve stationarity

Author: Joseph Adamski
Project: CareerFoundry Data Analytics - Achievement 6
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf
import warnings

warnings.filterwarnings("ignore")
plt.style.use('fivethirtyeight')

# =============================================================================
# LOAD BLS LAUS DATA
# Bureau of Labor Statistics - Local Area Unemployment Statistics
# =============================================================================

# Load raw fixed-width format BLS data
df_bls = pd.read_fwf(
    "../Data/Original/la.data.64.County",
    header=0
)

# Clean column names (remove whitespace, tabs, newlines)
df_bls.columns = df_bls.columns.str.replace(r"[\s\t\r\n]+", "", regex=True)

# Strip whitespace from string columns
for col in df_bls.columns:
    if df_bls[col].dtype == "object":
        df_bls[col] = df_bls[col].astype(str).str.strip()

print("BLS Raw Data Shape:", df_bls.shape)
print("Columns:", df_bls.columns.tolist())

# =============================================================================
# SELECT COUNTY FOR TIME SERIES ANALYSIS
# Using Autauga County, Alabama (FIPS: 01001) as example
# Series ID ending in "03" = unemployment rate
# =============================================================================

target_series = "LAUCN010010000000003"  # Autauga County, AL - Unemployment Rate

county_ts = df_bls[df_bls["series_id"] == target_series].copy()

print(f"\nSelected series: {target_series}")
print(f"Records: {len(county_ts)}")

# =============================================================================
# CONSTRUCT DATETIME INDEX
# BLS data has year and period (M01-M12) columns, not proper dates
# =============================================================================

# Clean year and month fields (remove non-digits)
county_ts["year"] = county_ts["year"].astype(str).str.replace(r"\D", "", regex=True)
county_ts["month"] = county_ts["period"].astype(str).str.replace(r"\D", "", regex=True)

# Convert to numeric
county_ts["year"] = pd.to_numeric(county_ts["year"], errors="coerce")
county_ts["month"] = pd.to_numeric(county_ts["month"], errors="coerce")

# Drop rows with invalid year/month
county_ts = county_ts.dropna(subset=["year", "month"])

# Ensure month is valid (1-12)
county_ts["month"] = county_ts["month"].clip(1, 12).astype(int)
county_ts["year"] = county_ts["year"].astype(int)

# Build datetime column
county_ts["date"] = pd.to_datetime(
    county_ts["year"].astype(str) + "-" + county_ts["month"].astype(str) + "-01",
    errors="coerce"
)

# Drop rows with invalid dates
county_ts = county_ts.dropna(subset=["date"])

# Set datetime index and sort
county_ts = county_ts.set_index("date").sort_index()

print(f"Date range: {county_ts.index.min()} to {county_ts.index.max()}")

# =============================================================================
# CLEAN UNEMPLOYMENT RATE VALUES
# =============================================================================

county_ts["value"] = (
    county_ts["value"]
    .astype(str)
    .str.replace(r"[^0-9.\-]", "", regex=True)
)

county_ts["value"] = pd.to_numeric(county_ts["value"], errors="coerce")

# Extract final time series
ts = county_ts["value"].dropna()

print(f"Time series length: {len(ts)} months")

# =============================================================================
# VISUALIZATION 1: Unemployment Rate Over Time
# =============================================================================

plt.figure(figsize=(12, 5))
plt.plot(ts)
plt.title("Unemployment Rate Over Time (Autauga County, AL)")
plt.xlabel("Date")
plt.ylabel("Unemployment Rate (%)")
plt.tight_layout()
plt.savefig("../Outputs/unemployment_time_series.png", dpi=150)
plt.show()

# =============================================================================
# TIME SERIES DECOMPOSITION
# Separate into: Trend + Seasonal + Residual components
# =============================================================================

decomp = sm.tsa.seasonal_decompose(ts, model="additive", period=12)

fig = decomp.plot()
fig.set_size_inches(12, 10)
plt.suptitle("Time Series Decomposition: Unemployment Rate", y=1.02)
plt.tight_layout()
plt.savefig("../Outputs/decomposition.png", dpi=150)
plt.show()

print("\nDecomposition complete:")
print("- Trend: Long-term direction of unemployment")
print("- Seasonal: Regular monthly patterns (e.g., holiday hiring)")
print("- Residual: Random fluctuations not explained by trend/seasonality")

# =============================================================================
# STATIONARITY TEST: Augmented Dickey-Fuller Test
# Null hypothesis: Series is non-stationary (has unit root)
# =============================================================================

print("\n" + "="*60)
print("AUGMENTED DICKEY-FULLER TEST (Original Series)")
print("="*60)

result = adfuller(ts)

print(f"ADF Statistic: {result[0]:.4f}")
print(f"p-value:       {result[1]:.4f}")
print("Critical Values:")
for key, value in result[4].items():
    print(f"  {key}: {value:.4f}")

if result[1] < 0.05:
    print("\nResult: STATIONARY (reject null hypothesis)")
else:
    print("\nResult: NON-STATIONARY (fail to reject null hypothesis)")
    print("→ Differencing required for modeling")

# =============================================================================
# DIFFERENCING: First Difference to Achieve Stationarity
# =============================================================================

ts_diff1 = ts.diff().dropna()

plt.figure(figsize=(12, 5))
plt.plot(ts_diff1)
plt.title("First Difference of Unemployment Rate")
plt.xlabel("Date")
plt.ylabel("Change in Unemployment Rate")
plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("../Outputs/first_difference.png", dpi=150)
plt.show()

# Test stationarity of differenced series
print("\n" + "="*60)
print("AUGMENTED DICKEY-FULLER TEST (After Differencing)")
print("="*60)

result_diff = adfuller(ts_diff1)

print(f"ADF Statistic: {result_diff[0]:.4f}")
print(f"p-value:       {result_diff[1]:.6f}")

if result_diff[1] < 0.05:
    print("\nResult: STATIONARY after first differencing ✓")
else:
    print("\nResult: Still non-stationary, may need more differencing")

# =============================================================================
# AUTOCORRELATION FUNCTION (ACF)
# =============================================================================

plt.figure(figsize=(10, 4))
plot_acf(ts_diff1, lags=40)
plt.title("Autocorrelation Function (Differenced Series)")
plt.tight_layout()
plt.savefig("../Outputs/acf_plot.png", dpi=150)
plt.show()

# =============================================================================
# SAVE CLEANED TIME SERIES
# =============================================================================

out = ts.reset_index()
out.columns = ["date", "unemployment_rate"]

out.to_csv("../Data/Prepared/bls_unemployment_clean.csv", index=False)

print("\nSaved: bls_unemployment_clean.csv")

# =============================================================================
# KEY FINDINGS
# =============================================================================

print("\n" + "="*60)
print("TIME SERIES KEY FINDINGS")
print("="*60)
print("1. Clear long-term cycles visible in unemployment data")
print("2. Major spikes during recessions (2008-2010, 2020)")
print("3. Seasonal patterns present (monthly hiring cycles)")
print("4. Original series is non-stationary")
print("5. First differencing achieves stationarity")
print("="*60)

print("\nTime series analysis complete!")
