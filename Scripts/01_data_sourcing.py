"""
01_data_sourcing.py
US County-Level Housing & Employment Analysis

This script loads and merges data from two sources:
- American Community Survey (ACS) 2022: County-level housing metrics
- Bureau of Labor Statistics (BLS) LAUS: County-level unemployment data

Author: Joseph Adamski
Project: CareerFoundry Data Analytics - Achievement 6
"""

import os
import pandas as pd

# =============================================================================
# FILE PATHS
# =============================================================================

ACS_PATH = "../Data/Original/acs_2022_county_data.csv"
BLS_PATH = "../Data/Original/la.data.64.County"

# =============================================================================
# LOAD ACS DATA
# American Community Survey 2022 - Housing metrics by county
# =============================================================================

acs = pd.read_csv(ACS_PATH)

print("ACS Data Shape:", acs.shape)
print("ACS Columns:", acs.columns.tolist())

# =============================================================================
# LOAD BLS LAUS DATA
# Local Area Unemployment Statistics - Monthly unemployment by county
# =============================================================================

laus_raw = pd.read_csv(BLS_PATH, sep="\t", engine="python")

# Clean column names and strip whitespace
laus_raw["series_id"] = laus_raw["series_id"].astype(str).str.strip()
laus_raw["period"] = laus_raw["period"].astype(str).str.strip()

print("LAUS Raw Data Shape:", laus_raw.shape)

# =============================================================================
# FILTER BLS DATA
# Series IDs ending in "03" represent unemployment rates
# =============================================================================

laus = laus_raw[laus_raw["series_id"].str.endswith("03")].copy()

# Remove annual averages (M13), keep only monthly data (M01-M12)
laus = laus[laus["period"].str.startswith("M") & (laus["period"] != "M13")]

# =============================================================================
# EXTRACT FIPS CODE AND CLEAN FIELDS
# =============================================================================

# Extract month from period (e.g., "M01" -> 1)
laus["month"] = laus["period"].str[1:].astype(int)

# Extract 5-digit FIPS code from series_id (positions 5-10)
laus["fips"] = laus["series_id"].str[5:10]

# Convert unemployment rate to numeric
laus["unemployment_rate"] = pd.to_numeric(laus["value"], errors="coerce")

# Keep only relevant columns
laus = laus[["fips", "year", "month", "unemployment_rate"]]

print("Cleaned LAUS Data Shape:", laus.shape)

# =============================================================================
# STANDARDIZE FIPS CODES
# Ensure 5-digit format with leading zeros
# =============================================================================

acs["fips"] = acs["fips"].astype(str).str.zfill(5)
laus["fips"] = laus["fips"].astype(str).str.zfill(5)

# =============================================================================
# MERGE DATASETS
# Join ACS housing data with BLS unemployment data on FIPS code
# =============================================================================

df = laus.merge(acs, on="fips", how="left")

print("Merged Data Shape:", df.shape)
print("Merged Data Columns:", df.columns.tolist())

# =============================================================================
# SAVE CLEANED DATA
# =============================================================================

# Save ACS data for Tableau
acs.to_csv("../Data/Prepared/acs_2022_county_data_clean.csv", index=False)
print("Saved: acs_2022_county_data_clean.csv")

print("\nData sourcing complete!")
