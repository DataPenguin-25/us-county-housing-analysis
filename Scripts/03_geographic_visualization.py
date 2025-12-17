"""
03_geographic_visualization.py
US County-Level Housing & Employment Analysis

This script creates an interactive choropleth map of US counties
showing rental vacancy rates using the Folium library.

Author: Joseph Adamski
Project: CareerFoundry Data Analytics - Achievement 6
"""

import pandas as pd
import folium

# =============================================================================
# LOAD DATA
# =============================================================================

df = pd.read_csv("../Data/Prepared/acs_2022_county_data_clean.csv")

print("Data Shape:", df.shape)

# =============================================================================
# PREPARE FIPS CODES
# Ensure 5-digit format with leading zeros for geographic matching
# =============================================================================

df["fips"] = df["fips"].astype(str).str.zfill(5)

print("Sample FIPS codes:", df["fips"].head().tolist())

# =============================================================================
# GEOJSON DATA SOURCE
# US county boundaries from Plotly's public datasets
# =============================================================================

county_geo = "https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json"

# =============================================================================
# CREATE CHOROPLETH MAP
# =============================================================================

# Initialize map centered on the US
m = folium.Map(
    location=[37.8, -96],  # Center of continental US
    zoom_start=4,
    tiles="cartodbpositron"
)

# Add choropleth layer showing rental vacancy rates
folium.Choropleth(
    geo_data=county_geo,
    name="choropleth",
    data=df,
    columns=["fips", "rental_vacancy_rate"],  # FIPS code and value to map
    key_on="feature.id",                       # GeoJSON property to match
    fill_color="YlOrRd",                       # Yellow-Orange-Red color scale
    fill_opacity=0.7,
    line_opacity=0.2,
    nan_fill_color="white",                    # Counties with no data
    legend_name="Rental Vacancy Rate (%)"
).add_to(m)

# Add layer control
folium.LayerControl().add_to(m)

# =============================================================================
# SAVE MAP
# =============================================================================

output_path = "../Outputs/task_6_3_map.html"
m.save(output_path)

print(f"\nMap saved to: {output_path}")
print("Open this file in a web browser to view the interactive map.")

# =============================================================================
# KEY OBSERVATIONS
# =============================================================================

print("\n" + "="*60)
print("GEOGRAPHIC PATTERNS OBSERVED")
print("="*60)
print("- High vacancy rates: Rural Midwest, parts of Appalachia")
print("- Low vacancy rates: Coastal urban areas, major metros")
print("- Regional clustering suggests local economic factors drive vacancy")
