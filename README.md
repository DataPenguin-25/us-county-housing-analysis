# US County-Level Housing & Employment Analysis

## Project Overview

This project analyzes the relationships between housing costs, household income, and employment across 3,200+ US counties using data from the American Community Survey (ACS) and Bureau of Labor Statistics (BLS).

**Author:** Joseph Adamski  
**Program:** CareerFoundry Data Analytics  
**Achievement:** 6 - Advanced Analytics & Dashboard Design

## Research Questions

1. What factors influence county-level rent prices?
2. How do housing markets cluster across the US?
3. What are the long-term unemployment trends?

## Key Findings

- **Income predicts rent:** R² = 0.60 — household income explains 60% of rent variation
- **Home values correlate with rent:** R² = 0.72 — strong positive relationship
- **Three distinct market types identified:**
  - Affluent/Tight Markets: High income, high rent, low vacancy
  - Middle-Tier Markets: Moderate across all metrics
  - Distressed/Low-Cost Markets: Low income, low rent, higher vacancy
- **Geographic patterns:** Coastal areas (California, Northeast) have highest rents
- **Unemployment cycles:** Major spikes during 2008-2010 recession and 2020 COVID pandemic

## Tableau Storyboard

📊 **[View Interactive Dashboard on Tableau Public](LINK_HERE)**

The storyboard includes:
- Correlation analysis visualizations
- Linear regression results
- K-Means clustering scatter plots
- US county choropleth map
- Unemployment time series

## Data Sources

### American Community Survey (ACS) 2022
- **Source:** U.S. Census Bureau
- **Coverage:** 3,212 US counties
- **Variables:** Median gross rent, median household income, median home value, rental vacancy rate, housing units

### Bureau of Labor Statistics (BLS) LAUS
- **Source:** Local Area Unemployment Statistics
- **Coverage:** Monthly data, 1990-2025
- **Variables:** Unemployment rate

## Repository Structure

```
us-county-housing-analysis/
│
├── README.md
│
├── Scripts/
│   ├── 01_data_sourcing.py         # Load and merge ACS + BLS data
│   ├── 02_exploratory_analysis.py  # Correlation analysis, visualizations
│   ├── 03_geographic_visualization.py  # Folium choropleth map
│   ├── 04_linear_regression.py     # Income → Rent regression model
│   ├── 05_clustering_analysis.py   # K-Means county segmentation
│   └── 06_time_series_analysis.py  # Decomposition, stationarity testing
│
├── Data/
│   └── Prepared/
│       ├── acs_2022_county_data_clean.csv
│       └── bls_unemployment_clean.csv
│
└── Outputs/
    └── task_6_3_map.html           # Interactive choropleth map
```

## Methods & Tools

| Task | Method | Tools |
|------|--------|-------|
| Data Cleaning | Missing value handling, type conversion | pandas, numpy |
| Exploratory Analysis | Correlation matrix, scatter plots | seaborn, matplotlib |
| Geographic Visualization | Choropleth mapping | folium |
| Regression | Linear regression, train/test split | scikit-learn |
| Clustering | K-Means, elbow method | scikit-learn |
| Time Series | Decomposition, Dickey-Fuller test | statsmodels |
| Dashboard | Interactive storyboard | Tableau Public |

## Limitations

- ACS data contains sampling error, especially for small rural counties
- County-level averages mask within-county inequality
- Single-variable regression does not capture all rent factors
- Time series analysis uses single county as example
- Cross-sectional data (2022 only), not longitudinal

## Future Work

- Add predictors to regression (vacancy rate, population density)
- Analyze rent burden (rent as % of income) by cluster
- Compare unemployment trends across market types
- Build predictive model for rent forecasting
- Incorporate population growth data

## How to Run

1. Clone this repository
2. Install dependencies: `pip install pandas numpy matplotlib seaborn scikit-learn folium statsmodels`
3. Place raw data files in `Data/Original/`
4. Run scripts in order: `01_data_sourcing.py` through `06_time_series_analysis.py`

## License

This project is for educational purposes as part of the CareerFoundry Data Analytics program.

---

*Note: The Tableau storyboard presents key visualizations and findings. Not all analysis steps are shown in the dashboard — refer to the Python scripts for complete methodology.*
