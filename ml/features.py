"""
features.py

This module contains domain-inspired feature engineering logic
for the Flood Risk Assessment system.

IMPORTANT DESIGN RULE:
- This file must be used ONLY AFTER train-test split
- No fitting, no scaling, no model logic here
- Prevents data leakage by design
"""

import pandas as pd


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds engineered flood-risk features to the dataframe.

    Expected input columns (must already exist):
    - Rainfall_mm
    - Water_Level_m
    - River_Discharge_m³_s
    - Elevation_m
    - Population_Density
    - Infrastructure

    Returns:
    - DataFrame with new engineered features added
    """

    df = df.copy()

    # 1️⃣ Rainfall × Water Level interaction
    # Captures compounding surface water stress
    df["Rainfall_WaterLevel"] = (
        df["Rainfall_mm"] * df["Water_Level_m"]
    )

    # 2️⃣ Flood Pressure
    # River discharge normalized by elevation
    # +1 prevents division instability at low elevation
    df["Flood_Pressure"] = (
        df["River_Discharge_m³_s"] / (df["Elevation_m"] + 1)
    )

    # 3️⃣ Exposure Index
    # Combines human presence and infrastructure vulnerability
    df["Exposure_Index"] = (
        df["Population_Density"] * df["Infrastructure"]
    )

    return df
