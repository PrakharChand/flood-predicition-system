"""
views.py

API views for the Flood Risk Assessment & Early Warning System.
"""

import json
import os
import pandas as pd
import joblib

from django.conf import settings
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt


# ==========================================================
# FEATURE ORDER (MUST MATCH TRAINING EXACTLY)
# ==========================================================
FEATURE_ORDER = [
    "Latitude",
    "Longitude",
    "Rainfall_mm",
    "Temperature_C",
    "Humidity_%",
    "River_Discharge_m³_s",
    "Water_Level_m",
    "Elevation_m",
    "Land_Cover",
    "Soil_Type",
    "Population_Density",
    "Infrastructure",
    "Historical_Floods",
    "Rainfall_WaterLevel",
    "Flood_Pressure",
    "Exposure_Index"
]


# ==========================================================
# CATEGORICAL ENCODING (MATCH TRAINING)
# ==========================================================
LAND_COVER_MAP = {
    "Forest": 0,
    "Rural": 1,
    "Urban": 2
}

SOIL_TYPE_MAP = {
    "Sandy": 0,
    "Silt": 1,
    "Clay": 2
}


def health_check(request):
    return JsonResponse(
        {
            "status": "ok",
            "message": "Flood Risk Backend is running successfully"
        },
        status=200
    )


@csrf_exempt
def predict_risk(request):
    if request.method != "POST":
        return JsonResponse(
            {"error": "Only POST requests are allowed"},
            status=405
        )

    try:
        payload = json.loads(request.body)

        # ==================================================
        # REQUIRED RAW FEATURES (13)
        # ==================================================
        required_features = [
            "Latitude",
            "Longitude",
            "Rainfall_mm",
            "Temperature_C",
            "Humidity_%",
            "River_Discharge_m³_s",
            "Water_Level_m",
            "Elevation_m",
            "Land_Cover",
            "Soil_Type",
            "Population_Density",
            "Infrastructure",
            "Historical_Floods"
        ]

        missing = [f for f in required_features if f not in payload]
        if missing:
            return JsonResponse(
                {"error": "Missing required features", "missing_features": missing},
                status=400
            )

        # ==================================================
        # LOAD MODEL
        # ==================================================
        model_path = os.path.join(
            settings.BASE_DIR.parent,
            "ml",
            "artifacts",
            "model.pkl"
        )

        model = joblib.load(model_path)

        # ==================================================
        # CREATE INPUT DATAFRAME
        # ==================================================
        input_df = pd.DataFrame([payload], columns=required_features)

        # ==================================================
        # ENCODE CATEGORICAL FEATURES
        # ==================================================
        input_df["Land_Cover"] = input_df["Land_Cover"].map(LAND_COVER_MAP)
        input_df["Soil_Type"] = input_df["Soil_Type"].map(SOIL_TYPE_MAP)

        if input_df[["Land_Cover", "Soil_Type"]].isnull().any().any():
            return JsonResponse(
                {"error": "Invalid categorical value"},
                status=400
            )

        # ==================================================
        # FEATURE ENGINEERING (3 FEATURES)
        # ==================================================
        input_df["Rainfall_WaterLevel"] = (
            input_df["Rainfall_mm"] * input_df["Water_Level_m"]
        )

        input_df["Flood_Pressure"] = (
            input_df["River_Discharge_m³_s"] / (input_df["Elevation_m"] + 1)
        )

        input_df["Exposure_Index"] = (
            input_df["Population_Density"] * input_df["Infrastructure"]
        )

        # ==================================================
        # PREDICTION (NUMPY ARRAY + CORRECT ORDER)
        # ==================================================
        X = input_df[FEATURE_ORDER].to_numpy()
        probability = model.predict_proba(X)[0][1]

        if probability < 0.3:
            risk = "Low Risk"
        elif probability < 0.7:
            risk = "Medium Risk"
        else:
            risk = "High Risk"

        return JsonResponse(
            {
                "probability": round(float(probability), 3),
                "risk": risk,
                "ml_loaded": True
            },
            status=200
        )

    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON payload"}, status=400)

    except FileNotFoundError:
        return JsonResponse(
            {"status": "degraded", "ml_loaded": False, "error": "Model file not found"},
            status=500
        )

    except Exception as e:
        return JsonResponse(
            {"error": "Internal server error", "details": str(e)},
            status=500
        )
