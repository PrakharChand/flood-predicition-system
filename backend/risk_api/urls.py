"""
URL configuration for risk_api.

Endpoints:
- /api/health/   → backend health check
- /api/predict/  → flood risk prediction (logic in views)
"""

from django.urls import path
from . import views

urlpatterns = [
    path('health/', views.health_check, name='health_check'),
    path('predict/', views.predict_risk, name='predict_risk'),
]
