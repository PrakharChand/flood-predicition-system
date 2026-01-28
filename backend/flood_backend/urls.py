"""
URL configuration for flood_backend.

Routes:
- /admin/     → Django admin
- /api/       → Flood risk API endpoints
"""

from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/', include('risk_api.urls')),
]
