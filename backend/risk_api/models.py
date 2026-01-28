from django.db import models

# No database models are required for this project.
#
# This backend serves as an API layer that:
# - Accepts environmental input data
# - Performs ML-based risk inference (later)
# - Returns flood risk probability and category
#
# If persistence is needed in the future (e.g., logging predictions),
# models can be safely added here without refactoring the API.
