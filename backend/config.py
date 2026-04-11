# ============================================================
# NOTE: This is a DEVELOPMENT/SETUP version only.
# Configuration module for the backend.
# Loads environment variables from .env file.
# ============================================================

from dotenv import load_dotenv
import os

# Load .env file
load_dotenv()

# App Configuration
APP_NAME = os.getenv("APP_NAME", "Signature Classifier API")
DEBUG    = os.getenv("DEBUG", "False") == "True"

# Model Configuration
# TODO: Update MODEL_PATH once your trained model is finalized
MODEL_PATH = os.getenv("MODEL_PATH", "model/signature_model.pt")

# CORS Configuration
# TODO: Restrict ALLOWED_ORIGINS to your frontend URL in production
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:5173").split(",")