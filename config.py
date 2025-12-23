"""
Application Configuration Settings
Created: 2025-12-23 23:23:27 UTC
Author: Ah11zx
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ============================================================================
# Application Settings
# ============================================================================

APP_NAME = "ExpertOptionAPI Analysis & Recommendations"
APP_VERSION = "1.0.0"
DEBUG = os.getenv("DEBUG", "False").lower() == "true"
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")

# ============================================================================
# API Configuration
# ============================================================================

EXPERT_OPTION_API_BASE_URL = os.getenv(
    "EXPERT_OPTION_API_BASE_URL", 
    "https://api.expertoption.com"
)
EXPERT_OPTION_API_TIMEOUT = int(os.getenv("EXPERT_OPTION_API_TIMEOUT", "30"))
EXPERT_OPTION_API_MAX_RETRIES = int(os.getenv("EXPERT_OPTION_API_MAX_RETRIES", "3"))

# ============================================================================
# Authentication
# ============================================================================

API_KEY = os.getenv("API_KEY", "")
API_SECRET = os.getenv("API_SECRET", "")
USERNAME = os.getenv("USERNAME", "")
PASSWORD = os.getenv("PASSWORD", "")

# ============================================================================
# Database Configuration
# ============================================================================

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "sqlite:///app.db"
)
DATABASE_ECHO = os.getenv("DATABASE_ECHO", "False").lower() == "true"

# ============================================================================
# Logging Configuration
# ============================================================================

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_FILE = os.getenv("LOG_FILE", "app.log")
LOG_MAX_BYTES = int(os.getenv("LOG_MAX_BYTES", "10485760"))  # 10MB
LOG_BACKUP_COUNT = int(os.getenv("LOG_BACKUP_COUNT", "5"))

# ============================================================================
# Analysis Configuration
# ============================================================================

ANALYSIS_LOOKBACK_PERIOD = int(os.getenv("ANALYSIS_LOOKBACK_PERIOD", "30"))  # days
ANALYSIS_MIN_DATA_POINTS = int(os.getenv("ANALYSIS_MIN_DATA_POINTS", "100"))
ANALYSIS_CONFIDENCE_THRESHOLD = float(os.getenv("ANALYSIS_CONFIDENCE_THRESHOLD", "0.75"))

# ============================================================================
# Recommendation Settings
# ============================================================================

RECOMMENDATION_MIN_SCORE = float(os.getenv("RECOMMENDATION_MIN_SCORE", "0.6"))
RECOMMENDATION_MAX_SUGGESTIONS = int(os.getenv("RECOMMENDATION_MAX_SUGGESTIONS", "10"))
ENABLE_RISK_ANALYSIS = os.getenv("ENABLE_RISK_ANALYSIS", "True").lower() == "true"

# ============================================================================
# Cache Configuration
# ============================================================================

CACHE_ENABLED = os.getenv("CACHE_ENABLED", "True").lower() == "true"
CACHE_TTL = int(os.getenv("CACHE_TTL", "3600"))  # seconds
CACHE_MAX_SIZE = int(os.getenv("CACHE_MAX_SIZE", "1000"))

# ============================================================================
# Rate Limiting
# ============================================================================

RATE_LIMIT_ENABLED = os.getenv("RATE_LIMIT_ENABLED", "True").lower() == "true"
RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "100"))
RATE_LIMIT_PERIOD = int(os.getenv("RATE_LIMIT_PERIOD", "60"))  # seconds

# ============================================================================
# Notification Settings
# ============================================================================

NOTIFICATION_ENABLED = os.getenv("NOTIFICATION_ENABLED", "False").lower() == "true"
NOTIFICATION_EMAIL = os.getenv("NOTIFICATION_EMAIL", "")
NOTIFICATION_WEBHOOK_URL = os.getenv("NOTIFICATION_WEBHOOK_URL", "")

# ============================================================================
# Feature Flags
# ============================================================================

ENABLE_ADVANCED_ANALYSIS = os.getenv("ENABLE_ADVANCED_ANALYSIS", "True").lower() == "true"
ENABLE_REAL_TIME_UPDATES = os.getenv("ENABLE_REAL_TIME_UPDATES", "False").lower() == "true"
ENABLE_HISTORICAL_DATA_EXPORT = os.getenv("ENABLE_HISTORICAL_DATA_EXPORT", "True").lower() == "true"

# ============================================================================
# Security
# ============================================================================

SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-change-in-production")
ALLOWED_HOSTS = os.getenv("ALLOWED_HOSTS", "localhost,127.0.0.1").split(",")
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")

# ============================================================================
# Configuration Validation
# ============================================================================

def validate_config():
    """Validate critical configuration settings."""
    errors = []
    
    if ENVIRONMENT == "production":
        if SECRET_KEY == "your-secret-key-change-in-production":
            errors.append("SECRET_KEY must be changed in production")
        if not API_KEY:
            errors.append("API_KEY is required in production")
    
    if errors:
        raise ValueError(f"Configuration validation errors: {', '.join(errors)}")
    
    return True

# ============================================================================
# Configuration Dictionary
# ============================================================================

CONFIG = {
    "app": {
        "name": APP_NAME,
        "version": APP_VERSION,
        "debug": DEBUG,
        "environment": ENVIRONMENT,
    },
    "api": {
        "base_url": EXPERT_OPTION_API_BASE_URL,
        "timeout": EXPERT_OPTION_API_TIMEOUT,
        "max_retries": EXPERT_OPTION_API_MAX_RETRIES,
    },
    "analysis": {
        "lookback_period": ANALYSIS_LOOKBACK_PERIOD,
        "min_data_points": ANALYSIS_MIN_DATA_POINTS,
        "confidence_threshold": ANALYSIS_CONFIDENCE_THRESHOLD,
    },
    "recommendations": {
        "min_score": RECOMMENDATION_MIN_SCORE,
        "max_suggestions": RECOMMENDATION_MAX_SUGGESTIONS,
        "enable_risk_analysis": ENABLE_RISK_ANALYSIS,
    },
}
