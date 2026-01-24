"""
Complete AQI Prediction Pipeline for Sukkur
Orchestrates: Data Fetching → Preprocessing → Feature Engineering → Model Training
"""


import sys
import logging
from pathlib import Path
from datetime import datetime
from config.db import get_db, close_client

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent / "scripts"))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_pipeline():
    """Execute complete ML pipeline"""
    logger.info("\n" + "=" * 80)
    logger.info(" " * 20 + "🌍 AQI PREDICTION SYSTEM - SUKKUR 🌍")
    logger.info("=" * 80)
    logger.info(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    db = get_db()
    try:
        # Step 1: Fetch Data
        logger.info("\U0001F4E1 STEP 1: Fetching Weather Data...")
        from scripts.fetch_weather import fetch_weather_data, store_weather_data
        weather_data = fetch_weather_data()
        if weather_data:
            store_weather_data(weather_data, db=db)

        logger.info("\n\U0001F4E1 STEP 2: Fetching AQI Data...")
        from scripts.fetch_AQI import fetch_aqi_data, store_aqi_data
        aqi_data = fetch_aqi_data()
        if aqi_data:
            store_aqi_data(aqi_data, db=db)

        # Step 2: Preprocess Data
        logger.info("\n\U0001F9F9 STEP 3: Preprocessing Data...")
        from scripts.data_preprocessing import preprocess_data
        preprocessed_df = preprocess_data(db=db)

        # Step 3: Feature Engineering & Store
        logger.info("\n\u2699\ufe0f STEP 4: Feature Engineering & Storage...")
        from scripts.feature_store import engineer_and_store_features
        features_df = engineer_and_store_features(db=db)

        # Step 4: Train Models
        logger.info("\n\U0001F916 STEP 5: Training Multiple Models...")
        from scripts.train_models import train_models
        train_models(db=db)

        # Step 5: Generate Predictions
        logger.info("\n\U0001F52E STEP 6: Generating AQI Predictions (Current + Next 3 Days)...")
        from scripts.predict_aqi import predict_aqi_forecast
        predict_aqi_forecast(db=db)
        return True
    except Exception as e:
        logger.error(f"\n❌ Pipeline failed: {str(e)}", exc_info=True)
        return False
    finally:
        close_client()


if __name__ == "__main__":
    success = run_pipeline()
    sys.exit(0 if success else 1)
