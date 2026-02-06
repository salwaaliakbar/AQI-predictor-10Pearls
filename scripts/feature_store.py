"""
Feature Store: Store and manage engineered features in MongoDB
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import pandas as pd
import logging
from datetime import datetime, timezone
from config.db import get_db
import os
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureStore:
    """Feature Store for managing ML features"""
    def __init__(self, db=None):
        self.db = db if db is not None else get_db()
        self.features_collection = self.db["feature_store"]
        self.metadata_collection = self.db["feature_metadata"]

    def register_feature(self, feature_name, description, data_type, source):
        metadata = {
            "feature_name": feature_name,
            "description": description,
            "data_type": data_type,
            "source": source,
            "created_at": datetime.now(timezone.utc),
            "updated_at": datetime.now(timezone.utc)
        }
        self.metadata_collection.update_one(
            {"feature_name": feature_name},
            {"$set": metadata},
            upsert=True
        )
        logger.info(f"✅ Registered feature: {feature_name}")

    def store_features(self, df):
        logger.info("Storing features in Feature Store...")
        numeric_cols = df.select_dtypes(include=['number']).columns
        for col in df.columns:
            if col == 'timestamp':
                continue
            feature_doc = {
                "feature_name": col,
                "timestamp": datetime.now(timezone.utc),
                "count": int(len(df[col])),
                "null_count": int(df[col].isna().sum())
            }
            if col in numeric_cols:
                feature_doc.update({
                    "mean": float(df[col].mean()) if not df[col].isna().all() else None,
                    "std": float(df[col].std()) if not df[col].isna().all() else None,
                    "min": float(df[col].min()) if not df[col].isna().all() else None,
                    "max": float(df[col].max()) if not df[col].isna().all() else None,
                    "median": float(df[col].median()) if not df[col].isna().all() else None
                })
            self.metadata_collection.update_one(
                {"feature_name": col},
                {"$set": {
                    "feature_name": col,
                    "statistics": feature_doc,
                    "updated_at": datetime.now(timezone.utc)
                }},
                upsert=True
            )
        self.features_collection.delete_many({})
        records = df.to_dict('records')
        if records:
            self.features_collection.insert_many(records)
        logger.info(f"✅ Stored {len(records)} records with {len(df.columns)} features")

    def get_features(self):
        df = pd.DataFrame(list(self.features_collection.find({}, {"_id": 0})))
        logger.info(f"Retrieved {len(df)} feature records")
        return df

    def get_feature_metadata(self):
        metadata = list(self.metadata_collection.find({}, {"_id": 0}))
        return pd.DataFrame(metadata)


def engineer_and_store_features(db=None):
    logger.info("=" * 70)
    logger.info("FEATURE ENGINEERING & STORAGE")
    logger.info("=" * 70)
    db = db if db is not None else get_db()
    df = pd.DataFrame(list(db["preprocessed_data"].find({}, {"_id": 0})))
    if df.empty:
        logger.error("No preprocessed data found!")
        return None
    logger.info(f"Loaded {len(df)} preprocessed records")
    feature_store = FeatureStore(db)
    features_metadata = {
        "pm2_5": ("PM2.5 concentration", "float", "open-meteo"),
        "pm10": ("PM10 concentration", "float", "open-meteo"),
        "us_aqi": ("US AQI index", "float", "open-meteo"),
        "temp": ("Temperature in Celsius", "float", "openweather"),
        "humidity": ("Humidity percentage", "float", "openweather"),
        "pressure": ("Atmospheric pressure", "float", "openweather"),
        "wind_speed": ("Wind speed", "float", "openweather"),
        "temp_humidity_interaction": ("Temperature-Humidity interaction", "float", "derived"),
        "wind_pressure_interaction": ("Wind-Pressure interaction", "float", "derived"),
    }
    for feature_name, (desc, dtype, src) in features_metadata.items():
        feature_store.register_feature(feature_name, desc, dtype, src)
    feature_store.store_features(df)
    logger.info("=" * 70)
    logger.info("FEATURE ENGINEERING COMPLETE")
    logger.info("=" * 70)
    return feature_store.get_features()
