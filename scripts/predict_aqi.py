"""
Automated AQI Prediction - Predict Current & Next 3 Days
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timezone
import os
from dotenv import load_dotenv
from pathlib import Path
import joblib
from config.db import get_db

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_forecast_weather(db=None):
    """Load forecast weather data (current + next 3 days)"""
    db = db if db is not None else get_db()
    # Get forecast data (source = open-meteo-forecast)
    weather_forecast = pd.DataFrame(list(db["raw_weather"].find(
        {"source": "open-meteo-forecast"}, 
        {"_id": 0}
    )))
    if weather_forecast.empty:
        logger.warning("No forecast weather data found!")
        return pd.DataFrame()
    weather_forecast['timestamp'] = pd.to_datetime(weather_forecast['timestamp'])
    # Filter for current time onwards (next 72 hours)
    now = pd.Timestamp.now().tz_localize(None)  # Remove timezone for comparison
    future_data = weather_forecast[weather_forecast['timestamp'] >= now].copy()
    future_data = future_data.sort_values('timestamp').head(72)  # Next 72 hours
    logger.info(f"Loaded {len(future_data)} forecast weather records")
    return future_data


def engineer_forecast_features(df):
    """Apply same feature engineering as training data"""
    if df.empty:
        return df
    
    # Time-based features
    df['hour'] = df['timestamp'].dt.hour
    df['day'] = df['timestamp'].dt.day
    df['month'] = df['timestamp'].dt.month
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['is_rush_hour'] = df['hour'].isin([7, 8, 9, 17, 18, 19]).astype(int)
    
    # Interaction features
    df['temp_humidity_interaction'] = df['temp'] * df['humidity']
    df['wind_pressure_interaction'] = df['wind_speed'] * df['pressure']
    
    # Rolling features
    df['pm2_5_rolling_3h'] = 0  # Default for forecast
    df['temp_rolling_3h'] = df['temp'].rolling(window=3, min_periods=1).mean()
    
    # Fill missing values
    df = df.ffill().bfill().fillna(0)
    
    logger.info(f"Engineered features for {len(df)} forecast records")
    return df


def predict_aqi_forecast(db=None):
    """Generate AQI predictions for current + next 3 days"""
    logger.info("=" * 70)
    logger.info("AUTOMATED AQI PREDICTION - CURRENT + NEXT 3 DAYS")
    logger.info("=" * 70)
    
    # Load forecast weather data
    db = db if db is not None else get_db()
    forecast_weather = load_forecast_weather(db)
    if forecast_weather.empty:
        logger.error("No forecast data available for predictions!")
        return
    
    # Engineer features
    forecast_df = engineer_forecast_features(forecast_weather)
    
    # Load model registry to get feature names
    models = list(db["model_registry"].find({}, {"_id": 0}))
    
    if not models:
        logger.error("No trained models found! Run training first.")
        return
    
    # Get best model
    best_model_doc = max(models, key=lambda x: x['metrics']['r2_test'])
    model_name = best_model_doc['model_name']
    feature_names = best_model_doc['feature_names']
    
    logger.info(f"Using model: {model_name} (R² = {best_model_doc['metrics']['r2_test']:.4f})")
    
    # Load model and scaler
    models_dir = Path("models")
    model_path = models_dir / f"{model_name.lower().replace(' ', '_')}.pkl"
    scaler_path = models_dir / "scaler.pkl"
    
    if not model_path.exists() or not scaler_path.exists():
        logger.error(f"Model or scaler not found!")
        return
    
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    
    # Prepare features in correct order
    X_forecast = []
    valid_rows = []
    
    for idx, row in forecast_df.iterrows():
        feature_values = []
        all_present = True
        
        for feat in feature_names:
            if feat in row.index:
                val = row[feat]
                feature_values.append(float(val) if not pd.isna(val) else 0.0)
            else:
                feature_values.append(0.0)
        
        X_forecast.append(feature_values)
        valid_rows.append(idx)
    
    X_forecast = np.array(X_forecast)
    forecast_subset = forecast_df.loc[valid_rows].copy()
    
    # Scale and predict
    X_forecast_scaled = scaler.transform(X_forecast)
    predictions = model.predict(X_forecast_scaled)
    
    # Create predictions dataframe
    forecast_subset['predicted_aqi'] = predictions
    forecast_subset['model_name'] = model_name
    forecast_subset['predicted_at'] = datetime.now(timezone.utc)
    
    # Determine AQI category
    def get_category(aqi):
        if aqi <= 50:
            return "Good"
        elif aqi <= 100:
            return "Moderate"
        elif aqi <= 150:
            return "Unhealthy for Sensitive Groups"
        elif aqi <= 200:
            return "Unhealthy"
        elif aqi <= 300:
            return "Very Unhealthy"
        else:
            return "Hazardous"
    
    forecast_subset['aqi_category'] = forecast_subset['predicted_aqi'].apply(get_category)
    
    # Store predictions in MongoDB
    collection = db["aqi_predictions"]
    # Clear old predictions
    collection.delete_many({})
    # Insert new predictions
    records = forecast_subset.to_dict('records')
    collection.insert_many(records)
    
    logger.info(f"✅ Stored {len(records)} AQI predictions in MongoDB")
    
    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("PREDICTION SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Predictions: {len(predictions)}")
    logger.info(f"Time range: {forecast_subset['timestamp'].min()} to {forecast_subset['timestamp'].max()}")
    logger.info(f"Average predicted AQI: {predictions.mean():.1f}")
    logger.info(f"Min predicted AQI: {predictions.min():.1f}")
    logger.info(f"Max predicted AQI: {predictions.max():.1f}")
    
    # Daily summary
    forecast_subset['date'] = forecast_subset['timestamp'].dt.date
    daily = forecast_subset.groupby('date')['predicted_aqi'].agg(['mean', 'min', 'max']).round(1)
    logger.info("\n📅 Daily Forecast:")
    for date, row in daily.iterrows():
        category = get_category(row['mean'])
        logger.info(f"  {date}: Avg={row['mean']:.1f}, Min={row['min']:.1f}, Max={row['max']:.1f} ({category})")
    
    # No client close needed for persistent connection
    
    logger.info("=" * 70)
    logger.info("✅ AUTOMATED PREDICTION COMPLETE")
    logger.info("=" * 70)


if __name__ == "__main__":
    predict_aqi_forecast()
