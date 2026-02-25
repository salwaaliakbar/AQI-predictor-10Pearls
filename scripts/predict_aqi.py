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

LAG_STEPS = [1, 3, 6, 12, 24]


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


def load_forecast_aqi(db=None):
    """Load forecast AQI data (current + next 3 days)"""
    db = db if db is not None else get_db()
    aqi_forecast = pd.DataFrame(list(db["raw_aqi"].find(
        {"source": "open-meteo-forecast"},
        {"_id": 0}
    )))
    if aqi_forecast.empty:
        logger.warning("No forecast AQI data found!")
        return pd.DataFrame()
    aqi_forecast['timestamp'] = pd.to_datetime(aqi_forecast['timestamp'])
    now = pd.Timestamp.now().tz_localize(None)
    future_data = aqi_forecast[aqi_forecast['timestamp'] >= now].copy()
    future_data = future_data.sort_values('timestamp').head(72)
    logger.info(f"Loaded {len(future_data)} forecast AQI records")
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
    
    # Weather change features (for pattern detection)
    df['wind_speed_change'] = df['wind_speed'].diff().fillna(0)
    df['pressure_change'] = df['pressure'].diff().fillna(0)
    df['temp_change'] = df['temp'].diff().fillna(0)
    
    # Rolling features
    df['temp_rolling_3h'] = df['temp'].rolling(window=3, min_periods=1).mean()
    
    # Fill missing values
    df = df.ffill().bfill().fillna(0)
    
    logger.info(f"Engineered features for {len(df)} forecast records")
    return df


def load_recent_aqi_history(db=None, hours=48):
    """Load recent AQI history for lag features"""
    db = db if db is not None else get_db()
    df = pd.DataFrame(list(db["preprocessed_data"].find({}, {"_id": 0, "timestamp": 1, "us_aqi": 1})))
    if df.empty:
        logger.warning("No historical AQI data found for lag features")
        return []
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp", "us_aqi"]).sort_values("timestamp")
    if df.empty:
        logger.warning("No valid AQI history after cleanup")
        return []
    history = df.tail(hours)["us_aqi"].astype(float).tolist()
    logger.info(f"Loaded {len(history)} AQI history points for lag features")
    return history


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

    # Load forecast AQI data (pollutants)
    forecast_aqi = load_forecast_aqi(db)
    if forecast_aqi.empty:
        logger.error("No forecast AQI data available for predictions!")
        return

    # Merge weather and AQI forecast data
    forecast_df = pd.merge(
        forecast_weather,
        forecast_aqi,
        on=['city', 'timestamp'],
        how='inner',
        suffixes=('_weather', '_aqi')
    )
    if forecast_df.empty:
        logger.error("No merged forecast data available after join!")
        return
    
    # Engineer features
    forecast_df = engineer_forecast_features(forecast_df)
    
    logger.info(f"✅ Engineered {len(forecast_df.columns)} forecast features")
    logger.info("Model will predict AQI using weather, pollutants, and lag features when available")
    
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
    logger.info(f"Expected features: {len(feature_names)} | Feature list: {feature_names[:5]}...")
    
    # Load model
    model_path = Path(best_model_doc.get("model_path", ""))
    if not model_path.exists():
        logger.error(f"Model file not found at {model_path}")
        return
    
    model = joblib.load(model_path)
    logger.info(f"✅ Loaded model from {model_path}")
    
    # Try to load scaler (optional, may not exist or may have dimension issues)
    scaler = None
    scaler_path = model_path.parent / "scaler.pkl"
    if scaler_path.exists():
        try:
            scaler_temp = joblib.load(scaler_path)
            # Check if scaler dimensions match
            if hasattr(scaler_temp, 'n_features_in_') and scaler_temp.n_features_in_ == len(feature_names):
                scaler = scaler_temp
                logger.info(f"✅ Loaded compatible scaler ({scaler.n_features_in_} features)")
            else:
                logger.warning(f"Scaler dimensions don't match ({scaler_temp.n_features_in_ if hasattr(scaler_temp, 'n_features_in_') else '?'} vs {len(feature_names)}). Skipping scaler.")
        except Exception as e:
            logger.warning(f"Could not load scaler: {e}. Proceeding without scaling.")
    
    # Load AQI history for lag features
    aqi_history = load_recent_aqi_history(db=db, hours=max(LAG_STEPS) + 24)
    history_mean = float(np.mean(aqi_history)) if aqi_history else 0.0

    # Prepare features in correct order (recursive for lag features)
    predictions = []
    valid_rows = []

    if scaler is None:
        logger.info("⚠️  No compatible scaler available. Using unscaled features.")

    for idx, row in forecast_df.iterrows():
        lag_values = {}
        for lag in LAG_STEPS:
            if len(aqi_history) >= lag:
                lag_values[f"aqi_lag_{lag}"] = float(aqi_history[-lag])
            else:
                lag_values[f"aqi_lag_{lag}"] = history_mean

        feature_values = []
        for feat in feature_names:
            if feat in row.index:
                val = row[feat]
                feature_values.append(float(val) if not pd.isna(val) else 0.0)
            elif feat in lag_values:
                feature_values.append(lag_values[feat])
            else:
                feature_values.append(0.0)

        X_row = pd.DataFrame([feature_values], columns=feature_names)
        if scaler is not None:
            try:
                X_row = scaler.transform(X_row)
            except ValueError as e:
                logger.warning(f"Scaler transform failed: {e}. Using unscaled features.")

        try:
            pred = float(model.predict(X_row)[0])
        except Exception as e:
            logger.error(f"Prediction failed at {idx}: {e}")
            return

        predictions.append(pred)
        aqi_history.append(pred)
        valid_rows.append(idx)

    forecast_subset = forecast_df.loc[valid_rows].copy()
    predictions = np.array(predictions, dtype=float)
    logger.info(f"Prepared {len(predictions)} forecast samples with {len(feature_names)} features")
    
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
    logger.info(f"Std Dev predicted AQI: {predictions.std():.1f}")
    
    # Daily summary
    forecast_subset['date'] = forecast_subset['timestamp'].dt.date
    daily = forecast_subset.groupby('date')['predicted_aqi'].agg(['mean', 'min', 'max']).round(1)
    logger.info("\n📅 Daily Forecast:")
    for date, row in daily.iterrows():
        category = get_category(row['mean'])
        logger.info(f"  {date}: Avg={row['mean']:.1f}, Min={row['min']:.1f}, Max={row['max']:.1f} ({category})")
    
    logger.info("=" * 70)
    logger.info("✅ AUTOMATED PREDICTION COMPLETE")
    logger.info("=" * 70)


if __name__ == "__main__":
    predict_aqi_forecast()