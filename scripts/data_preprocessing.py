"""
Data Preprocessing: Clean, Augment, Handle Missing Values
"""
import pandas as pd
import numpy as np
import logging
from pymongo import MongoClient
import os
from dotenv import load_dotenv
from config.db import get_db

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_raw_data(db=None):
    """Load raw weather and AQI data from MongoDB"""
    db = db if db is not None else get_db()
    weather_df = pd.DataFrame(list(db["raw_weather"].find({}, {"_id": 0})))
    aqi_df = pd.DataFrame(list(db["raw_aqi"].find({}, {"_id": 0})))
    logger.info(f"Loaded {len(weather_df)} weather records, {len(aqi_df)} AQI records")
    return weather_df, aqi_df


def merge_data(weather_df, aqi_df):
    """Merge weather and AQI data on timestamp"""
    # Convert timestamp columns
    weather_df['timestamp'] = pd.to_datetime(weather_df['timestamp'])
    aqi_df['timestamp'] = pd.to_datetime(aqi_df['timestamp'])
    
    # Merge on city and timestamp
    merged = pd.merge(
        weather_df,
        aqi_df,
        on=['city', 'timestamp'],
        how='inner',
        suffixes=('_weather', '_aqi')
    )
    
    logger.info(f"Merged data: {len(merged)} records")
    return merged


def compute_us_aqi_from_pm25(pm25):
    """Approximate US AQI from PM2.5 using EPA breakpoints"""
    if pd.isna(pm25):
        return np.nan
    breakpoints = [
        (0.0, 12.0, 0, 50),
        (12.1, 35.4, 51, 100),
        (35.5, 55.4, 101, 150),
        (55.5, 150.4, 151, 200),
        (150.5, 250.4, 201, 300),
        (250.5, 350.4, 301, 400),
        (350.5, 500.4, 401, 500),
    ]
    for c_low, c_high, i_low, i_high in breakpoints:
        if c_low <= pm25 <= c_high:
            return ((i_high - i_low) / (c_high - c_low)) * (pm25 - c_low) + i_low
    return np.nan


def handle_missing_values(df):
    """Handle missing values using various strategies"""
    logger.info("Handling missing values...")
    
    # Numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    # Forward fill for time-series data
    df[numeric_cols] = df[numeric_cols].fillna(method='ffill')
    
    # Backward fill remaining
    df[numeric_cols] = df[numeric_cols].fillna(method='bfill')
    
    # Fill any remaining with median
    for col in numeric_cols:
        if df[col].isna().any():
            df[col].fillna(df[col].median(), inplace=True)
    
    # Remove rows with too many missing values (>50%)
    threshold = len(df.columns) * 0.5
    df = df.dropna(thresh=threshold)
    
    logger.info(f"✅ Missing values handled. {len(df)} records remaining")
    return df


def remove_outliers(df, columns, threshold=3):
    """Remove outliers using Z-score method"""
    logger.info("Removing outliers...")
    initial_len = len(df)

    # If dataset is small, skip outlier removal to avoid wiping everything
    if initial_len < 50:
        logger.warning("⚠️ Dataset < 50 rows; skipping outlier removal to preserve data")
        return df

    for col in columns:
        if col not in df.columns:
            continue

        col_std = df[col].std()
        if col_std == 0 or np.isnan(col_std):
            logger.warning(f"⚠️ Column {col} has zero/NaN std; skipping outlier removal for this column")
            continue

        z_scores = np.abs((df[col] - df[col].mean()) / col_std)
        mask = z_scores < threshold

        # Prevent over-removal: if keeping less than 80% rows, skip this column
        if mask.mean() < 0.8:
            logger.warning(f"⚠️ Outlier filter for {col} would drop >20% rows; skipping for this column")
            continue

        df = df[mask]

    removed = initial_len - len(df)
    logger.info(f"✅ Removed {removed} outlier records")
    return df


def augment_data(df):
    """Add derived features and temporal features"""
    logger.info("Augmenting data with additional features...")
    
    # Temporal features
    df['hour'] = df['timestamp'].dt.hour
    df['day'] = df['timestamp'].dt.day
    df['month'] = df['timestamp'].dt.month
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['is_rush_hour'] = ((df['hour'] >= 7) & (df['hour'] <= 9) | (df['hour'] >= 17) & (df['hour'] <= 19)).astype(int)
    
    # Interaction features
    if 'temp' in df.columns and 'humidity' in df.columns:
        df['temp_humidity_interaction'] = df['temp'] * df['humidity']
    
    if 'wind_speed' in df.columns and 'pressure' in df.columns:
        df['wind_pressure_interaction'] = df['wind_speed'] * df['pressure']
    
    # Rolling averages (3-hour window)
    if 'pm2_5' in df.columns:
        df['pm2_5_rolling_3h'] = df['pm2_5'].rolling(window=3, min_periods=1).mean()
    
    if 'temp' in df.columns:
        df['temp_rolling_3h'] = df['temp'].rolling(window=3, min_periods=1).mean()
    
    logger.info(f"✅ Data augmented with {df.shape[1]} features")
    return df


def preprocess_data(db=None):
    """Main preprocessing pipeline"""
    logger.info("=" * 70)
    logger.info("DATA PREPROCESSING PIPELINE")
    logger.info("=" * 70)
    
    # Load data
    weather_df, aqi_df = load_raw_data(db)
    
    # Merge datasets
    df = merge_data(weather_df, aqi_df)

    # If US AQI missing, approximate from PM2.5
    if 'us_aqi' in df.columns and df['us_aqi'].isna().any():
        df['us_aqi'] = df['us_aqi'].fillna(df['pm2_5'].apply(compute_us_aqi_from_pm25))
        logger.info("Filled missing us_aqi using PM2.5-based approximation")
    
    # Handle missing values
    df = handle_missing_values(df)
    
    # Remove outliers from key columns
    outlier_cols = ['pm2_5', 'pm10', 'temp', 'humidity', 'pressure', 'wind_speed']
    df = remove_outliers(df, outlier_cols)
    
    # Augment with additional features
    df = augment_data(df)
    
    # Store preprocessed data using MongoDBConnection
    db = db if db is not None else get_db()
    collection = db["preprocessed_data"]
    # Clear existing and insert new
    collection.delete_many({})
    records = df.to_dict('records')
    if records:
        collection.insert_many(records)
    logger.info(f"✅ Stored {len(records)} preprocessed records in MongoDB")
    
    logger.info("=" * 70)
    logger.info(f"PREPROCESSING COMPLETE: {df.shape[0]} rows, {df.shape[1]} features")
    logger.info("=" * 70)
    
    return df


if __name__ == "__main__":
    preprocess_data()
