"""
Fetch Weather Data for Sukkur from Open-Meteo API (Historical + Current)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import os
import requests
import logging
from datetime import datetime, timezone, timedelta
from config.db import get_db
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load configuration from environment
CITY = os.getenv("CITY", "Sukkur")
SUKKUR_LAT = float(os.getenv("LATITUDE", "27.7058"))
SUKKUR_LON = float(os.getenv("LONGITUDE", "68.8574"))

MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
DB_NAME = os.getenv("DB_NAME", "aqi_db")


def fetch_historical_weather():
    """Fetch historical weather data (last 30 days) using Open-Meteo"""
    url = "https://archive-api.open-meteo.com/v1/archive"
    
    # Date range: Last 30 days (Open-Meteo requires historical dates)
    end_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
    
    params = {
        "latitude": SUKKUR_LAT,
        "longitude": SUKKUR_LON,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "temperature_2m,relative_humidity_2m,pressure_msl,wind_speed_10m,cloud_cover",
        "timezone": "UTC"
    }
    
    logger.info(f"Fetching historical weather data for {CITY} from {start_date} to {end_date}...")
    response = requests.get(url, params=params, timeout=60)
    response.raise_for_status()
    
    data = response.json()
    hourly = data.get("hourly", {})
    times = hourly.get("time", [])
    
    weather_records = []
    for i, time_str in enumerate(times):
        timestamp = datetime.fromisoformat(time_str.replace("Z", "+00:00"))
        
        record = {
            "city": CITY,
            "lat": SUKKUR_LAT,
            "lon": SUKKUR_LON,
            "timestamp": timestamp,
            "temp": hourly.get("temperature_2m", [])[i] if i < len(hourly.get("temperature_2m", [])) else None,
            "humidity": hourly.get("relative_humidity_2m", [])[i] if i < len(hourly.get("relative_humidity_2m", [])) else None,
            "pressure": hourly.get("pressure_msl", [])[i] if i < len(hourly.get("pressure_msl", [])) else None,
            "wind_speed": hourly.get("wind_speed_10m", [])[i] if i < len(hourly.get("wind_speed_10m", [])) else None,
            "clouds": hourly.get("cloud_cover", [])[i] if i < len(hourly.get("cloud_cover", [])) else None,
            "source": "open-meteo-archive",
            "fetched_at": datetime.now(timezone.utc)
        }
        weather_records.append(record)
    
    logger.info(f"✅ Fetched {len(weather_records)} historical weather records")
    return weather_records


def fetch_current_weather():
    """Fetch current/forecast weather data from Open-Meteo Forecast API"""
    url = "https://api.open-meteo.com/v1/forecast"
    
    params = {
        "latitude": SUKKUR_LAT,
        "longitude": SUKKUR_LON,
        "current": "temperature_2m,relative_humidity_2m,pressure_msl,wind_speed_10m,cloud_cover",
        "hourly": "temperature_2m,relative_humidity_2m,pressure_msl,wind_speed_10m,cloud_cover",
        "past_hours": 24,  # Include last 24 hours
        "forecast_hours": 48,  # Include next 48 hours
        "timezone": "UTC"
    }
    
    logger.info(f"Fetching current weather data for {CITY}...")
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    
    data = response.json()
    weather_records = []
    
    # Process hourly data
    hourly = data.get("hourly", {})
    times = hourly.get("time", [])
    
    for i, time_str in enumerate(times):
        timestamp = datetime.fromisoformat(time_str.replace("Z", "+00:00"))
        
        record = {
            "city": CITY,
            "lat": SUKKUR_LAT,
            "lon": SUKKUR_LON,
            "timestamp": timestamp,
            "temp": hourly.get("temperature_2m", [])[i] if i < len(hourly.get("temperature_2m", [])) else None,
            "humidity": hourly.get("relative_humidity_2m", [])[i] if i < len(hourly.get("relative_humidity_2m", [])) else None,
            "pressure": hourly.get("pressure_msl", [])[i] if i < len(hourly.get("pressure_msl", [])) else None,
            "wind_speed": hourly.get("wind_speed_10m", [])[i] if i < len(hourly.get("wind_speed_10m", [])) else None,
            "clouds": hourly.get("cloud_cover", [])[i] if i < len(hourly.get("cloud_cover", [])) else None,
            "source": "open-meteo-forecast",
            "fetched_at": datetime.now(timezone.utc)
        }
        weather_records.append(record)
    
    logger.info(f"✅ Fetched {len(weather_records)} current/forecast weather records")
    return weather_records


def fetch_weather_data():
    """Fetch weather data (historical + current)"""
    # Fetch historical data (for training)
    historical = fetch_historical_weather()
    
    # Fetch current/forecast data (for predictions)
    current = fetch_current_weather()
    
    # Combine and deduplicate
    all_records = historical + current
    logger.info(f"Total records (historical + current): {len(all_records)}")
    
    return all_records


def store_weather_data(records, db=None):
    """Store weather data in MongoDB"""
    if not records:
        logger.warning("No records to store")
        return
    db = db if db is not None else get_db()
    collection = db["raw_weather"]
    for record in records:
        collection.update_one(
            {"city": record["city"], "timestamp": record["timestamp"]},
            {"$set": record},
            upsert=True
        )
    logger.info(f"✅ Stored {len(records)} weather records in MongoDB")


if __name__ == "__main__":
    weather_data = fetch_weather_data()
    if weather_data:
        store_weather_data(weather_data)
