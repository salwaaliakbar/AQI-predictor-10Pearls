"""
Fetch AQI Data for Sukkur from Open-Meteo API (Historical + Current)
"""
import requests
import logging
from datetime import datetime, timezone, timedelta
import os
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


def fetch_historical_aqi():
    """Fetch historical AQI data (last 30 days) using Open-Meteo Air Quality API"""
    # Air quality endpoint returns pollutant concentrations; supports start/end date
    url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    
    # Date range: Last 30 days
    end_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
    
    params = {
        "latitude": SUKKUR_LAT,
        "longitude": SUKKUR_LON,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone,us_aqi,european_aqi",
        "timezone": "UTC"
    }
    
    logger.info(f"Fetching historical AQI data for {CITY} from {start_date} to {end_date}...")
    response = requests.get(url, params=params, timeout=60)
    response.raise_for_status()
    
    data = response.json()
    hourly = data.get("hourly", {})
    times = hourly.get("time", [])
    
    aqi_records = []
    for i, time_str in enumerate(times):
        timestamp = datetime.fromisoformat(time_str.replace("Z", "+00:00"))
        
        record = {
            "city": CITY,
            "lat": SUKKUR_LAT,
            "lon": SUKKUR_LON,
            "timestamp": timestamp,
            "pm10": hourly.get("pm10", [])[i] if i < len(hourly.get("pm10", [])) else None,
            "pm2_5": hourly.get("pm2_5", [])[i] if i < len(hourly.get("pm2_5", [])) else None,
            "carbon_monoxide": hourly.get("carbon_monoxide", [])[i] if i < len(hourly.get("carbon_monoxide", [])) else None,
            "nitrogen_dioxide": hourly.get("nitrogen_dioxide", [])[i] if i < len(hourly.get("nitrogen_dioxide", [])) else None,
            "sulphur_dioxide": hourly.get("sulphur_dioxide", [])[i] if i < len(hourly.get("sulphur_dioxide", [])) else None,
            "ozone": hourly.get("ozone", [])[i] if i < len(hourly.get("ozone", [])) else None,
            "us_aqi": hourly.get("us_aqi", [])[i] if i < len(hourly.get("us_aqi", [])) else None,
            "european_aqi": hourly.get("european_aqi", [])[i] if i < len(hourly.get("european_aqi", [])) else None,
            "source": "open-meteo-archive",
            "fetched_at": datetime.now(timezone.utc)
        }
        aqi_records.append(record)
    
    logger.info(f"✅ Fetched {len(aqi_records)} historical AQI records")
    return aqi_records


def fetch_aqi_data():
    """Fetch AQI data (historical + current)"""
    return fetch_historical_aqi()


def store_aqi_data(records, db=None):
    """Store AQI data in MongoDB"""
    if not records:
        logger.warning("No records to store")
        return
    db = db if db is not None else get_db()
    collection = db["raw_aqi"]
    for record in records:
        collection.update_one(
            {"city": record["city"], "timestamp": record["timestamp"]},
            {"$set": record},
            upsert=True
        )
    logger.info(f"✅ Stored {len(records)} AQI records in MongoDB")


if __name__ == "__main__":
    aqi_data = fetch_aqi_data()
    if aqi_data:
        store_aqi_data(aqi_data)
