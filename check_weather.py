"""Check current weather data"""
from pymongo import MongoClient
import pandas as pd
from dotenv import load_dotenv
import os

load_dotenv()
client = MongoClient(os.getenv('MONGODB_URI', 'mongodb://localhost:27017'))
db = client[os.getenv('DB_NAME', 'aqi_db')]

weather = pd.DataFrame(list(db['raw_weather'].find({}, {'_id': 0})))
weather['timestamp'] = pd.to_datetime(weather['timestamp'])
weather = weather.sort_values('timestamp')

print('=' * 70)
print('CURRENT WEATHER DATA')
print('=' * 70)

# Get today's data
now = pd.Timestamp.now()
current_hour = now.replace(minute=0, second=0, microsecond=0)

# Find closest timestamp to current hour
weather['time_diff'] = abs(weather['timestamp'] - current_hour)
closest = weather.sort_values('time_diff').iloc[0]

print(f"Current time: {now}")
print(f"Closest data point: {closest['timestamp']}")
print(f"\nCurrent Weather in Sukkur:")
print(f"  Temperature: {closest['temp']:.1f}°C")
print(f"  Humidity: {closest['humidity']:.0f}%")
print(f"  Pressure: {closest['pressure']:.1f} hPa")
print(f"  Wind Speed: {closest['wind_speed']:.1f} km/h")
print(f"  Source: {closest['source']}")

print('\n' + '=' * 70)
print('TODAY\'S TEMPERATURE TREND')
print('=' * 70)
today = weather[weather['timestamp'].dt.date == now.date()]
if not today.empty:
    print(today[['timestamp', 'temp', 'humidity']].to_string(index=False))
else:
    print("No data available for today yet")

client.close()

