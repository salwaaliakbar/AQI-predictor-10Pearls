from config.db import get_db
import pandas as pd

db = get_db()

# Load and check lag features
df = pd.DataFrame(list(db['feature_store'].find({}, {'_id': 0})))
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values('timestamp')

print("=" * 70)
print("LAG FEATURES ANALYSIS")
print("=" * 70)

# Check recent data (high AQI period)
recent = df[df['us_aqi'] >= 150].tail(20)
print("\nRecent HIGH AQI Records (>=150):")
print(recent[['timestamp', 'us_aqi', 'aqi_lag_1', 'aqi_lag_3', 'aqi_lag_6']].to_string())

# Check statistics
print("\n" + "=" * 70)
print("Lag Feature Statistics:")
print("=" * 70)
if 'aqi_lag_1' in df.columns:
    lag_cols = [col for col in df.columns if 'aqi_lag' in col]
    for col in lag_cols:
        print(f"\n{col}:")
        print(f"  Mean: {df[col].mean():.2f}")
        print(f"  Max: {df[col].max():.2f}")
        print(f"  Min: {df[col].min():.2f}")
        print(f"  NaN count: {df[col].isna().sum()}")
