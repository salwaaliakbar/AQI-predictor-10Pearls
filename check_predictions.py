"""
Check prediction accuracy and model performance
"""
import pandas as pd
from pymongo import MongoClient
import os
from dotenv import load_dotenv
import numpy as np
import joblib
from pathlib import Path

load_dotenv()
MONGODB_URI = os.getenv('MONGODB_URI', 'mongodb://localhost:27017')
DB_NAME = os.getenv('DB_NAME', 'aqi_db')

client = MongoClient(MONGODB_URI)
db = client[DB_NAME]

# Check raw AQI data
raw_aqi = pd.DataFrame(list(db['raw_aqi'].find({}, {'_id': 0})))
raw_aqi['timestamp'] = pd.to_datetime(raw_aqi['timestamp'])
raw_aqi = raw_aqi.sort_values('timestamp')

print('=' * 70)
print('RECENT AQI TREND (Last 20 hours)')
print('=' * 70)
print(raw_aqi[['timestamp', 'us_aqi']].tail(20).to_string(index=False))

print('\n' + '=' * 70)
print('AQI STATISTICS (Last 7 days)')
print('=' * 70)
recent_7d = raw_aqi.tail(168)
print(f'Mean AQI: {recent_7d["us_aqi"].mean():.2f}')
print(f'Std Dev: {recent_7d["us_aqi"].std():.2f}')
print(f'Min: {recent_7d["us_aqi"].min():.2f}')
print(f'Max: {recent_7d["us_aqi"].max():.2f}')
print(f'Current (latest): {recent_7d["us_aqi"].iloc[-1]:.2f}')
print(f'Average (last 24h): {recent_7d["us_aqi"].tail(24).mean():.2f}')
print(f'Trend direction: {"📈 UP" if recent_7d["us_aqi"].tail(24).mean() > recent_7d["us_aqi"].iloc[-1] else "📉 DOWN"}')

# Check model metrics
print('\n' + '=' * 70)
print('MODEL PERFORMANCE')
print('=' * 70)
models = list(db['model_registry'].find({}, {'_id': 0}))
for model in models:
    print(f"""{model['model_name']}:
  R² Score (Test): {model['metrics']['r2_test']:.4f}
  MAE (Avg Error): {model['metrics']['mae_test']:.2f} AQI units
  RMSE: {model['metrics']['rmse_test']:.2f}
  Status: {"✅ Good" if model['metrics']['r2_test'] > 0.7 else "⚠️ Needs improvement"}\n""")

# Test model predictions on recent data
print('=' * 70)
print('MODEL VALIDATION (Predicting last 24 hours)')
print('=' * 70)

feature_store = pd.DataFrame(list(db['feature_store'].find({}, {'_id': 0})))
feature_store['timestamp'] = pd.to_datetime(feature_store['timestamp'])
feature_store = feature_store.sort_values('timestamp')

if len(feature_store) > 24:
    # Get last 24 records
    last_24 = feature_store.tail(24)
    actual_aqi = raw_aqi[raw_aqi['timestamp'].isin(last_24['timestamp'])]['us_aqi'].values
    
    # Load scaler and model
    models_dir = Path(__file__).parent / "models"
    scaler = joblib.load(models_dir / "scaler.pkl")
    best_model = models[0]  # Assuming first is best
    model_path = models_dir / f"{best_model['model_name'].lower().replace(' ', '_')}.pkl"
    
    if model_path.exists():
        model = joblib.load(model_path)
        feature_names = best_model['feature_names']
        
        # Prepare features
        X_test = []
        for _, row in last_24.iterrows():
            features = []
            for feat in feature_names:
                if feat in row.index:
                    val = row[feat]
                    features.append(float(val) if not pd.isna(val) else 0.0)
                else:
                    features.append(0.0)
            X_test.append(features)
        
        X_test = np.array(X_test)
        X_test_scaled = scaler.transform(X_test)
        predictions = model.predict(X_test_scaled)
        
        # Compare
        print(f"Model: {best_model['model_name']}\n")
        for i, (ts, pred, actual) in enumerate(zip(last_24['timestamp'].values, predictions, actual_aqi)):
            error = abs(pred - actual)
            error_pct = (error / actual * 100) if actual > 0 else 0
            print(f"{pd.Timestamp(ts).strftime('%Y-%m-%d %H:%M')} | Predicted: {pred:6.1f} | Actual: {actual:6.1f} | Error: {error:5.2f} ({error_pct:5.1f}%)")
        
        mae = np.mean(np.abs(predictions - actual_aqi))
        rmse = np.sqrt(np.mean((predictions - actual_aqi) ** 2))
        print(f"\nValidation MAE: {mae:.2f}")
        print(f"Validation RMSE: {rmse:.2f}")

client.close()
print('\n' + '=' * 70)
