"""
Train 3 Optimized Models for AQI Prediction: Random Forest, XGBoost, Gradient Boosting
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import pandas as pd
import numpy as np
import logging
from pymongo import MongoClient
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

import os
from dotenv import load_dotenv
from pathlib import Path
import joblib
from config.db import get_db
from model_registry import ModelRegistry

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import ML libraries
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor

logger.info("✅ All ML libraries loaded: Random Forest, XGBoost")


def load_features(db=None):
    """Load features from Feature Store"""
    db = db if db is not None else get_db()
    df = pd.DataFrame(list(db["feature_store"].find({}, {"_id": 0})))
    logger.info(f"Loaded {len(df)} records from Feature Store")
    return df


def prepare_training_data(df, target_col="us_aqi"):
    """Prepare features and target for training"""
    exclude_cols = ['timestamp', 'city', 'lat', 'lon', 'fetched_at', 
                   'weather_main', 'weather_description', 'source_weather', 'source_aqi',
                   target_col, 'pm2_5', 'pm10', 'european_aqi']
    
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    df = df.dropna(subset=[target_col])
    
    X = df[feature_cols].select_dtypes(include=[np.number])
    y = df[target_col]
    
    # Fill any remaining NaN in features
    X = X.fillna(X.median())
    
    logger.info(f"Features: {X.shape[1]}, Samples: {len(X)}")
    logger.info(f"Feature names: {list(X.columns)[:10]}...")
    
    return X, y, list(X.columns)


def add_lag_features(df, target_col="us_aqi", lags=[1, 3, 6, 12, 24, 48]):
    """Add lagged target values as features for time-series prediction"""
    for lag in lags:
        df[f"aqi_lag_{lag}"] = df[target_col].shift(lag)
    return df


def prepare_lstm_data(X, y, timesteps=24):
    """Prepare data for LSTM (time series format)"""
    X_lstm = []
    y_lstm = []
    
    for i in range(len(X) - timesteps):
        X_lstm.append(X.iloc[i:i+timesteps].values)
        y_lstm.append(y.iloc[i+timesteps])
    
    return np.array(X_lstm), np.array(y_lstm)


def train_models(db=None):
    """Train 3 specialized AQI prediction models"""
    logger.info("=" * 70)
    logger.info("TRAINING 3 OPTIMIZED MODELS FOR AQI PREDICTION")
    logger.info("=" * 70)
    
    # Load features
    db = db if db is not None else get_db()
    df = load_features(db)
    if df.empty:
        logger.error("No features found!")
        return
    
    # ⚠️ CRITICAL: Sort by timestamp for time-series evaluation
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        before = len(df)
        df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
        after = len(df)
        logger.info(f"Sorted by timestamp | Dropped {before - after} invalid rows")
    
    # Add lag features (previous AQI values are strong predictors)
    logger.info("Adding lag features (past AQI values)...")
    df = add_lag_features(df, target_col="us_aqi", lags=[1, 3, 6, 12, 24, 48])
    df = df.dropna()  # Drop rows with NaN from lagging
    
    # Prepare data
    X, y, feature_names = prepare_training_data(df)
    
    # Time-Series Cross-Validation (proper evaluation for time-series)
    tscv = TimeSeriesSplit(n_splits=5)
    
    # Use final split for final predictions (80/20)
    split_index = int(len(X) * 0.8)
    X_train = X.iloc[:split_index]
    X_test  = X.iloc[split_index:]
    y_train = y.iloc[:split_index]
    y_test  = y.iloc[split_index:]
    
    # # Scale features
    # scaler = StandardScaler()
    # X_train_scaled = scaler.fit_transform(X_train)
    # X_test_scaled = scaler.transform(X_test)
    
    # Save scaler
    models_dir = Path(__file__).parent.parent / "models"
    models_dir.mkdir(exist_ok=True)
    # joblib.dump(scaler, models_dir / "scaler.pkl")
    # logger.info("✅ Scaler saved")
    
    # Initialize Model Registry
    registry = ModelRegistry(db)
    
    results = []
    
    # ================================================================
    # MODEL 1: RANDOM FOREST (Baseline - Ensemble)
    # ================================================================
    logger.info("\n" + "=" * 70)
    logger.info("MODEL 1: Random Forest (Ensemble Baseline)")
    logger.info("=" * 70)
    
    from sklearn.ensemble import RandomForestRegressor
    rf_model = RandomForestRegressor(
        n_estimators=200,
        max_depth=8,
        min_samples_split=15,
        min_samples_leaf=8,
        random_state=42,
        n_jobs=-1
    )
    
    rf_model.fit(X_train, y_train)
    y_train_pred_rf = rf_model.predict(X_train)
    y_test_pred_rf = rf_model.predict(X_test)
    
    rf_metrics = {
        "r2_train": r2_score(y_train, y_train_pred_rf),
        "r2_test": r2_score(y_test, y_test_pred_rf),
        "mae_train": mean_absolute_error(y_train, y_train_pred_rf),
        "mae_test": mean_absolute_error(y_test, y_test_pred_rf),
        "rmse_train": np.sqrt(mean_squared_error(y_train, y_train_pred_rf)),
        "rmse_test": np.sqrt(mean_squared_error(y_test, y_test_pred_rf))
    }
    
    registry.register_model(
        model_name="Random Forest",
        model=rf_model,
        metrics=rf_metrics,
        feature_names=feature_names,
        hyperparameters=rf_model.get_params()
    )
    
    results.append({
        "Model": "Random Forest",
        "R2 (Train)": rf_metrics["r2_train"],
        "R2 (Test)": rf_metrics["r2_test"],
        "MAE (Train)": rf_metrics["mae_train"],
        "MAE (Test)": rf_metrics["mae_test"],
        "RMSE (Train)": rf_metrics["rmse_train"],
        "RMSE (Test)": rf_metrics["rmse_test"]
    })
    
    logger.info(f"  ✅ R2 Test: {rf_metrics['r2_test']:.4f}, MAE: {rf_metrics['mae_test']:.4f}, RMSE: {rf_metrics['rmse_test']:.4f}")
    
    # ================================================================
    # MODEL 2: XGBOOST (Gradient Boosting - Fast & Powerful)
    # ================================================================
    logger.info("\n" + "=" * 70)
    logger.info("MODEL 2: XGBoost (Gradient Boosting)")
    logger.info("=" * 70)
    
    xgb_model = XGBRegressor(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.05,
        min_child_weight=5,
        gamma=0.5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=1.0,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=30,
        objective="reg:squarederror",
        eval_metric="rmse",
    )
    
    xgb_model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )
    
    y_train_pred_xgb = xgb_model.predict(X_train)
    y_test_pred_xgb = xgb_model.predict(X_test)
    
    xgb_metrics = {
        "r2_train": r2_score(y_train, y_train_pred_xgb),
        "r2_test": r2_score(y_test, y_test_pred_xgb),
        "mae_train": mean_absolute_error(y_train, y_train_pred_xgb),
        "mae_test": mean_absolute_error(y_test, y_test_pred_xgb),
        "rmse_train": np.sqrt(mean_squared_error(y_train, y_train_pred_xgb)),
        "rmse_test": np.sqrt(mean_squared_error(y_test, y_test_pred_xgb))
    }
    
    registry.register_model(
        model_name="XGBoost",
        model=xgb_model,
        metrics=xgb_metrics,
        feature_names=feature_names,
        hyperparameters=xgb_model.get_params()
    )
    
    results.append({
        "Model": "XGBoost",
        "R2 (Train)": xgb_metrics["r2_train"],
        "R2 (Test)": xgb_metrics["r2_test"],
        "MAE (Train)": xgb_metrics["mae_train"],
        "MAE (Test)": xgb_metrics["mae_test"],
        "RMSE (Train)": xgb_metrics["rmse_train"],
        "RMSE (Test)": xgb_metrics["rmse_test"]
    })
    
    logger.info(f"  ✅ R2 Test: {xgb_metrics['r2_test']:.4f}, MAE: {xgb_metrics['mae_test']:.4f}, RMSE: {xgb_metrics['rmse_test']:.4f}")
    
    # ================================================================
    # MODEL 3: GRADIENT BOOSTING (Ensemble - Advanced)
    # ================================================================
    logger.info("\n" + "=" * 70)
    logger.info("MODEL 3: Gradient Boosting (Advanced Ensemble)")
    logger.info("=" * 70)
    
    from sklearn.ensemble import GradientBoostingRegressor
    
    gb_model = GradientBoostingRegressor(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        min_samples_leaf=10,
        random_state=42
    )
    
    gb_model.fit(X_train, y_train)
    y_train_pred_gb = gb_model.predict(X_train)
    y_test_pred_gb = gb_model.predict(X_test)
    
    gb_metrics = {
        "r2_train": r2_score(y_train, y_train_pred_gb),
        "r2_test": r2_score(y_test, y_test_pred_gb),
        "mae_train": mean_absolute_error(y_train, y_train_pred_gb),
        "mae_test": mean_absolute_error(y_test, y_test_pred_gb),
        "rmse_train": np.sqrt(mean_squared_error(y_train, y_train_pred_gb)),
        "rmse_test": np.sqrt(mean_squared_error(y_test, y_test_pred_gb))
    }
    
    registry.register_model(
        model_name="Gradient Boosting",
        model=gb_model,
        metrics=gb_metrics,
        feature_names=feature_names,
        hyperparameters=gb_model.get_params()
    )
    
    results.append({
        "Model": "Gradient Boosting",
        "R2 (Train)": gb_metrics["r2_train"],
        "R2 (Test)": gb_metrics["r2_test"],
        "MAE (Train)": gb_metrics["mae_train"],
        "MAE (Test)": gb_metrics["mae_test"],
        "RMSE (Train)": gb_metrics["rmse_train"],
        "RMSE (Test)": gb_metrics["rmse_test"]
    })
    
    logger.info(f"  ✅ R2 Test: {gb_metrics['r2_test']:.4f}, MAE: {gb_metrics['mae_test']:.4f}, RMSE: {gb_metrics['rmse_test']:.4f}")
    
    # ================================================================
    # RESULTS & COMPARISON
    # ================================================================
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("R2 (Test)", ascending=False)
    results_df.to_csv(models_dir / "model_comparison.csv", index=False)
    
    logger.info("\n" + "=" * 70)
    logger.info("MODEL TRAINING RESULTS - COMPARISON")
    logger.info("=" * 70)
    logger.info(f"\n{results_df.to_string(index=False)}")
    
    # Get best model
    best_model_doc = registry.get_best_model()
    logger.info(f"\n🏆 Best Model: {best_model_doc['model_name']}")
    logger.info(f"   R² Score: {best_model_doc['metrics']['r2_test']:.4f}")
    logger.info(f"   MAE: {best_model_doc['metrics']['mae_test']:.4f}")
    logger.info(f"   RMSE: {best_model_doc['metrics']['rmse_test']:.4f}")
    
    registry.close()
    
    logger.info("=" * 70)
    logger.info("✅ MODEL TRAINING COMPLETE - 3 MODELS TRAINED & REGISTERED")
    logger.info("=" * 70)


if __name__ == "__main__":
    train_models()
