"""
Model Registry: Store and manage trained models in MongoDB
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import joblib
import logging
from datetime import datetime, timezone
from config.db import get_db
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelRegistry:
    """Model Registry for tracking ML models"""
    def __init__(self, db=None):
        self.db = db if db is not None else get_db()
        self.registry_collection = self.db["model_registry"]
        self.models_dir = Path(__file__).parent.parent / "models"
        self.models_dir.mkdir(exist_ok=True)

    def register_model(self, model_name, model, metrics, feature_names, hyperparameters=None):
        model_path = self.models_dir / f"{model_name}.pkl"
        joblib.dump(model, model_path)
        model_doc = {
            "model_name": model_name,
            "model_path": str(model_path),
            "metrics": metrics,
            "feature_names": feature_names,
            "hyperparameters": hyperparameters or {},
            "registered_at": datetime.now(timezone.utc),
            "status": "active"
        }
        self.registry_collection.update_one(
            {"model_name": model_name},
            {"$set": model_doc},
            upsert=True
        )
        logger.info(f"✅ Registered model: {model_name} | R2: {metrics.get('r2_test', 0):.4f}")

    def get_model(self, model_name):
        model_doc = self.registry_collection.find_one({"model_name": model_name})
        if not model_doc:
            logger.error(f"Model {model_name} not found in registry")
            return None
        model_path = Path(model_doc["model_path"])
        if not model_path.exists():
            logger.error(f"Model file not found: {model_path}")
            return None
        model = joblib.load(model_path)
        logger.info(f"Loaded model: {model_name}")
        return model, model_doc

    def get_best_model(self, metric="r2_test"):
        models = list(self.registry_collection.find({"status": "active"}))
        if not models:
            logger.warning("No models found in registry")
            return None
        best_model = max(models, key=lambda x: x["metrics"].get(metric, 0))
        logger.info(f"Best model: {best_model['model_name']} | {metric}: {best_model['metrics'].get(metric, 0):.4f}")
        return best_model

    def list_models(self):
        models = list(self.registry_collection.find({}, {"_id": 0}))
        return models

    def close(self):
        pass  # No-op for persistent connection

if __name__ == "__main__":
    registry = ModelRegistry()
    models = registry.list_models()
    print(f"\nRegistered Models: {len(models)}")
    for model in models:
        print(f"  - {model['model_name']}: R2={model['metrics'].get('r2_test', 0):.4f}")
    registry.close()
