from config.db import get_db

db = get_db()
models = list(db['model_registry'].find({}))
for m in models:
    if m['model_name'] == 'Gradient Boosting':
        features = m.get('feature_names', [])
        print(f"All {len(features)} features:")
        for i, f in enumerate(features, 1):
            print(f"{i:2d}. {f}")
        break
