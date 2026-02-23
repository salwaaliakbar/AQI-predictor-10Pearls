from config.db import get_db
import pandas as pd

db = get_db()

# Check feature store statistics
fs = pd.DataFrame(list(db['feature_store'].find({}, {'_id': 0})))
print('FEATURE STORE (Training Data Used):')
print(f'Records: {len(fs)}')
if 'us_aqi' in fs.columns:
    print(f'AQI Mean: {fs["us_aqi"].mean():.2f}')
    print(f'AQI Min: {fs["us_aqi"].min():.2f}')
    print(f'AQI Max: {fs["us_aqi"].max():.2f}')
    print(f'AQI Std: {fs["us_aqi"].std():.2f}')
    print(f'Records with AQI >= 150: {len(fs[fs["us_aqi"] >= 150])}')
    print(f'Records with AQI < 100: {len(fs[fs["us_aqi"] < 100])}')
else:
    print('NO us_aqi column!')
    print(f'Available columns: {fs.columns.tolist()}')

# Check raw AQI
raw = pd.DataFrame(list(db['raw_aqi'].find({}, {'_id': 0})))
print('\n\nRAW AQI DATA:')
print(f'Records: {len(raw)}')
if 'us_aqi' in raw.columns:
    print(f'AQI Max: {raw["us_aqi"].max():.2f}')
    print(f'AQI >= 150: {len(raw[raw["us_aqi"] >= 150])} records')
    
# Check preprocessed data
preproc = pd.DataFrame(list(db['preprocessed_data'].find({}, {'_id': 0})))
print('\n\nPREPROCESSED DATA:')
print(f'Records: {len(preproc)}')
if 'us_aqi' in preproc.columns:
    print(f'AQI Max: {preproc["us_aqi"].max():.2f}')
    print(f'AQI >= 150: {len(preproc[preproc["us_aqi"] >= 150])} records')
