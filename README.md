# AQI Predictor - 10Pearls

[![Live Dashboard](https://img.shields.io/badge/Live%20App-Streamlit-green?logo=streamlit)](https://aqi-predictor-10pearls-fyvgvappmxdmqkwjnakxme6.streamlit.app/)

**Live Dashboard:** [https://aqi-predictor-10pearls-fyvgvappmxdmqkwjnakxme6.streamlit.app/](https://aqi-predictor-10pearls-fyvgvappmxdmqkwjnakxme6.streamlit.app/)

## Overview
This project is an end-to-end Air Quality Index (AQI) prediction system for Sukkur, built with Python, MongoDB, and Streamlit. It fetches weather and AQI data, preprocesses and augments it, trains machine learning models, and provides a dashboard for visualization and prediction.

## Features
- Fetches historical and current weather and AQI data using APIs
- Stores raw and processed data in MongoDB
- Data preprocessing: cleaning, handling missing values, outlier removal, feature engineering
- Model training and comparison (XGBoost, TensorFlow, Keras, scikit-learn)
- Interactive dashboard built with Streamlit and Plotly
- CI/CD pipeline with GitHub Actions for automated training and testing

## Project Structure
```
AQI-predictor-10Pearls/
├── app/
│   └── dashboard.py         # Streamlit dashboard
├── config/
│   └── db.py                # MongoDB connection
├── models/
│   └── model_comparison.csv # Model results
├── notebooks/
│   └── EDA_Analysis.ipynb   # Exploratory Data Analysis
├── scripts/
│   ├── data_preprocessing.py
│   ├── feature_store.py
│   ├── fetch_AQI.py
│   ├── fetch_weather.py
│   ├── model_registry.py
│   ├── predict_aqi.py
│   └── train_models.py
├── requirements.txt         # Python dependencies
├── .env                     # Environment variables (not for public repos)
├── .github/workflows/aqi-pipeline.yml # CI/CD pipeline
├── main.py                  # Pipeline entry point
└── README.md                # Project documentation
```

## Setup & Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/salwaaliakbar/AQI-predictor-10Pearls.git
   cd AQI-predictor-10Pearls
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up your `.env` file with required secrets and config (see example below).
4. Run the main pipeline:
   ```bash
   python main.py
   ```
5. Launch the dashboard:
   ```bash
   streamlit run app/dashboard.py
   ```

## Environment Variables (.env Example)
```
MONGODB_URI=your_mongodb_uri
DB_NAME=aqi_db
COLL_WEATHER=raw_weather
COLL_AQI=raw_aqi
COLL_FEATURES=feature_store
COLL_MODELS=model_registry
CITY=Sukkur
LAT=27.7058
LON=68.8574
OPENWEATHER_API_KEY=your_openweather_api_key
```

## CI/CD
- Automated with GitHub Actions (`.github/workflows/aqi-pipeline.yml`)
- Runs on push, schedule, or manual dispatch
- Installs dependencies, runs tests, and trains models

## Deployment
- Recommended: Streamlit Community Cloud for dashboard hosting
- Add secrets in TOML format in Streamlit Cloud settings

## Usage
- View and interact with AQI predictions and visualizations on the dashboard
- Models are retrained and updated automatically via CI/CD

## License
This project is for educational and demonstration purposes at 10Pearls.

---
For questions or contributions, open an issue or pull request on GitHub.