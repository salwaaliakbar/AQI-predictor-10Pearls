"""
AQI Dashboard - Streamlit Visualization App
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
from dotenv import load_dotenv
from pathlib import Path
import sys
import joblib
import numpy as np

# Add project root to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.db import get_db

load_dotenv()

st.set_page_config(page_title="AQI Prediction Dashboard", page_icon="🌍", layout="wide")

# Custom CSS
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    color: #1f7b4;
    text-align: center;
    padding: 10px;
}
.metric-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 20px;
    border-radius: 10px;
    color: white;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
}
[data-testid="stSidebar"],
[data-testid="stSidebar"] *,
[data-testid="stSidebar"] .st-radio label,
[data-testid="stSidebar"] .stMarkdown,
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] span {
    color: #ffffff !important;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_db_connection():
    """Get persistent MongoDB connection from config/db.py"""
    return get_db()

@st.cache_data(ttl=600)
def load_data(collection_name):
    """Load data from MongoDB"""
    db = get_db_connection()
    df = pd.DataFrame(list(db[collection_name].find({}, {"_id": 0})))
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

def get_aqi_category(aqi):
    """Get AQI category and color"""
    if aqi <= 50:
        return "Good", "#039303"
    elif aqi <= 100:
        return "Moderate", "#D9D902"
    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups", "#FF7E00"
    elif aqi <= 200:
        return "Unhealthy", "#FF0000"
    elif aqi <= 300:
        return "Very Unhealthy", "#8F3F97"
    else:
        return "Hazardous", "#7E0023"

def main():
    # Header
    st.markdown('<h1 class="main-header">🌍 AQI Prediction Dashboard - Sukkur</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Overview", "Data Analysis", "Model Performance", "Predictions"])
    
    if page == "Overview":
        show_overview()
    elif page == "Data Analysis":
        show_data_analysis()
    elif page == "Model Performance":
        show_model_performance()
    else:
        show_predictions()

def show_overview():
    """Show overview page"""
    st.header("📊 System Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    # Load latest data
    try:
        raw_aqi = load_data("raw_aqi")
        feature_store = load_data("feature_store")
        
        with col1:
            st.metric("Total Records", len(raw_aqi))
        
        with col2:
            st.metric("Features Stored", len(feature_store))
        
        with col3:
            if not raw_aqi.empty and 'us_aqi' in raw_aqi.columns:
                latest_aqi = raw_aqi.sort_values('timestamp').iloc[-1]['us_aqi']
                category, color = get_aqi_category(latest_aqi)
                st.metric("Latest AQI", f"{latest_aqi:.0f}", category)
        
        with col4:
            db = get_db_connection()
            model_count = db["model_registry"].count_documents({})
            st.metric("Trained Models", model_count)
        
        # Latest AQI Trend
        if not raw_aqi.empty:
            st.subheader("📈 AQI Trend (Last 7 Days)")
            recent = raw_aqi.sort_values('timestamp').tail(168)  # Last 7 days hourly
            
            fig = px.line(recent, x='timestamp', y='us_aqi', 
                         title='US AQI Over Time',
                         labels={'us_aqi': 'AQI', 'timestamp': 'Time'})
            fig.add_hline(y=50, line_dash="dash", line_color="green", annotation_text="Good")
            fig.add_hline(y=100, line_dash="dash", line_color="yellow", annotation_text="Moderate")
            fig.add_hline(y=150, line_dash="dash", line_color="orange", annotation_text="Unhealthy")
            st.plotly_chart(fig, use_container_width=True)
    
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")

def show_data_analysis():
    """Show data analysis page"""
    st.header("🔍 Data Analysis")
    
    try:
        preprocessed = load_data("preprocessed_data")
        
        if preprocessed.empty:
            st.warning("No preprocessed data available")
            return
        
        # Correlation heatmap
        st.subheader("Correlation Heatmap")
        numeric_cols = preprocessed.select_dtypes(include=['number']).columns
        corr = preprocessed[numeric_cols].corr()
        
        fig = px.imshow(corr, text_auto=True, aspect="auto",
                       title="Feature Correlations")
        st.plotly_chart(fig, use_container_width=True)
        
        # Distribution plots
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("PM2.5 Distribution")
            fig = px.histogram(preprocessed, x='pm2_5', nbins=50,
                              title="PM2.5 Concentration Distribution")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Temperature Distribution")
            fig = px.histogram(preprocessed, x='temp', nbins=50,
                              title="Temperature Distribution")
            st.plotly_chart(fig, use_container_width=True)
        
        # Time series
        st.subheader("Weather & AQI Time Series")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=preprocessed['timestamp'], y=preprocessed['temp'],
                                name='Temperature', yaxis='y'))
        fig.add_trace(go.Scatter(x=preprocessed['timestamp'], y=preprocessed['pm2_5'],
                                name='PM2.5', yaxis='y2'))
        
        fig.update_layout(
            yaxis=dict(title="Temperature (°C)"),
            yaxis2=dict(title="PM2.5", overlaying='y', side='right')
        )
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error: {str(e)}")

def show_model_performance():
    """Show model performance page"""
    st.header("🤖 Model Performance")
    
    try:
        # Load model comparison
        models_dir = Path(__file__).parent.parent / "models"
        comparison_file = models_dir / "model_comparison.csv"
        
        if comparison_file.exists():
            df = pd.read_csv(comparison_file)
            
            st.subheader("Model Comparison - All Metrics")
            st.dataframe(df)
            
            # Create visualizations for all metrics
            st.subheader("Performance Metrics Comparison")
            
            # R2 Score Comparison (Train vs Test)
            if 'R2 (Train)' in df.columns and 'R2 (Test)' in df.columns:
                r2_df = df[['Model', 'R2 (Train)', 'R2 (Test)']].melt(
                    id_vars='Model', var_name='Dataset', value_name='R2 Score'
                )
                fig = px.bar(
                    r2_df,
                    x='Model',
                    y='R2 Score',
                    color='Dataset',
                    barmode='group',
                    title='R² Score Comparison (Train vs Test)',
                    color_discrete_map={'R2 (Train)': '#3498db', 'R2 (Test)': '#e74c3c'}
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # MAE Comparison (Train vs Test)
            if 'MAE (Train)' in df.columns and 'MAE (Test)' in df.columns:
                mae_df = df[['Model', 'MAE (Train)', 'MAE (Test)']].melt(
                    id_vars='Model', var_name='Dataset', value_name='MAE'
                )
                fig = px.bar(
                    mae_df,
                    x='Model',
                    y='MAE',
                    color='Dataset',
                    barmode='group',
                    title='Mean Absolute Error (Train vs Test)',
                    color_discrete_map={'MAE (Train)': '#2ecc71', 'MAE (Test)': '#f39c12'}
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # RMSE Comparison (Train vs Test)
            if 'RMSE (Train)' in df.columns and 'RMSE (Test)' in df.columns:
                rmse_df = df[['Model', 'RMSE (Train)', 'RMSE (Test)']].melt(
                    id_vars='Model', var_name='Dataset', value_name='RMSE'
                )
                fig = px.bar(
                    rmse_df,
                    x='Model',
                    y='RMSE',
                    color='Dataset',
                    barmode='group',
                    title='Root Mean Squared Error (Train vs Test)',
                    color_discrete_map={'RMSE (Train)': '#9b59b6', 'RMSE (Test)': '#e67e22'}
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No model comparison data available. Run training first!")
        
    except Exception as e:
        st.error(f"Error: {str(e)}")

def show_predictions():
    """Show predictions page"""
    st.header("🔮 AQI Predictions")
    
    models_dir = Path(__file__).parent.parent / "models"
    scaler_path = models_dir / "scaler.pkl"
    
    # Check if models are trained
    db = get_db_connection()
    models = list(db["model_registry"].find({}, {"_id": 0}))
    
    if not models or not scaler_path.exists():
        st.warning("⚠️ No trained models found! Please train models first using the pipeline.")
        st.code("python scripts/train_models.py", language="bash")
        return
    
    st.success(f"✅ {len(models)} trained models available")
    
    # Tabs for current and future predictions
    tab1, tab2 = st.tabs(["📊 Current Prediction", "📅 3-Day Forecast"])
    
    with tab1:
        show_current_prediction(models, scaler_path)
    
    with tab2:
        show_forecast(models, scaler_path)


def show_current_prediction(models, scaler_path):
    """Show current AQI prediction from pre-computed predictions"""
    models_dir = Path(__file__).parent.parent / "models"
    
    # Load pre-computed predictions
    try:
        predictions_df = load_data("aqi_predictions")
        
        if predictions_df.empty:
            st.warning("⚠️ No predictions available. Run the pipeline to generate predictions.")
            st.code("python main.py", language="bash")
            return
        
        # Get current prediction (closest to current time)
        predictions_df = predictions_df.sort_values('timestamp')
        now = pd.Timestamp.now().tz_localize(None)  # Remove timezone for comparison
        predictions_df['time_diff'] = abs(predictions_df['timestamp'] - now)
        current = predictions_df.sort_values('time_diff').iloc[0]
        
        # Display current prediction
        st.subheader("📍 Current AQI Prediction for Sukkur")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Timestamp", str(current['timestamp']))
            st.metric("Temperature", f"{current['temp']:.1f}°C")
            st.metric("Humidity", f"{current['humidity']:.0f}%")
        
        with col2:
            st.metric("Wind Speed", f"{current['wind_speed']:.1f} km/h")
            st.metric("Pressure", f"{current['pressure']:.1f} hPa")
            st.metric("Cloud Cover", f"{current['clouds']:.0f}%")
        
        with col3:
            st.metric("Predicted AQI", f"{current['predicted_aqi']:.1f}")
            category, color = get_aqi_category(current['predicted_aqi'])
            text_color = "#111" if color in ("#00E400", "#FFFF00") else "#fff"
            st.markdown(
                f'<div style="background:linear-gradient(135deg,{color} 0%, {color}cc 100%); padding:20px; border-radius:10px; text-align:center; font-size:20px; color:{text_color}; box-shadow:0 4px 12px rgba(0,0,0,0.15)"><b>{category}</b></div>',
                unsafe_allow_html=True
            )
            st.metric("Model Used", current['model_name'])
        
        # Show recent predictions (last 24 hours)
        st.subheader("📈 Recent Predictions (Last 24 Hours)")
        recent = predictions_df.head(24)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=recent['timestamp'],
            y=recent['predicted_aqi'],
            mode='lines+markers',
            name='Predicted AQI',
            line=dict(color='#1f77b4', width=3),
            marker=dict(size=6)
        ))
        
        # Add AQI thresholds
        fig.add_hline(y=50, line_dash="dash", line_color="green", annotation_text="Good")
        fig.add_hline(y=100, line_dash="dash", line_color="yellow", annotation_text="Moderate")
        fig.add_hline(y=150, line_dash="dash", line_color="orange", annotation_text="Unhealthy")
        
        fig.update_layout(
            title='AQI Predictions - Last 24 Hours',
            xaxis_title='Time',
            yaxis_title='Predicted AQI',
            hovermode='x unified',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Weather conditions
        with st.expander("🌤️ View Current Weather Conditions"):
            weather_data = {
                'Temperature': f"{current['temp']:.1f}°C",
                'Humidity': f"{current['humidity']:.0f}%",
                'Pressure': f"{current['pressure']:.1f} hPa",
                'Wind Speed': f"{current['wind_speed']:.1f} km/h",
                'Cloud Cover': f"{current['clouds']:.0f}%"
            }
            st.json(weather_data)
            
    except Exception as e:
        st.error(f"Error making predictions: {str(e)}")
        import traceback
        st.code(traceback.format_exc())


def show_forecast(models, scaler_path):
    """Show 3-day forecast from pre-computed predictions"""
    st.subheader("📅 3-Day AQI Forecast")
    
    st.info("✅ Automated predictions generated by trained models using forecast weather data.")
    
    try:
        # Load pre-computed predictions
        predictions_df = load_data("aqi_predictions")
        
        if predictions_df.empty:
            st.warning("⚠️ No predictions available. Run the pipeline to generate predictions.")
            st.code("python main.py", language="bash")
            return
        
        predictions_df = predictions_df.sort_values('timestamp')
        
        # Display summary metrics
        st.subheader(f"Using {predictions_df.iloc[0]['model_name']} (Best Model)")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            avg_aqi = predictions_df['predicted_aqi'].mean()
            st.metric("Average AQI", f"{avg_aqi:.1f}")
        with col2:
            max_aqi = predictions_df['predicted_aqi'].max()
            st.metric("Max AQI", f"{max_aqi:.1f}")
        with col3:
            min_aqi = predictions_df['predicted_aqi'].min()
            st.metric("Min AQI", f"{min_aqi:.1f}")
        with col4:
            most_common = predictions_df['aqi_category'].mode()[0]
            st.metric("Most Common", most_common)
            
            # Plot forecast with color-coded AQI zones
            fig = go.Figure()
        
        # Add AQI category background zones
        aqi_zones = [
            (0, 50, '#00E400', 'Good'),
            (50, 100, '#FFFF00', 'Moderate'),
            (100, 150, '#FF7E00', 'Unhealthy for Sensitive'),
            (150, 200, '#FF0000', 'Unhealthy'),
            (200, 300, '#8F3F97', 'Very Unhealthy'),
            (300, 500, '#7E0023', 'Hazardous')
        ]
        
        for y0, y1, color, label in aqi_zones:
            fig.add_hrect(
                y0=y0, y1=y1,
                fillcolor=color,
                opacity=0.15,
                line_width=0,
                annotation_text=label,
                annotation_position="left",
                annotation=dict(font_size=10, font_color=color)
            )
        
        # Add forecast line
        fig.add_trace(go.Scatter(
            x=predictions_df['timestamp'],
            y=predictions_df['predicted_aqi'],
            mode='lines+markers',
            name='Forecasted AQI',
            line=dict(color='#000080', width=3),
            marker=dict(size=5, color='#000080'),
            hovertemplate='<b>%{x}</b><br>AQI: %{y:.1f}<extra></extra>'
        ))
        
        # Add threshold lines
        fig.add_hline(y=50, line_dash="dot", line_color="green", line_width=1.5)
        fig.add_hline(y=100, line_dash="dot", line_color="#FFD700", line_width=1.5)
        fig.add_hline(y=150, line_dash="dot", line_color="orange", line_width=1.5)
        fig.add_hline(y=200, line_dash="dot", line_color="red", line_width=1.5)
        
        fig.update_layout(
            title='72-Hour AQI Forecast with Health Categories',
            xaxis_title='Timestamp',
            yaxis_title='Predicted AQI',
            hovermode='x unified',
            height=550,
            yaxis=dict(range=[0, min(350, predictions_df['predicted_aqi'].max() + 50)])
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Daily summary
        st.subheader("📊 Daily Summary")
        
        predictions_df['Date'] = predictions_df['timestamp'].dt.date
        daily_summary = predictions_df.groupby('Date').agg({
            'predicted_aqi': ['mean', 'min', 'max']
        }).round(1)
        daily_summary.columns = ['Average AQI', 'Min AQI', 'Max AQI']
        daily_summary['Category'] = daily_summary['Average AQI'].apply(
            lambda x: get_aqi_category(x)[0]
        )
        
        st.dataframe(daily_summary.style.background_gradient(
            subset=['Average AQI'], cmap='RdYlGn_r'
        ))
        
        # Hourly breakdown
        with st.expander("🕐 View Hourly Forecast"):
            hourly_display = predictions_df[['timestamp', 'predicted_aqi', 'aqi_category']].copy()
            hourly_display['timestamp'] = hourly_display['timestamp'].dt.strftime('%Y-%m-%d %H:%00')
            hourly_display.columns = ['Timestamp', 'Predicted AQI', 'Category']
            st.dataframe(
                hourly_display.style.format({'Predicted AQI': '{:.1f}'}).background_gradient(
                    subset=['Predicted AQI'], cmap='RdYlGn_r'
                ),
                height=400
            )
        
        # Distribution
        col1, col2 = st.columns(2)
        
        with col1:
            # AQI distribution
            fig = px.histogram(
                predictions_df,
                x='predicted_aqi',
                nbins=30,
                title='AQI Distribution (Next 3 Days)',
                color_discrete_sequence=['skyblue']
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Category breakdown
            category_counts = predictions_df['aqi_category'].value_counts()
            fig = px.pie(
                values=category_counts.values,
                names=category_counts.index,
                title='AQI Category Distribution',
                color_discrete_sequence=px.colors.sequential.RdBu_r
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Temperature Forecast
        st.markdown("---")
        st.subheader("🌡️ Temperature Forecast")
        
        if 'temp' in predictions_df.columns:
            temp_data = predictions_df['temp'].values
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Average Temp", f"{temp_data.mean():.1f}°C")
            with col2:
                st.metric("Max Temp", f"{temp_data.max():.1f}°C")
            with col3:
                st.metric("Min Temp", f"{temp_data.min():.1f}°C")
            
            # Temperature trend chart
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=predictions_df['timestamp'],
                y=temp_data,
                mode='lines+markers',
                name='Temperature',
                line=dict(color='#FF6347', width=2),
                marker=dict(size=4),
                fill='tozeroy',
                fillcolor='rgba(255,99,71,0.2)'
            ))
            fig.update_layout(
                title='Temperature Forecast (Next 3 Days)',
                xaxis_title='Timestamp',
                yaxis_title='Temperature (°C)',
                hovermode='x unified',
                height=350
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # AQI Category Information (Collapsible)
        st.markdown("---")
        with st.expander("📖 AQI Health Categories - Click to View Details"):
            aqi_info = [
                ("0-50", "Good", "#00E400", "Air quality is satisfactory, and air pollution poses little or no risk."),
                ("51-100", "Moderate", "#FFFF00", "Air quality is acceptable; however, there may be a risk for some people."),
                ("101-150", "Unhealthy for Sensitive Groups", "#FF7E00", "Members of sensitive groups may experience health effects."),
                ("151-200", "Unhealthy", "#FF0000", "Everyone may begin to experience health effects; sensitive groups at greater risk."),
                ("201-300", "Very Unhealthy", "#8F3F97", "Health alert: everyone may experience more serious health effects."),
                ("301+", "Hazardous", "#7E0023", "Health warnings of emergency conditions; entire population is likely to be affected.")
            ]
            
            # Create rows for each category
            for idx, (aqi_range, category, color, description) in enumerate(aqi_info):
                st.markdown(f"""
                <div style="background-color:{color}; padding:20px; border-radius:10px; margin-bottom:10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                    <div style="display:flex; align-items:center; justify-content:space-between;">
                        <div>
                            <h3 style="color:{'white' if idx >= 3 else 'black'}; margin:0; font-size:22px; font-weight:bold;">{category}</h3>
                            <h4 style="color:{'white' if idx >= 3 else 'black'}; margin:5px 0; font-size:16px;">AQI Range: {aqi_range}</h4>
                        </div>
                    </div>
                    <p style="color:{'white' if idx >= 3 else 'black'}; font-size:14px; margin:10px 0 0 0; line-height:1.5;">{description}</p>
                </div>
                """, unsafe_allow_html=True)
    
    except Exception as e:
        st.error(f"Error generating forecast: {str(e)}")
        import traceback
        st.code(traceback.format_exc())


if __name__ == "__main__":
    main()

