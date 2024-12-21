#importing libraries

import streamlit as st
import geocoder
import requests
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import tensorflow as tf
import numpy as np  
from streamlit_option_menu import option_menu
from ultralytics import YOLO
import tempfile
from PIL import Image
from English_code_ppl import Englsh
from Hindi_code_ppl import Hindi

#functions
 #functions
# Function to automatically get user location based on IP
def get_user_location():
    g = geocoder.ip('me')
    if g.ok:
        location = g.latlng  # Get latitude and longitude
        city = g.city if g.city else "Unknown"  # Safeguard in case of missing city
        country = g.country if g.country else "Unknown"
        return location, city, country
    else:
        st.warning("Unable to detect location. Using default location coordinates.")
        return [0, 0], "Unknown", "Unknown"

# Function to fetch weather forecast from OpenWeatherMap API
def get_weather_forecast(lat, lon):
    API_KEY = '7bd12cbbf7283b1b6d3ae9f67c201bdf'  # Replace with your OpenWeatherMap API key
    URL = f"http://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={API_KEY}&units=metric"
    response = requests.get(URL)
    if response.status_code == 200:
        data = response.json()
        forecast_data = []
        for forecast in data.get('list', []):  # Use .get() to handle missing 'list'
            date_time = datetime.fromtimestamp(forecast['dt'])
            temp = forecast['main']['temp']
            humidity = forecast['main']['humidity']
            precipitation = forecast.get('rain', {}).get('3h', 0)
            windspeed = forecast['wind']['speed']
            forecast_data.append({
                'datetime': date_time,
                'temperature': temp,
                'humidity': humidity,
                'precipitation': precipitation,
                'windspeed': windspeed
            })
        return pd.DataFrame(forecast_data)
    else:
        st.error("Failed to fetch weather data.")
        return pd.DataFrame()


# Function to plot the weather forecast
def plot_weather_forecast(forecast_df):
    # Create subplots for temperature, humidity, and precipitation
    fig, (ax1, ax2, ax3,ax4) = plt.subplots(4, 1, figsize=(10, 12))
    
    # Plot temperature
    ax1.plot(forecast_df['datetime'], forecast_df['temperature'], color='tab:red', label='Temperature (°C)')
    ax1.set_xlabel('Date and Time')
    ax1.set_ylabel('Temperature (°C)', color='tab:red')
    ax1.set_title('Temperature Forecast', fontsize=16, fontweight='bold')
    ax1.tick_params(axis='y', labelcolor='tab:red')
    ax1.grid(True)

    # Plot humidity
    ax2.plot(forecast_df['datetime'], forecast_df['humidity'], color='tab:blue', label='Humidity (%)')
    ax2.set_xlabel('Date and Time')
    ax2.set_ylabel('Humidity (%)', color='tab:blue')
    ax2.set_title('Humidity Forecast', fontsize=16, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='tab:blue')
    ax2.grid(True)

    # Plot precipitation
    ax3.plot(forecast_df['datetime'], forecast_df['precipitation'], color='tab:green', label='Precipitation (mm)')
    ax3.set_xlabel('Date and Time')
    ax3.set_ylabel('Precipitation (mm)', color='tab:green')
    ax3.set_title('Precipitation Forecast', fontsize=16, fontweight='bold')
    ax3.tick_params(axis='y', labelcolor='tab:green')
    ax3.grid(True)

    # plot windspeed 
    ax4.plot(forecast_df['datetime'], forecast_df['windspeed'], color='tab:orange', label='Wind Speed (m/s)')
    ax4.set_xlabel('Date and Time')
    ax4.set_ylabel('Wind Speed (m/s)', color='tab:orange')
    ax4.set_title('Windspeed Forecast', fontsize=16, fontweight='bold')
    ax4.tick_params(axis='y', labelcolor='tab:orange')
    ax4.grid(True)

    plt.subplots_adjust(hspace=0.5)
    st.pyplot(fig)

# Function to calculate 5-day averages and give precautions
def calculate_averages_and_precautions(forecast_df,x):
    # Filter forecast for the next 5 days
    forecast_df['date'] = forecast_df['datetime'].dt.date
    next_days = forecast_df.groupby('date').mean().head(x)
    
    # Calculate averages
    avg_temp = next_days['temperature'].mean()
    avg_humidity = next_days['humidity'].mean()
    avg_precipitation = next_days['precipitation'].mean()
    avg_windspeed = next_days['windspeed'].mean()
    return avg_temp,avg_humidity,avg_precipitation,avg_windspeed



def model_prediction(test_image):
    global classes
    classes = [
        'Apple__Apple_scab', 'Apple_Black_rot', 'Apple_healthy', 'Corn_Blight', 'Corn_Common_Rust',
        'Corn_Gray_Leaf_Spot', 'Corn_Healthy', 'Cotton_Healthy', 'Cotton_bacterial_blight', 'Cotton_curl_virus',
        'Grape_Black_rot', 'Grape_Leaf_blight(Isariopsis_Leaf_Spot)', 'Grape__healthy', 
        'Pepper,_bell_Bacterial_spot', 'Pepper,_bell_healthy', 'Potato_Early_blight', 
        'Potato_Late_blight', 'Potato_healthy', 'Rice_Healthy', 'Rice_bacterial_leaf_blight', 
        'Rice_brown_spot', 'Rice_leaf_blast', 'Sugarcan_Mosaic', 'Sugarcane_Healthy', 
        'Sugarcane_RedRot', 'Sugarcane_Rust', 'Sugarcane_Yellow', 'Tomato_Bacterial_spot', 
        'Tomato_Early_blight', 'Tomato_Late_blight', 'Tomato_Septoria_leaf_spot', 'Tomato__healthy', 
        'Wheat_Brown_rust', 'Wheat_Healthy', 'Wheat_Loose_Smut', 'Wheat_Yellow_rust'
    ]

    model1 = YOLO(r'Models\best.pt')  # Load last custom model
    
    # Save the uploaded image to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
        # Convert to PIL image for compatibility
        image = Image.open(test_image)
        image = image.convert('RGB')  # Ensure 3 channels
        image.save(tmp_file.name)
        tmp_image_path = tmp_file.name

    # Perform inference
    results1 = model1(tmp_image_path)
    probs1 = results1[0].probs.data.tolist()
    return classes[np.argmax(probs1)]

# Language selector
selected_language = st.sidebar.selectbox("SELECT YOUR LANGUAGE:", options=['English','हिंदी'])


if selected_language=='English':
    Englsh()
else:
    Hindi()