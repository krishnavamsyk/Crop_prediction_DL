import streamlit as st
import geocoder
import requests
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import tensorflow as tf
import numpy as np
import base64
import time
import joblib   
from streamlit_option_menu import option_menu
import torch
from ultralytics import YOLO
import tempfile
from PIL import Image
import base64

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
    if response:
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


st.title('Weather Forecast and Crop Precautions')
st.write('')
st.write('')

# Show loading spinner while detecting location
with st.spinner("Detecting your location..."):
    location, city, country = get_user_location()
    time.sleep(2)  # Optional: simulate loading delay for demonstration purposes

# Get user location
location, city, country = get_user_location()

if location:
    st.success(f"Detected Location: **{city}, {country}** (Lat: {location[0]}, Lon: {location[1]})")
    st.write('')
    st.write('')
    # Fetch and display weather forecast
    forecast_df = get_weather_forecast(location[0], location[1])
    plot_weather_forecast(forecast_df)

    # Calculate averages and give precautions
    st.write('')  # Adding space for better readability
    st.write('')
    avg_temp,avg_humidity,avg_precipitation,avg_windspeed=calculate_averages_and_precautions(forecast_df,5)
    # Display average values
    st.subheader('5-Day Average Weather Forecast')
    st.write('')
    st.write(f"**Average Temperature:** {avg_temp:.2f}°C")
    st.write(f"**Average Humidity:** {avg_humidity:.2f}%")
    st.write(f"**Average Precipitation:** {avg_precipitation:.2f} mm")
    st.write(f"**Average Windspeed:** {avg_windspeed:.2f} m/s") 
    st.write('')
    
    # Display crop precautions based on averages
    st.subheader('Recommended Precautions')
    st.write('')
    if avg_temp > 35:
        st.write("⚠️ **Precaution:** High average temperature. Consider irrigating crops to prevent heat stress.")
    elif avg_temp < 15:
        st.write("⚠️ **Precaution:** Low average temperature. Cover sensitive crops to protect them from cold damage.")
    
    if avg_humidity > 80:
        st.write("⚠️ **Precaution:** High humidity. Monitor for fungal diseases such as mildew and rust.")
    
    if avg_precipitation > 5:
        st.write("⚠️ **Precaution:** Heavy rainfall expected. Ensure proper drainage to avoid waterlogging and root rot.")
    elif avg_precipitation == 0:
        st.write("⚠️ **Precaution:** No rainfall expected. Consider irrigation to maintain soil moisture.")
    
    if avg_windspeed > 15:
        st.write("-High winds can damage crops. Secure loose plants.\n")
    
    st.write('')
    st.write('')   
else:
    st.write("Sorry, could not determine your location. Please try again.")
