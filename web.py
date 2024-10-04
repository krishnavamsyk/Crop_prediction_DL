import streamlit as st
import geocoder
import requests
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime


# Function to automatically get user location based on IP
def get_user_location():
    g = geocoder.ip('me')
    if g.ok:
        location = g.latlng  # Get latitude and longitude
        city = g.city  # Get city name
        country = g.country  # Get country name
        return location, city, country
    else:
        return None, None, None

# Function to fetch weather forecast from OpenWeatherMap API
def get_weather_forecast(lat, lon):
    API_KEY = 'yourapikey'  # Replace with your OpenWeatherMap API key
    URL = f"http://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={API_KEY}&units=metric"
    response = requests.get(URL)
    data = response.json()
    
    forecast_data = []
    for forecast in data['list']:
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
def calculate_averages_and_precautions(forecast_df):
    # Filter forecast for the next 5 days
    forecast_df['date'] = forecast_df['datetime'].dt.date
    next_5_days = forecast_df.groupby('date').mean().head(5)
    
    # Calculate averages
    avg_temp = next_5_days['temperature'].mean()
    avg_humidity = next_5_days['humidity'].mean()
    avg_precipitation = next_5_days['precipitation'].mean()
    avg_windspeed = next_5_days['windspeed'].mean()

    # Display average values
    st.subheader('5-Day Average Weather Forecast')
    st.write('')
    st.write(f"**Average Temperature:** {avg_temp:.2f}°C")
    st.write(f"**Average Humidity:** {avg_humidity:.2f}%")
    st.write(f"**Average Precipitation:** {avg_precipitation:.2f} mm")
    st.write(f"**Average Windspeed:** {avg_windspeed:.2f} m/s") 
    st.write('')
    
    # Display crop precautions based on averages
    st.subheader('Recommended Crop Precautions')
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
        precautions += "- High winds can damage crops. Secure loose plants.\n"
    


#dashboard
st.sidebar.title("DASHBOARD")
mode=st.sidebar.selectbox("Select Page:",["Home","Disease-Recognition","5-Day forecast","About"])


if (mode=="Home"):
    # Title of the app
    st.markdown("<h1 style='text-align: center;'>🌾 GreenWatch: AI Crop Health Advisor</h1>", unsafe_allow_html=True)
    st.write("<h6 style='text-align: right;'>- By team AI-CRAFT</h6>", unsafe_allow_html=True)

    # Introduction and tagline
    st.write('')
    st.write('')
    st.write('')
    st.write('')
    st.subheader("Protect Your Crops, Secure Your Yield")
    st.write("""
    At **GreenWatch AI**, we harness the power of artificial intelligence to help farmers like you detect crop diseases early and take preventive measures before it's too late.
    """)

    # How It Works section
    st.header("🌱 How It Works")
    st.write("""
    1. **Upload an Image**: Simply take a picture of your crop showing signs of illness and upload it to the app.
    2. **AI Analysis**: Our advanced deep learning model will analyze the image and identify potential diseases affecting your crop.
    3. **Get Results & Guidance**: You’ll receive an immediate diagnosis along with detailed precautions and solutions to manage the disease.
    """)

    # Why Use  GreenWatch AI section
    st.header("🚜 Why Use GreenWatch AI?")
    st.write("""
    - **Fast & Accurate**: Get real-time predictions powered by state-of-the-art AI technology.
    - **User-Friendly**: Just upload an image, and let our AI do the rest—no technical expertise needed.
    - **Tailored Guidance**: Receive not just a diagnosis but actionable measures to prevent crop loss and improve yield.
    - **Supporting Farmers**: Our goal is to empower farmers with technology to make smarter decisions and secure their livelihood.
    """)

    #How to start
    st.header("❓ How to Start?")
    st.write("Go to the Disease Recognition tab in the side panel to start.")
    # Need Help section
    st.header("💡 Need Help?")
    st.write("""
    If you’re unsure or want more information, feel free to check our [FAQs](#) or contact us directly.
    """)


#About page
elif (mode=="About"):
    st.header("About")
    st.text("")
    
    st.markdown('''
The primary goal of your crop disease prediction and prevention system is to provide farmers with an AI-powered platform that can forecast potential disease outbreaks and suggest appropriate preventive measures and treatments. This system leverages real-time data such as crop images, environmental factors (temperature, humidity, soil moisture), and historical disease patterns to deliver actionable insights that can help farmers protect their crops and improve yields.

### How It Helps Farmers

1. **Early Detection of Disease:**
   - The system can identify early signs of diseases through image analysis and detect patterns of symptoms before they become visually obvious to farmers. Early intervention allows farmers to address the issue before it spreads and causes significant damage.
   
2. **Data-Driven Decision Making:**
   - By analyzing environmental conditions like temperature, humidity, and rainfall in conjunction with the type of crops, the system can predict when certain diseases are more likely to occur. This allows farmers to make informed decisions about treatments, irrigation, and even planting times.

3. **Minimization of Crop Losses:**
   - Early warnings about potential disease outbreaks enable farmers to take preventive actions, reducing crop losses. Disease-related crop damage is one of the major threats to food production, and timely interventions can save significant portions of a harvest.

4. **Cost Savings:**
   - The system can recommend targeted treatments, like specific pesticides or organic solutions, only when necessary. This can prevent overuse of chemicals and fertilizers, saving farmers money and reducing environmental damage.
   
5. **Increased Yields:**
   - By preventing disease and managing crops more effectively, farmers are more likely to achieve higher yields. This increases profitability and can contribute to food security on a larger scale.

6. **Sustainability:**
   - The system encourages sustainable farming practices by optimizing the use of resources and minimizing unnecessary chemical applications. By using AI, the system can suggest the most environmentally friendly options available.


By integrating AI-driven analysis with real-time data, farmers will have an innovative way to safeguard their crops and optimize their farming practices, making agriculture more productive and sustainable.
            
                ''')

    st.markdown('''The Dataset being used for training the model is a comprehensive collection of images depicting various diseases affecting major crops including **wheat, maize, cotton, sugarcane, and rice**.

The dataset contains a diverse range of crop disease images, meticulously curated from multiple sources to ensure completeness and relevance. It encompasses images of common and rare diseases afflicting each crop, captured at different stages of development and severity.")

### Contents:
\n\n

1. Train dataset contains 13,920 images
2. Test set contains 1856 images
3. Validation set contains 28880 images
4. There are a total of 42 classes in which these images are categorised into.
''')



#disease recognition-page
elif mode=='Disease-Recognition':
    st.markdown("# Disease-Recognition:")
    st.write(" ")
    st.write(" ")
    # Start Uploading Image section
    st.header("📷 Start by Uploading a Crop Image")
    st.write(" ")
    st.write("Upload your crop image below to begin diagnosing potential diseases. Our AI is designed to support a wide range of crops, including vegetables, fruits, and grains.")
    st.write(" ")
    st.write(" ")
    # Call to action button for image upload
    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if (st.button("Show image")):
        st.image(uploaded_image,use_column_width=True)




# 5-day weather predictiong and precaution page---------------------------------------------------------




elif mode == "5-Day forecast":
    st.title('Weather Forecast and Crop Precautions')
    st.write('')
    st.write('')

    # Get user location
    location, city, country = get_user_location()
    
    if location:
        st.write(f"Detected Location: **{city}, {country}** (Lat: {location[0]}, Lon: {location[1]})")
        st.write('')
        st.write('')
        # Fetch and display weather forecast
        forecast_df = get_weather_forecast(location[0], location[1])
        plot_weather_forecast(forecast_df)
        
        # Calculate averages and give precautions
        st.write('')  # Adding space for better readability
        st.write('')
        calculate_averages_and_precautions(forecast_df)
    else:
        st.write("Sorry, could not determine your location. Please try again.")
        
