#importing libraries

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

# # Use st.cache_data to cache the function
# @st.cache_data
# def get_img_as_base64(file):
#     with open(file, "rb") as f:
#         data = f.read()
#     return base64.b64encode(data).decode()

# img = get_img_as_base64("proj_bg2.jpg")
# img1=get_img_as_base64("sidebar_proj.jpg")

# page_bg_img = f"""  
# <style>
# [data-testid="stAppViewContainer"] {{
# background-image: url("data:image/jpg;base64,{img}");
# background-size: cover;
# }}

# [data-testid="stHeader"]{{
# background-color: rgba(0,0,0,0);
# }}

# [data-testid="stToolbar"]{{
# right:2rem;
# }}


# }}
# </style>

# """

# st.markdown(page_bg_img, unsafe_allow_html=True)

#web.py


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
    API_KEY = 'yourapikey'  # Replace with your OpenWeatherMap API key
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



    # model=tf.keras.models.load_model('my_model.h5')  #loading trained model
    # image=tf.keras.preprocessing.image.load_img(test_image,target_size=(128,128))
    # input_arr=tf.keras.preprocessing.image.img_to_array(image)  #converting image to array
    # input_arr=np.array([input_arr])  #coneverting array to batch
    # prediction=model.predict(input_arr)
    # result_index=np.argmax(prediction)
    # return result_index




#dashboard
st.sidebar.title("DASHBOARD")
mode=st.sidebar.selectbox("Select Page:",["🏡 Home", "👤 About","🦠 Disease-Recognition","🌥️ 5-Day forecast", "🌱 Crop recommender", "🧪 Fertilizer recommender","👥 Team"])


st.sidebar.markdown("<h1 style='text-align: left; margin-top: 220px;'>  🌾 GreenWatch</h1>", unsafe_allow_html=True)
st.sidebar.markdown("<h6 style='text-align: right;'>- By AI-CRAFT</h6>", unsafe_allow_html=True)
if (mode=="🏡 Home"):
     # Title of the app
    st.markdown("<h1 style='text-align: center;'>🌾 GreenWatch: AI Crop Health Advisor</h1>", unsafe_allow_html=True)

    # Introduction and tagline
    st.write('')
    st.write('')
    st.write('')
    st.write('')

    # List of images (replace with your actual paths)
    image_files = ["imge_inp1.jpg", "imge_inp2.jpg", "imge_inp3.jpg"]  # Replace with actual paths

    # def display_images(image_list):
    #     # Initialize session state for image index
    #     if "image_index" not in st.session_state:
    #         st.session_state.image_index = 0

    #     # Container to hold the image
    #     image_placeholder = st.empty()

    #     # Auto-cycle through images every 'display_duration' seconds
    #     while True:
    #         # Show the current image
    #         image_placeholder.image(image_list[st.session_state.image_index], use_column_width=True)

    #         # Update the image index for next cycle
    #         st.session_state.image_index = (st.session_state.image_index + 1) % len(image_list)

    #         # Sleep for the display duration before changing to the next image
    #         time.sleep(3)

    # Start displaying the images
    # display_images(image_files)
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
elif (mode=="👤 About"):
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
elif mode == '🦠 Disease-Recognition':
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
    
    if uploaded_image is not None:
        if st.button("Show image"):
            st.image(uploaded_image, use_column_width=True)

        # Predict button
        if st.button("Predict"):
            # Store the prediction result
            result = model_prediction(uploaded_image)
            
            # Define class names
            progress_text = "Model is predicting.....Please wait"
            my_bar = st.progress(0, text=progress_text)
            for perc_completed in range(100):
                time.sleep(0.01)
                my_bar.progress(perc_completed + 1, text=progress_text)
                
            st.success("It's a {}".format(result))

            if result == 'Potato___Early_blight':
                    st.header("Potato Early Blight")
                    st.markdown("""
                    **Description**: Potato early blight, also known as target spot, is a common fungal disease that affects potato leaves, stems, and tubers. It's caused by the fungus *Alternaria solani*. Early blight is primarily a disease of stressed or senescing plants.

                    ### Symptoms:
                    1. Small, dry, papery spots that turn dark brown or black and become oval or angular (0.12 to 0.16 inch (3–4 mm) in diameter). These spots first appear on older leaves and spread to younger leaves. The spots can develop concentric rings and yellow margins, giving them a target-like appearance.
                    2. Dark brown or black spots appear on the stems.
                    3. Dark, sunken, cork-like spots with raised margins appear on the tuber surface after harvest. Infected tubers show a brown, corky dry rot.

                    ### Causes:
                    1. Early blight of potato is caused by the fungus, *Alternaria solani*.
                    2. Early blight is favored by warm temperatures and high humidity.

                    ### Precautions:
                    1. Maintain optimum growing conditions, including proper fertilization, irrigation, and management of other pests.
                    2. Protect crops from stress-causing factors; harvest at the right time.
                    3. Use fungicides such as AZOXYSTROBIN, BOSCALID, CHLOROTHALONIL, etc.
                    4. Keep cull/compost piles away from potato growing areas.
                    5. Cover tubers with soil throughout the season.
                    6. Use drip irrigation to water at the base of each plant.
                    7. Select potato varieties that are resistant to early blight.

                    ### How to Avoid Early Blight:
                    1. A 3-4 year crop rotation cycle is ideal to break the disease cycle.
                    2. Select potato varieties that are bred for resistance or tolerance to early blight.
                    3. Avoid overhead irrigation to minimize leaf wetness.
                    4. Space plants properly for good air circulation.
                    """)

            elif result == 'Potato___Late_blight':
                    st.header("Potato Late Blight")
                    st.markdown("""
                    **Description**: Late blight is caused by the fungal-like oomycete pathogen *Phytophthora infestans*.

                    ### Symptoms:
                    1. Small, light to dark green, circular to irregular-shaped water-soaked spots. These lesions usually appear first on the lower leaves.
                    2. Lesions expand rapidly into large, dark brown or black lesions during cool, moist weather.
                    3. A pale green to yellow border often surrounds the lesions. Severely infected fields often produce a distinct odor.
                    4. Late blight infection of tubers is characterized by irregularly shaped, slightly depressed areas that can vary considerably in size and color.

                    ### Causes:
                    1. Late blight is caused by the fungus *Phytophthora infestans*.
                    2. The pathogen is favored by moist, cool environments.

                    ### Precautions:
                    1. Destroy all cull and volunteer potatoes.
                    2. Plant late blight-free seed tubers and resistant potato varieties.
                    3. Avoid mixing seed lots as cutting can transmit late blight.
                    4. Eliminate sources of inoculum such as hairy nightshade weed species and volunteer potatoes.
                    """)
            elif result=="Apple__Apple_scab":
                st.header("Apple Scab")
                st.markdown(""" 
        Apple scab is a common fungal disease caused by Venturia inaequalis, affecting apple trees and other related plants. It leads to significant damage to leaves, fruit, and even shoots, impacting both the appearance and marketability of the fruit.

        ##Symptoms: 
        1.Olive-green to brown velvety spots on leaves and fruits

        2.Leaves may become distorted and fall prematurely

        3.Dark, raised lesions on fruits that can crack as they enlarge

        4.In severe cases, reduced fruit yield and tree vigor


        ##Prevention:

        1.Cultivar selection: Plant resistant apple varieties to minimize the risk.

        2.Proper sanitation: Remove and destroy fallen leaves and infected fruit to reduce the spread of the fungus.

        3.Pruning: Maintain good air circulation through proper pruning to keep foliage dry.

        4.Preventative fungicide: Apply fungicides during the early stages of leaf and fruit development, especially in regions prone to wet conditions.


        ##Cure:

        1. Fungicide Treatment:

        Apply fungicides specifically effective against Venturia inaequalis at the first sign of infection or preventatively during early spring. Fungicides containing myclobutanil, captan, or mancozeb are commonly used. Always follow local guidelines for safe and effective use.



        2. Cultural Practices:

        Remove Infected Material: Rake and dispose of fallen leaves and fruit to eliminate sources of spores.

        Prune for Airflow: Regularly prune the tree to improve air circulation, which helps keep leaves dry and prevents fungal growth.

        Regular monitoring of trees, maintaining cleanliness in the orchard, and applying treatments as needed can greatly reduce the impact of apple scab.

                """)
            





# 5-day weather predictiong and precaution page---------------------------------------------------------




elif mode == "🌥️ 5-Day forecast":
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

elif mode=="🌱 Crop recommender":
    
    st.title('Crop recommender 🪴')
    st.write('')
    st.write('')
    st.write("")
    st.write("")
    
    st.write("Are you confused with what crop to cultivate?")
    st.write("Fill the following details to get the solution for your confusion.")
    st.write("")
    st.write("")
    # Create two columns for the form inputs
    col1, col2 = st.columns([1, 1])  # This will give both columns equal width
    with st.form("User_details"):
        with col1:
            N_input = st.text_input("Enter the percentage of Nitrogen content in soil:")
            P_input = st.text_input("Enter the percentage of Phosphorous content in soil:")
            K_input = st.text_input("Enter the percentage of Potassium content in soil:")
            temp_input = st.text_input("Enter the temperature (Celsius):")

        with col2:
            humid_input = st.text_input("Enter the percentage of humidity:")
            ph_input = st.text_input("Enter the pH value of the soil:")
            rainfall_input = st.text_input("Enter the amount of rainfall received:")

        submit_button = st.form_submit_button("Submit")

     # Actions after form submission
    if submit_button:
        try:
            # Preprocess inputs (convert text inputs to numeric values)
            temp = float(temp_input)
            humid = float(humid_input)
            ph = float(ph_input)
            rainfall = float(rainfall_input)
            N_input=int(N_input)
            P_input=int(P_input)
            K_input=int(K_input)

            # Create a feature array (ensure it has the correct shape)
            features = np.array([[N_input, P_input, K_input, temp, humid, ph, rainfall]])

            # Load the model (ensure the path to the model is correct)
            try:
                model = joblib.load(r'Models\Naive_bayes_crp.pkl')
            except FileNotFoundError:
                st.error("Model file 'Naive_bayes_crp.pkl' not found. Please check the file path.")
                raise

            # Make prediction
            prediction = model.predict(features)  # Ensure this works with your model type

            # Display the prediction or any relevant output
            st.success(f'Predicted plant for your conditions: {prediction[0]}')
            st.write("")
            st.write(f"**{str.upper(prediction[0])}** is the likely crop to be planted for your weather and soil conditions.")
            st.write("")
            st.subheader("Please note that we only predict for these crops on any given conditions:")
            st.write("Apple, Banana, Rice, Pomegranate, Pigeonpeas ,Papaya, Orange, Muskmelon, Mungbean, Mothbeans,")
            st.write("Mango, Maize, Lentil, Kidneybeans, Jute, Grapes, Cotton, Coffee, Coconut, Chickpea, Blackgram, Watermelon")



        except ValueError as e:
            st.error(f"Error in input conversion: {e}")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")


elif mode=="🧪 Fertilizer recommender":
    st.title('🧪 Fertilizer recommender')
    st.write('')
    st.write('')
    st.write("")
    st.write("")

    st.write("Please test your soil at your nearest soil testing center and also check weather conditions in your area and fill the below columns with appropriate data")
    st.write("")
    # Create two columns for the form inputs
    col1, col2 = st.columns([1, 1])  # This will give both columns equal width
    
    crop_mapping = {
    'rice': 0,
    'Wheat': 1,
    'Tobacco': 2,
    'Sugarcane': 3,
    'Pulses': 4,
    'pomegranate': 5,
    'Paddy': 6,
    'Oil seeds': 7,
    'Millets': 8,
    'Maize': 9,
    'Ground Nuts': 10,
    'Cotton': 11,
    'coffee': 12,
    'watermelon': 13,
    'Barley': 14,
    'kidneybeans': 15,
    'orange': 16
    }
    soil_mapping = {
    'Clayey': 0,
    'Loamy': 1,
    'Red': 2,
    'Black':3,
    'Sandy':4

        }
    
    with st.form("User_details"):
        with col1:
            temp_input = st.text_input("Enter the temperature (Celsius):")
            humid_input = st.text_input("Enter the humidity:")
            moist_input = st.text_input("Enter the Moisture in soil:")
            soil_input = st.selectbox("Select the soil type:", soil_mapping.keys())
        with col2:
            crop_input = st.selectbox("Select the crop type:",crop_mapping.keys())
            N_input = st.text_input("Enter the Nitrogen(N) content in soil:")
            P_input = st.text_input("Enter the Phosphorous(P) content:")
            K_input = st.text_input("Enter the Potassium(K) content:")

        submit_button = st.form_submit_button("Submit")

     # Actions after form submission
    if submit_button:
        try:
            # Preprocess inputs (convert text inputs to numeric values)
            temp = int(temp_input)
            humid = int(humid_input)
            moist=int(moist_input)
            N_input=int(N_input)
            P_input=int(P_input)
            K_input=int(K_input)

       
            # Load the model (ensure the path to the model is correct)
            model = joblib.load(r'Models\svm_model.pkl_2')

            #input array to model
            data=np.array([temp,humid,moist,soil_mapping[soil_input],crop_mapping[crop_input],N_input,K_input,P_input])
            prediction=model.predict([data])   # Make prediction
            

            # Display the prediction or any relevant output
            st.success(f'Predicted Appropriate Fertilizer is: {prediction[0]}')
            st.write("")
            st.write(f"**{str.upper(prediction[0])}** is the likely fertilizer to be used for your crop as per the conditions.")
            st.write("")

        except ValueError as e:
            st.error(f"Error in input conversion: {e}")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")





elif mode=="👥 Team":
        # Function to convert image to base64 encoding
        def get_base64_image(image_path):
            with open(image_path, "rb") as file:
                data= file.read()
                return base64.b64encode(data).decode()

        # Teams Tab content
        st.title("Meet the Team:  AI-Craft")
        st.write("")
        st.write("")
        st.write("")
        st.markdown("""
        **GreenWatch AI** is a cutting-edge solution designed to assist farmers with AI-powered crop disease detection, weather-based predictions, and tailored recommendations to enhance crop health and yield. Our team is dedicated to empowering agriculture with advanced technology and sustainable practices.

                    

        Below are the amazing team members who contributed to the success of this project:
        """)

        # Create a layout with columns for displaying team photos
        st.write("")

        # Add a section for team members with photos
        team_members = [
            {"name": "Krishna Vamsy K", "Enrollment no:": "23/11/EC/002", "image_path": "team\krishna.jpg"},
            {"name": "M.Pradeep", "Enrollment no:": "23/11/EC/063", "image_path": "team\pradeep.jpg"},
            {"name": "A.Sampath Dev", "Enrollment no:": "23/11/EC/029", "image_path": "team\sampath.jpg"},
            {"name": "Vignesh Thangabalan B", "Enrollment no:": "23/11/EC/020", "image_path": r"team\vignesh.jpg"},
            {"name": "M.Jai Ram Chandra", "Enrollment no:": "23/11/EC/071", "image_path": "team\jairam.jpg"}
        ]

        for member in team_members:
            # Create a two-column layout
            col1, col2 = st.columns([1, 4])  # Adjust the column width as needed
            with col1:
                # Convert the image to base64 for displaying in HTML
                try:
                    img_base64 = get_base64_image(member["image_path"])
                    st.markdown(
                        f"""
                        <div style="
                            border: 2px solid #ccc;
                            border-radius: 10px;
                            padding: 5px;
                            display: inline-block;
                            text-align: center;
                            max-width: 150px;">
                            <img src="data:image/jpg;base64,{img_base64}" style="max-width: 100%; border-radius: 10px;">
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                except FileNotFoundError:
                    st.warning(f"Image for {member['name']} not found.")
            with col2:
                st.subheader(member["name"])
                st.write(f"**Enrollment No:**: {member['Enrollment no:']}")
                st.write("")
                st.write("")
                st.write("")
                st.write("")
                st.write("")





            # st.success(f'Predicted plant for your conditions: {prediction[0]}')
            # st.write("")
            # st.write(f"**{str.upper(prediction[0])}** is the likely crop to be planted for your weather and soil conditions.")
            # st.write("")
            # st.subheader("Please also see that we only predict for these crops on any given conditions:")
            # st.write("")
            # st.write("apple, banana, rice, pomegranate, pigeonpeas ,papaya, orange, muskmelon, mungbean, mothbeans,")
            # st.write("mango, maize, lentil, kidneybeans, jute, grapes, cotton, coffee, coconut, chickpea, blackgram, watermelon")


