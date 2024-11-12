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

 
def Englsh():
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


    #dashboard
   # st.sidebar.title("DASHBOARD")
    # mode=st.sidebar.selectbox("Select Page:",["🏡 Home", "👤 About","🦠 Disease-Recognition","🌥️ 5-Day forecast", "🌱 Crop recommender", "🧪 Fertilizer recommender","👥 Team"])
    with st.sidebar:
        mode=option_menu(
            menu_title=None,
            options=["🏡 Home", "👤 About","🦠 Disease-Recognition","🌥️ 5-Day forecast", "🌱 Crop recommender", "🧪Fertilizer recommender","👥 Team"]
                   
        )

    st.sidebar.markdown("<h1 style='text-align: left; margin-top: 150px;'>  🌾 GreenWatch</h1>", unsafe_allow_html=True)
    st.sidebar.markdown("<h6 style='text-align: right;'>- By AI-CRAFT</h6>", unsafe_allow_html=True)
    if (mode=="🏡 Home"):
        # Title of the app
        st.markdown("<h1 style='text-align: center;'>🌾 GreenWatch: AI Crop Health Advisor</h1>", unsafe_allow_html=True)

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
        st.write("")
        st.write("")
        
        st.markdown('''
    ### About Crop Disease predictor:
    \n\n
    Our project leverages machine learning to predict crop diseases, helping farmers make proactive decisions to protect their yields. By analyzing data from sources such as crop images and weather forecasts, our model identifies patterns that indicate potential disease outbreaks.
    This early detection enables timely intervention, reducing crop losses and promoting sustainable agriculture. Our solution integrates with easy-to-use platforms, providing farmers with actionable insights to enhance crop health and productivity.''')
                
        
        st.markdown('''
    ### About Crop Recommendation System:
    \n\n
    Our crop recommendation system uses machine learning to suggest the best crops for farmers to grow based on soil properties, climate conditions. By analyzing these factors, the system helps farmers choose crops that are most likely to thrive in their specific conditions, optimizing yield potential, resource usage, and overall farm productivity. This technology supports sustainable agriculture by aligning crop choices with environmental suitability.''')

        st.markdown('''
    ### About Fertilizer Recommendation System:
    \n\n
    Our fertilizer recommendation system uses machine learning to suggest optimal fertilizer types and quantities based on crop type, soil composition, and environmental conditions. By analyzing these factors, the system provides tailored recommendations that improve crop health and maximize yield, while minimizing waste and environmental impact. This approach promotes efficient nutrient management, supporting sustainable farming practices.''')

        st.markdown('''
    ### About Weather Forecasting System:
    \n\n
    Our weather forecasting system uses advanced machine learning algorithms to predict weather patterns and conditions that affect crop growth. By real-time observations, and climate models, it provides accurate forecasts for temperature, rainfall, humidity, and other key factors. These predictions help farmers make informed decisions on planting, irrigation, and pest management, ultimately optimizing crop yield and reducing risks from adverse weather events.''')

        st.markdown('''
    ## Tools and Technology used:
    \n\n
    1.Frontend -->Streamlit\n
    2.Backend -->Machine Learning models and algorithms \n
    3.Machine models and algorithms -->CNN, SVM, Naive-Bayes, YOLO, Transfer Learning.\n
                    
    ## How we did?
    We used CNN architecture for crop disease prediction for various plants, along with that we used transfer learning for betterment for our model and finally applied YOLO to reach the peak form.
    Naive-Bayes and SVM were used in predicting crop recommender and fertilizer recommender respectively.

    What we did in front-end?\n
    ->Crop-Disease Predictor\n
    ->Weather forecast and location detector\n
    ->Crop-Recommender\n
    ->Fertilizer-Recommender
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

                if result == 'Potato_Early_blight':
                        st.header("Potato Early Blight")
                        st.markdown("""
    Potato Early Blight is a common fungal disease caused by Alternaria solani. It typically affects potato plants during warm, humid weather and can lead to reduced yield and quality if not managed effectively.

    ### Symptoms:

    1.Leaf Spots: Dark brown to black spots with concentric rings (target-like appearance) develop on older leaves. These spots may enlarge, leading to significant leaf damage.

    2.Yellowing: Leaves surrounding the lesions may turn yellow and eventually die off.

    3.Stem Lesions: Dark, elongated lesions may appear on stems.

    4.Tuber Infection: In severe cases, tubers can show sunken, dark lesions that may have a leathery texture.


    ### Prevention:

    1. Crop Rotation: Rotate crops with non-host plants (such as legumes) to break the disease cycle.


    2. Resistant Varieties: Use potato varieties that are resistant to early blight.


    3. Good Field Hygiene: Remove and destroy infected plant debris after harvest to reduce the presence of the fungus.


    4. Proper Spacing and Pruning: Ensure adequate plant spacing and prune to allow better air circulation and reduce humidity around the plants.


    5. Water Management: Avoid overhead watering, as wet foliage promotes fungal growth. Water early in the day to allow leaves to dry quickly.


    6. Balanced Fertilization: Use balanced fertilization, ensuring adequate levels of potassium, which can help strengthen plant resistance.



    ### Cure:

    1. Fungicide Application:

    Chlorothalonil (e.g., Bravo)

    Mancozeb (e.g., Dithane)

    Azoxystrobin or Difenoconazole for systemic protection



    2. Application Timing: Begin fungicide application at the first sign of disease or as a preventive measure during conditions conducive to early blight (warm, wet weather).


    3. Copper-based Sprays: For organic control, copper-based fungicides can help manage early blight when used regularly.


    4. Integrated Management: Combine fungicide treatments with good cultural practices for best results.



    ### Key Points:

    1.Early Detection: Regularly inspect crops for early signs of infection and treat promptly.

    2.Integrated Management Approach: Use a combination of resistant varieties, good cultural practices, and fungicide applications to effectively control early blight.


    By implementing these practices, potato crops can be protected from early blight, ensuring healthier plants and better yields.""")
                if result == 'Potato_Late_blight':
                        st.header("Potato Late Blight")
                        st.markdown("""
    Potato Late Blight is a devastating disease caused by the oomycete pathogen Phytophthora infestans. It is notorious for causing the Irish Potato Famine and continues to be a major threat to potato production, especially in areas with cool, wet weather.

    ### Symptoms:

    1.Leaf Lesions: Water-soaked, pale green to dark brown or black lesions appear on leaves, often starting at the edges. Lesions may expand rapidly and become necrotic.

    2.White Fungal Growth: A white, fuzzy mold often develops on the undersides of infected leaves in humid conditions.

    3.Stem Rot: Dark brown to black lesions can develop on stems, weakening and eventually killing them.

    4.Tuber Rot: Tubers show dark, firm, sunken areas that may develop into a brown, granular rot internally.


    ### Prevention:

    1. Resistant Varieties: Plant potato varieties that are resistant or tolerant to late blight.


    2. Seed Treatment: Use certified, disease-free seed potatoes to prevent initial infection.


    3. Crop Rotation: Practice crop rotation with non-host crops to reduce pathogen build-up in the soil.


    4. Proper Spacing: Space plants adequately to allow good air circulation and reduce leaf moisture.


    5. Field Hygiene: Remove and destroy infected plant debris and volunteer plants after harvest.


    6. Water Management: Avoid overhead watering and water early in the day so foliage dries quickly. Ensure proper drainage to prevent water accumulation.



    ### Cure:

    1. Fungicide Application:

    Chlorothalonil (e.g., Bravo)

    Mancozeb (e.g., Dithane)

    Metalaxyl or Mefenoxam for systemic action (e.g., Ridomil Gold)

    Cyazofamid and Fluopicolide for effective control



    2. Early and Regular Spraying: Apply fungicides preventively, especially during favorable conditions (cool, moist weather). Repeat applications as needed according to the product's guidelines.


    3. Copper-based Sprays: For organic management, copper-based fungicides can provide some protection.


    4. Integrated Disease Management (IDM): Combine chemical treatments with cultural practices for a more sustainable approach.



    ### Key Points:

    1.Rapid Action Needed: Late blight can spread quickly, so immediate action at the first sign of infection is crucial.

    2.Monitor Weather Conditions: Pay attention to weather forecasts; late blight thrives in cool, wet conditions, so preventive fungicide use is recommended during these periods.

    3.Field Sanitation: Properly dispose of infected plants and tubers to reduce sources of reinfection.


    With vigilant monitoring, preventive measures, and timely treatment, late blight can be managed to protect potato crops and ensure high yields.""")
                
                elif result=="Potato_healthy":
                    st.header("Your Potato Crop is Healthy")
                    st.markdown(""" 
                    Your crop is thriving, and to boost your yield further, applying the right fertilizer is essential.
                    Our fertilizer recommendation system can assist you in selecting the most suitable fertilizer for optimal growth.  """)
            
                elif result=="Apple__Apple_scab":
                    st.header("Apple Scab")
                    st.markdown(""" 
            Apple scab is a common fungal disease caused by Venturia inaequalis, affecting apple trees and other related plants. It leads to significant damage to leaves, fruit, and even shoots, impacting both the appearance and marketability of the fruit.

            ### Symptoms: 
            1.Olive-green to brown velvety spots on leaves and fruits

            2.Leaves may become distorted and fall prematurely

            3.Dark, raised lesions on fruits that can crack as they enlarge

            4.In severe cases, reduced fruit yield and tree vigor


            ### Prevention:

            1.Cultivar selection: Plant resistant apple varieties to minimize the risk.

            2.Proper sanitation: Remove and destroy fallen leaves and infected fruit to reduce the spread of the fungus.

            3.Pruning: Maintain good air circulation through proper pruning to keep foliage dry.

            4.Preventative fungicide: Apply fungicides during the early stages of leaf and fruit development, especially in regions prone to wet conditions.


            ### Cure:

            1. Fungicide Treatment:

            Apply fungicides specifically effective against Venturia inaequalis at the first sign of infection or preventatively during early spring. Fungicides containing myclobutanil, captan, or mancozeb are commonly used. Always follow local guidelines for safe and effective use.



            2. Cultural Practices:

            1.Remove Infected Material: Rake and dispose of fallen leaves and fruit to eliminate sources of spores.

            2.Prune for Airflow: Regularly prune the tree to improve air circulation, which helps keep leaves dry and prevents fungal growth.

            3.Regular monitoring of trees, maintaining cleanliness in the orchard, and applying treatments as needed can greatly reduce the impact of apple scab.

                    """)
                elif result=="Apple_Black_rot":
                    st.header("Apple Black rot")
                    st.markdown(""" 

            Apple black rot is a fungal disease caused by Botryosphaeria obtusa. It affects apple trees by causing rot in the fruit, leaf spots, and even cankers on branches, leading to reduced fruit quality and tree health.

            ### Symptoms:

            1.Fruit Rot: Dark, circular, sunken lesions on the fruit that expand and turn black with concentric rings.

            2.Leaf Spots: Small, purple-bordered lesions on leaves that may lead to early leaf drop.

            3.Cankers: Sunken, dark areas on branches or trunks, which can crack and spread, weakening the tree.


            ### Prevention:

            1.Sanitation: Remove fallen leaves, fruit, and pruned branches to limit fungal spore sources.

            2.Proper Pruning: Prune and dispose of infected branches to prevent the disease from spreading.

            3.Tree Health: Maintain tree health through proper watering, fertilization, and pest control to boost natural resistance.

            4.Fungicide Use: Apply fungicides as a preventive measure during the growing season, especially in humid areas.


            ### Cure:

            1.Pruning Out Cankers: Cut out affected branches at least 6–12 inches beyond the visible damage, disinfecting tools between cuts.

            2.Fungicide Application: Use fungicides such as those containing captan or thiophanate-methyl to manage the spread of the disease. Apply according to local extension service recommendations.

            3.Good Orchard Management: Keep the orchard clean and well-maintained, ensuring trees have proper spacing for air circulation.
                            """)
                            
                elif result=="Apple_healthy":
                    st.header("Your Apple Crop is Healthy")
                    st.markdown(""" 
                    Your crop is thriving, and to boost your yield further, applying the right fertilizer is essential.
                    Our fertilizer recommendation system can assist you in selecting the most suitable fertilizer for optimal growth.  """)
            

                elif result=="Corn_Blight":
                    st.header("Corn Blight")
                    st.markdown("""

    Corn blight is a fungal disease affecting corn crops, commonly caused by Bipolaris maydis (Southern corn leaf blight) or Exserohilum turcicum (Northern corn leaf blight). It can result in significant yield losses, particularly under warm and humid conditions.

    ### Symptoms:

    1.Southern Corn Leaf Blight: Tan, oval to elongated lesions on leaves with dark margins.

    2.Northern Corn Leaf Blight: Large, cigar-shaped grayish lesions that can merge and cover entire leaves, reducing photosynthesis.

    3.Advanced Stages: Premature leaf death, weakened plants, and reduced ear development.


    ### Prevention:

    1.Resistant Varieties: Plant corn hybrids that are resistant to the specific type of blight common in the region.

    2.Crop Rotation: Rotate crops with non-host plants to break the life cycle of the fungi.

    3.Field Sanitation: Remove and destroy infected plant debris post-harvest to reduce overwintering fungal spores.

    4.Proper Spacing: Ensure adequate spacing between plants to promote air circulation and reduce moisture buildup.


    ### Cure:

    1.Fungicide Application: Use fungicides such as those containing azoxystrobin or propiconazole when symptoms first appear or as a preventive measure in high-risk conditions.

    2.Timely Intervention: Monitor crops regularly and apply treatments early to minimize disease spread.

    3.Integrated Pest Management (IPM): Combine cultural practices, resistant varieties, and chemical treatments for comprehensive control.


    Regular monitoring and early detection are key to effectively managing corn blight and ensuring minimal crop damage. 
    """)
                elif result=="Corn_Common_Rust":
                    st.header("Corn Common Rust")
                    st.markdown(""" 
    Corn common rust is a fungal disease caused by Puccinia sorghi. It can affect corn crops under cool and humid conditions, leading to reduced yield and weakened plants if severe.

    ### Symptoms:

    1.Pustules: Small, oval to circular, reddish-brown pustules on both sides of the leaves.

    2.Leaf Damage: As the disease progresses, pustules may merge, causing leaves to dry out and die prematurely.

    3.Reduced Photosynthesis: Severe infections can impair the plant's ability to photosynthesize, weakening its overall growth.


    ### Prevention:

    1.Resistant Varieties: Plant corn hybrids that are resistant to common rust to reduce susceptibility.

    2.Crop Rotation: Practice crop rotation to minimize the buildup of fungal spores in the soil.

    3.Monitor Weather Conditions: Be vigilant during cool, moist weather, which is favorable for the development of rust.


    ### Cure:

    1.Fungicide Application: Apply fungicides such as those containing strobilurins or triazoles when rust is detected early. Ensure proper coverage and follow label instructions.

    2.Timely Spraying: Early intervention with fungicide applications can help control the spread and limit damage.

    3.Field Hygiene: Remove infected plant debris post-harvest to reduce the source of future infections.


    Regular monitoring, use of resistant corn varieties, and timely application of fungicides are essential to managing and curing corn common rust effectively.""")

                elif result=="Corn_Gray_Leaf_Spot":
                    st.header("Corn Gray Leaf Spot")
                    st.markdown(""" 
    Corn gray leaf spot is a common fungal disease caused by Cercospora zeae-maydis. It thrives in warm, humid conditions and can significantly impact yields by reducing photosynthetic leaf area.

    ### Symptoms:

    1.Initial Lesions: Small, rectangular, grayish-tan lesions on the lower leaves.

    2.Progression: Lesions may elongate and develop parallel to leaf veins, eventually merging and causing large areas of dead tissue.

    3.Reduced Photosynthesis: Severe infections can lead to premature leaf death, reducing photosynthesis and crop yield.


    ### Prevention:

    1.Resistant Varieties: Plant hybrids that have genetic resistance to gray leaf spot.

    2.Crop Rotation: Rotate crops with non-host plants to reduce fungal spore carryover.

    3.Residue Management: Plow under or remove infected plant debris to prevent overwintering of the fungus.

    4.Plant Spacing: Use proper plant spacing to improve air circulation and reduce leaf wetness.


    ### Cure:

    1.Fungicide Application: Apply fungicides containing:

            Azoxystrobin (e.g., Quadris)

            Pyraclostrobin (e.g., Headline)

            Propiconazole (e.g., Tilt)

            Tebuconazole (e.g., Folicur)


    2.Timing: Apply fungicides at early stages of disease development or at the VT (tasseling) to R1 (silking) growth stages for best results.

    3.Integrated Management: Combine chemical control with cultural practices for better disease management.


    Timely fungicide application, along with preventive measures like crop rotation and resistant hybrids, can help effectively manage and reduce the impact of corn gray leaf spot.""")


                elif result=="Corn_Healthy":
                    st.header("Your Corn Crop is Healthy")
                    st.markdown(""" 
    Your crop is thriving, and to boost your yield further, applying the right fertilizer is essential.
                    Our fertilizer recommendation system can assist you in selecting the most suitable fertilizer for optimal growth.  """)

                elif result=="Cotton_Healthy":
                    st.header("Your Cotton Crop is Healthy")
                    st.markdown(""" 
    Your crop is thriving, and to boost your yield further, applying the right fertilizer is essential.
                    Our fertilizer recommendation system can assist you in selecting the most suitable fertilizer for optimal growth.  """)

                elif result=="Cotton_bacterial_blight":
                    st.header("Cotton Bacterial Blight")
                    st.markdown(""" 
    Cotton bacterial blight, caused by Xanthomonas citri pv. malvacearum, is a serious disease that can affect all parts of the cotton plant. It thrives in warm, humid conditions and can lead to significant yield losses.

    ### Symptoms:

    1.Leaf Spots: Water-soaked, angular lesions on leaves that turn brown or black and may merge, causing leaf drop.

    2.Stem and Boll Lesions: Dark, sunken spots on stems and bolls that can lead to boll rot.

    3.Defoliation: Severe infections can cause early leaf drop, weakening the plant and reducing photosynthesis.


    ### Prevention:

    1.Resistant Varieties: Plant cotton varieties that are resistant to bacterial blight.

    2.Seed Treatment: Use certified, disease-free seeds and treat seeds with bactericides as a preventive measure.

    3.Crop Rotation: Rotate cotton with non-host crops to reduce pathogen buildup in the soil.

    4.Field Sanitation: Remove and destroy plant debris after harvest to minimize overwintering bacteria.

    5.Avoid Overhead Irrigation: Minimize leaf wetness by avoiding overhead irrigation during susceptible growth stages.


    ### Cure:

    1.Copper-based Bactericides:

            Copper hydroxide (e.g., Kocide, Champion) and copper oxychloride (e.g., Copper Sulfate) are the most commonly used bactericides to control bacterial blight. These can be applied as foliar sprays to reduce bacterial spread, especially during wet conditions.


    2.Timely Monitoring: Regularly scout fields for early detection and quick response.

    3.Integrated Management: Combine cultural practices, resistant varieties, and targeted bactericide application for effective control.


    Maintaining good field hygiene, using resistant varieties, and timely  bactericide application are key strategies for managing cotton bacterial blight and minimizing its impact.""")
                    
                elif result=="Cotton_curl_virus":
                    st.header("Cotton Curl Virus")
                    st.markdown(""" 
    Cotton Curl Virus (also known as Cotton Leaf Curl Virus or CLCuV) is a viral disease transmitted primarily by whiteflies (Bemisia tabaci). It is a significant problem for cotton crops in many regions, leading to reduced yields and poor quality cotton.

    ### Symptoms:

    1.Leaf Curling: The most characteristic symptom is the upward curling of leaves.

    2.Yellowing: Affected leaves may become yellow, especially along the veins (called chlorosis).

    3.Stunting: Plants may be stunted, with reduced growth and smaller bolls.

    4.Mosaic Patterns: Leaves may exhibit a mosaic or mottled appearance due to irregular chlorosis.

    5.Premature Dropping: In severe cases, leaves may drop prematurely, and bolls may fail to develop properly.


    ### Prevention:

    1.Insect Control: Control whitefly populations using insecticides such as:

        Imidacloprid (e.g., Confidor)

        Thiamethoxam (e.g., Actara)

        Pyriproxyfen (e.g., Knack)


    2.Use of Resistant Varieties: Plant cotton varieties that are resistant to Cotton Curl Virus.

    3.Field Sanitation: Remove infected plants and debris after harvest to reduce viral sources.

    4.Crop Rotation: Avoid continuous cotton cropping to break the cycle of whitefly and viral spread.

    5.Reflective Mulches: Use reflective mulches or cover crops to repel whiteflies and reduce their numbers.


    ### Cure:

    No Cure for the Virus: Once infected, plants cannot be cured of Cotton Curl Virus.

    Vector Control: Focus on controlling the whitefly vector to prevent further spread of the virus within the crop.

    Early Detection: Regularly monitor crops for symptoms and immediately remove infected plants to limit the spread of the virus to healthy plants.


    Cotton Curl Virus requires integrated management, with a focus on controlling whitefly populations and using resistant cotton varieties for effective prevention.""")
                    

                elif result=="Grape_Black_rot":
                    st.header("Grape Black rot")
                    st.markdown(""" 
    Grape Black Rot is a fungal disease caused by Guignardia bidwellii. It affects grapes and is particularly common in areas with warm, humid conditions. The disease can reduce both the quantity and quality of the grape harvest.

    ### Symptoms:

    1.Leaf Spots: Small, dark, circular lesions with yellow halos appear on leaves.

    2.Fruit Rot: Dark, sunken lesions develop on grape clusters. Infected fruit becomes hard, shriveled, and may fall off prematurely.

    3.Vine Damage: In severe cases, the fungus can also infect the stems and tendrils, causing them to darken and decay.


    ### Prevention:

    1.Resistant Varieties: Plant grape varieties that are resistant to black rot.

    2.Sanitation: Remove and destroy infected leaves, berries, and other plant debris to reduce sources of fungal spores.

    3.Proper Spacing: Ensure good air circulation around vines by pruning and providing adequate spacing between plants to reduce humidity.

    4.Fungicide Application: Apply fungicides during early growth stages and just before flowering. Effective fungicides include:

    ### Captan

            Chlorothalonil (e.g., Bravo)

            Mancozeb


    Timing of Application: Apply fungicides before rainy periods and after harvest to protect new growth and prevent overwintering of the fungus.


    ### Cure:

    1.Fungicide Treatments: Apply fungicides such as captan, chlorothalonil, or mancozeb at regular intervals to prevent infection. Treatment should begin early in the season (before symptoms appear) and continue during high-risk periods.

    2.Pruning Infected Areas: Prune and remove infected parts of the vine to limit the spread of the fungus.

    3.Spray for Residual Protection: Continue spraying fungicides until late in the growing season to prevent the spread of black rot to new fruit.


    Regular monitoring, timely fungicide application, and proper sanitation are key to preventing and controlling grape black rot.""")


                elif result=="Grape_Leaf_blight(Isariopsis_Leaf_Spot)":
                    st.header("Grape Leaf Blight / Isariopsis Leaf Spot")
                    st.markdown(""" 
    Grape Blight is a fungal disease caused by Phomopsis viticola. It affects grapevines, especially in humid climates, and can lead to significant damage, particularly during the early growing season.

    ### Symptoms:

    1.Leaf Spots: Small, dark, round lesions with a light gray center appear on leaves, often with a yellow halo.

    2.Shoot Blight: Young shoots may develop brown, sunken lesions that cause dieback, affecting the growth of the vine.

    3.Fruit Damage: Infected grape clusters show water-soaked spots that turn dark and can cause the fruit to shrivel and rot.

    4.Vine Dieback: In severe cases, infected canes may die, and the vine may become stunted or weakened.


    ### Prevention:

    1.Resistant Varieties: Select grape varieties resistant to Phomopsis Blight.

    2.Pruning and Sanitation: Regularly prune infected vines, removing and destroying infected shoots and leaves. Clean up fallen debris that could harbor spores.

    3.Proper Spacing: Ensure adequate air circulation around the vines to reduce humidity, which favors fungal growth.

    4.Fungicide Application: Apply fungicides such as:

    ### Captan

        Chlorothalonil (e.g., Bravo)

        Myclobutanil (e.g., Rally)


    Spray Timing: Begin spraying early in the season, before the buds break and during flowering, and continue at regular intervals, particularly after rain.


    ### Cure:

    1.Fungicide Treatment: Apply fungicides like captan, chlorothalonil, or myclobutanil during critical periods to protect new growth and prevent fungal infection.

    2.Pruning Infected Areas: Remove and destroy any infected shoots, leaves, or fruit to stop the spread of the disease.

    3.Monitor Regularly: Regularly inspect vines, particularly after wet weather, and reapply fungicides as needed.


    Grape Blight requires proactive management, combining fungicide applications with proper pruning and field sanitation to reduce the risk of infection and minimize damage.""")

                elif result=="Grape__healthy":
                    st.header("Your Grape Crop is Healthy")
                    st.markdown(""" 

            Your crop is thriving, and to boost your yield further, applying the right fertilizer is essential.
                    Our fertilizer recommendation system can assist you in selecting the most suitable fertilizer for optimal growth.  """)
            

                elif result=="Pepper,_bell_Bacterial_spot":
                    st.header("Bell Pepper Bacterial spot")
                    st.markdown(""" 
    Pepper bell bacterial spot is a serious disease caused by Xanthomonas euvesicatoria. It primarily affects peppers (including bell peppers) and can lead to significant yield losses, especially under warm and humid conditions.

    ### Symptoms:

    1.Leaf Spots: Small, water-soaked lesions on the upper surface of leaves that become angular with yellow halos.

    2.Lesions on Fruit: Dark, sunken lesions may appear on the fruit, which can become soft and rot.

    3.Defoliation: Severe infection leads to premature leaf drop, reducing photosynthesis and weakening the plant.

    4.Stunted Growth: Plants may become stunted, with reduced fruit size and overall yield.


    ### Prevention:

    1. Resistant Varieties: Choose pepper varieties that are resistant to bacterial spot.


    2. Seed Treatment: Use disease-free seeds and treat seeds with bactericides before planting.


    3. Proper Spacing and Pruning: Ensure good air circulation by spacing plants properly and pruning excess foliage to reduce humidity around the plant.


    4. Crop Rotation: Avoid planting peppers in the same soil consecutively to reduce bacterial build-up in the soil.


    5. Field Sanitation: Remove and destroy infected plant debris, as the bacteria can overwinter in plant residues.


    6. Avoid Overhead Irrigation: Use drip irrigation to keep foliage dry, as wet conditions promote the spread of the bacteria.



    ### Cure:

    1. Copper-based Bactericides: Apply copper-based products such as:

    Copper hydroxide (e.g., Kocide)

    Copper sulfate (e.g., Champion)

    Copper oxychloride


    These can help manage bacterial growth, especially during wet conditions.


    2. Biological Control: Some biological agents like Bacillus subtilis or Pseudomonas fluorescens can suppress bacterial activity.


    3. Timely Spraying: Apply bactericides at the first sign of infection or as a preventive measure during high-risk periods (warm, wet weather).


    4. Pruning Infected Plant Parts: Remove and dispose of infected leaves, stems, and fruit to reduce the spread of the bacteria.



    ### Key Points:

    1.No Complete Cure: Bacterial spot cannot be completely cured once it infects the plant, so early prevention and control are crucial.

    2.Integrated Pest Management (IPM): Combine cultural practices (such as proper spacing and crop rotation), resistant varieties, and chemical control to manage the disease effectively.


    Regular monitoring, proper care, and timely treatment are essential for controlling pepper bell bacterial spot and minimizing its impact on your crop.""")

                elif result=="Pepper,_bell_healthy":
                    st.header("Your Bell Pepper Crop is Heathy ")
                    st.markdown(""" 
    Pepper bell bacterial spot is a serious disease caused by Xanthomonas euvesicatoria. It primarily affects peppers (including bell peppers) and can lead to significant yield losses, especially under warm and humid conditions.

    ### Symptoms:

    1.Leaf Spots: Small, water-soaked lesions on the upper surface of leaves that become angular with yellow halos.

    2.Lesions on Fruit: Dark, sunken lesions may appear on the fruit, which can become soft and rot.

    3.Defoliation: Severe infection leads to premature leaf drop, reducing photosynthesis and weakening the plant.

    4.Stunted Growth: Plants may become stunted, with reduced fruit size and overall yield.


    ### Prevention:

    1. Resistant Varieties: Choose pepper varieties that are resistant to bacterial spot.


    2. Seed Treatment: Use disease-free seeds and treat seeds with bactericides before planting.


    3. Proper Spacing and Pruning: Ensure good air circulation by spacing plants properly and pruning excess foliage to reduce humidity around the plant.


    4. Crop Rotation: Avoid planting peppers in the same soil consecutively to reduce bacterial build-up in the soil.


    5. Field Sanitation: Remove and destroy infected plant debris, as the bacteria can overwinter in plant residues.


    6. Avoid Overhead Irrigation: Use drip irrigation to keep foliage dry, as wet conditions promote the spread of the bacteria.



    ### Cure:

    1. Copper-based Bactericides: Apply copper-based products such as:

        Copper hydroxide (e.g., Kocide)

        Copper sulfate (e.g., Champion)

        Copper oxychloride


    These can help manage bacterial growth, especially during wet conditions.


    2. Biological Control: Some biological agents like Bacillus subtilis or Pseudomonas fluorescens can suppress bacterial activity.


    3. Timely Spraying: Apply bactericides at the first sign of infection or as a preventive measure during high-risk periods (warm, wet weather).


    4. Pruning Infected Plant Parts: Remove and dispose of infected leaves, stems, and fruit to reduce the spread of the bacteria.



    ### Key Points:

    No Complete Cure: Bacterial spot cannot be completely cured once it infects the plant, so early prevention and control are crucial.

    Integrated Pest Management (IPM): Combine cultural practices (such as proper spacing and crop rotation), resistant varieties, and chemical control to manage the disease effectively.


    Regular monitoring, proper care, and timely treatment are essential for controlling pepper bell bacterial spot and minimizing its impact on your crop.""")

                elif result=="Rice_Healthy":
                    st.header("Your Rice Crop is Healthy")
                    st.markdown(""" 
                    Your crop is thriving, and to boost your yield further, applying the right fertilizer is essential.
                    Our fertilizer recommendation system can assist you in selecting the most suitable fertilizer for optimal growth.  """)
            

                elif result=="Rice_bacterial_leaf_blight":
                    st.header("Rice Bacterial Blight")
                    st.markdown(""" 
    Rice Bacterial Leaf Blight (BLB) is a destructive disease caused by Xanthomonas oryzae pv. oryzae. It primarily affects rice crops in warm and humid regions, leading to significant yield losses, especially in severe cases.

    ### Symptoms:

    1.Initial Lesions: Water-soaked streaks appear on the leaf margins, gradually turning yellow and then brown as the disease progresses.

    2.V-shaped Lesions: The streaks often extend from the leaf tips and margins, forming a characteristic V shape.

    3.Leaf Drying: Severely affected leaves dry out and turn a pale gray or white, giving a scorched appearance.

    4.Stunted Growth: Plants may show stunted growth, reduced tillering, and lower grain yield.


    ### Prevention:

    1. Resistant Varieties: Plant disease-resistant rice varieties to reduce the risk of infection.


    2. Seed Treatment: Use certified, disease-free seeds and treat seeds with bactericides or hot water to kill potential pathogens.


    3. Field Sanitation: Remove and destroy crop residues and infected plants after harvest to reduce sources of the bacteria.


    4. Balanced Fertilization: Avoid excessive nitrogen application as it can promote disease susceptibility. Use balanced fertilization with adequate potassium and phosphorus.


    5. Water Management: Manage water levels properly and avoid standing water for prolonged periods to limit the spread of bacteria.


    6. Avoid Overcrowding: Plant rice at recommended densities to improve airflow and reduce humidity, limiting the spread of the bacteria.



    ### Cure:

    1. Copper-based Bactericides: Spraying copper-based compounds can help manage the spread of the bacteria:

    Copper hydroxide (e.g., Kocide)

    Copper oxychloride



    2. Antibiotic Treatments: Streptomycin-based sprays or compounds with oxytetracycline can help manage bacterial growth but should be used cautiously to prevent resistance.


    3. Biological Control: Utilize biological agents like Pseudomonas fluorescens to suppress the activity of the bacterial pathogen.


    4. Early Intervention: Monitor the crop regularly, and take action as soon as early symptoms appear to contain the spread.


                                
    ### Key Points:

    No Complete Cure: There is no cure for rice bacterial leaf blight once the infection becomes severe. Therefore, prevention and early control are essential.

    Integrated Disease Management (IDM): Combine the use of resistant varieties, proper field management, balanced fertilization, and chemical treatments for effective control.


    Maintaining vigilance and applying preventive measures early can significantly reduce the risk of severe rice bacterial leaf blight and protect crop yield.""")
                
                
                elif result=="Rice_brown_spot":
                    st.header("Rice Brown Spot")
                    st.markdown(""" 
    Rice Brown Spot is a fungal disease caused by Bipolaris oryzae (previously Helminthosporium oryzae). It is common in areas with poor soil conditions and inadequate nutrition, impacting both the yield and quality of the rice.

    ### Symptoms:

    1.Leaf Spots: Small, circular to oval brown lesions with a grayish center and dark brown margins appear on leaves.

    2.Leaf Drying: Severe infection can cause leaves to turn brown and die prematurely.

    3.Grain Discoloration: Dark spots may appear on the rice grains, reducing their market quality.

    4.Stunted Growth: Plants may be stunted, and panicle development may be affected.


    ### Prevention:

    1. Nutrient Management: Ensure proper soil nutrition, especially adequate levels of potassium and phosphorus, to strengthen plant resistance.


    2. Seed Treatment: Use certified disease-free seeds and treat seeds with fungicides before planting to reduce initial infection.


    3. Proper Water Management: Maintain proper water levels and avoid stress conditions such as drought, which can predispose plants to infection.


    4. Crop Rotation: Practice crop rotation with non-host crops to break the disease cycle.


    5. Field Sanitation: Remove and destroy crop residues after harvest to minimize the presence of fungal spores in the field.



    ### Cure:

    1. Fungicide Application: Use appropriate fungicides during early stages of disease development:

    Mancozeb

    Tricyclazole

    Propiconazole



    2. Seed Treatment Fungicides: Treat seeds with carbendazim or thiram before sowing to reduce initial fungal load.


    3. Timely Application: Apply fungicides at the first sign of symptoms and repeat as needed based on the disease pressure and environmental conditions.


    4. Balanced Fertilization: Supplement soil with organic matter and balanced fertilizers to improve plant health and resistance.



    ### Key Points:

    Monitor Fields Regularly: Early detection and treatment can prevent significant yield losses.

    Integrated Management: Combine cultural practices, such as proper nutrition and field sanitation, with chemical control for effective management of rice brown spot.


    Adopting a proactive approach with good agricultural practices and timely fungicide application can help manage and reduce the impact of rice brown spot, ensuring healthier crops and better yields.""")
                
                
                
                elif result=="Rice_leaf_blast":
                    st.header("Rice Leaf Blast")
                    st.markdown(""" 
    Rice Leaf Blast is one of the most destructive fungal diseases affecting rice, caused by Magnaporthe oryzae (formerly Pyricularia oryzae). This disease can lead to significant yield losses, particularly under warm, moist conditions.

    ### Symptoms:

    1.Leaf Spots: Diamond-shaped or spindle-shaped lesions with grayish centers and brown borders appear on leaves. These spots may enlarge and join together, causing large, dead areas.

    2.Collar Rot: The disease may affect the collar where the leaf blade and sheath meet, causing the leaf to die.

    3.Node and Panicle Blast: In severe cases, the fungus infects nodes, weakening the stem and causing it to break. Panicle blast can lead to incomplete grain filling or panicle blight.


    ###Prevention:

    1. Resistant Varieties: Plant rice varieties known for their resistance to leaf blast.


    2. Balanced Fertilization: Apply nitrogen in balanced amounts. Excessive nitrogen promotes lush growth, which is more susceptible to infection.


    3. Water Management: Maintain consistent water levels in paddies. Alternate wetting and drying can increase susceptibility.


    4. Planting Density: Avoid overcrowded planting to ensure adequate air circulation and reduce humidity around plants.


    5. Field Hygiene: Remove and destroy crop residues and infected plant parts after harvest to limit sources of fungal spores.



    ### Cure:

    1. Fungicide Application:

    Tricyclazole: Effective in controlling rice blast.

    Isoprothiolane (e.g., Fuji-one)

    Edifenphos (e.g., Hinosan)



    2. Timely Spraying: Apply fungicides at the early stage of infection or as a preventive measure during weather conducive to fungal growth (high humidity and moderate temperatures).


    3. Crop Monitoring: Regularly inspect crops for early signs of infection and apply treatment as necessary.


    4. Silicon Fertilization: Adding silicon to the soil can help strengthen plant cell walls, making them less susceptible to fungal penetration.



    ### Key Points:

    1.Integrated Disease Management (IDM): Combine the use of resistant varieties, cultural practices, and chemical control for effective management.

    2.Prompt Action: Immediate response to initial symptoms can help prevent the disease from spreading to the entire crop.


    Regular monitoring, balanced nutrient application, and a combination of preventive and curative measures are essential to manage rice leaf blast effectively and protect the yield.""")


                elif result=="Sugarcan_Mosaic":
                    st.header("Sugarcane Mosaic")
                    st.markdown(""" 
    Sugarcane Mosaic is a viral disease caused by the Sugarcane mosaic virus (SCMV). It can lead to significant reductions in sugar yield and overall crop health. The virus is spread by aphids and can also be transmitted through infected plant material.

    ### Symptoms:

    1.Mosaic Patterns: Light and dark green patches appear on the leaves, creating a mosaic-like pattern.

    2.Stunted Growth: Infected plants may show stunted growth and reduced vigor.

    3.Leaf Distortion: Leaves may become distorted or show irregular streaking.

    4.Reduced Yield: Severe infections can lead to decreased cane and sugar production.


    ### Prevention:

    1. Plant Disease-Free Material: Use certified, virus-free seed cane to prevent initial infection.


    2. Resistant Varieties: Plant sugarcane varieties that are resistant or tolerant to mosaic virus.


    3. Aphid Control: Manage aphid populations using biological control methods or insecticides to reduce virus spread.


    4. Crop Rotation: Implement crop rotation and avoid planting sugarcane in fields with a history of the disease.


    5. Field Sanitation: Remove and destroy infected plants and weeds that can act as virus reservoirs.



    ### Cure:

    1.No Direct Cure: There is no chemical cure for viral infections. Management focuses on prevention and control measures.

    2.Aphid Management:

    Use insecticides like Imidacloprid to control aphid populations.


    3.Rogueing: Remove and destroy infected plants to reduce the spread within the field.

    4.Integrated Management: Combine the use of resistant varieties, good cultural practices, and vector control to effectively manage the disease.


    ### Key Points:

    1.Virus Transmission: The primary method of virus spread is through aphids, so vector control is essential.

    2.Regular Monitoring: Inspect fields frequently to identify and remove infected plants early.

    3.Resistant Varieties: Continuously seek and use improved resistant varieties to reduce the impact of Sugarcane Mosaic.


    Effective management of Sugarcane Mosaic involves preventive measures, diligent field monitoring, and vector control to protect crops from infection and minimize yield loss.""")


                elif result=="Sugarcane_Redrot":
                    st.header("Sugarcane RedRot")
                    st.markdown(""" 
    Sugarcane Red Rot is a serious fungal disease caused by Colletotrichum falcatum (also known as Glomerella tucumanensis in its sexual form). It is one of the most destructive diseases affecting sugarcane, leading to significant losses in cane yield and sugar recovery.

    ### Symptoms:

    1.Red Discoloration: Internal tissues of affected stalks show a characteristic red coloration with white patches when split open.

    2.Spongy Stalks: Infected stalks may become spongy and emit a foul odor as the disease progresses.

    3.Yellowing and Drying Leaves: Initial symptoms may include yellowing and drying of the leaves from the top down.

    4.Cane Lodging: Severely infected plants may weaken and collapse.

    5.Fungal Growth: In moist conditions, small, black fungal fruiting bodies may appear on the affected areas.


    ### Prevention:

    1. Resistant Varieties: Plant sugarcane varieties that are resistant or tolerant to red rot.


    2. Field Hygiene: Remove and destroy infected plant material after harvest to reduce the source of infection.


    3. Healthy Seed Material: Use certified, disease-free seed canes for planting.


    4. Crop Rotation: Rotate with non-host crops like legumes to reduce the buildup of the pathogen in the soil.


    5. Adequate Drainage: Ensure proper field drainage to prevent waterlogging, which encourages fungal growth.



    ### Cure:

    1. Fungicide Treatment:

    Carbendazim and Thiophanate-methyl are commonly used fungicides to treat planting material and prevent infection.



    2. Hot Water Treatment:

    Soak seed canes in hot water at 52°C for 30 minutes to kill fungal spores.



    3. Biological Control:

    Use beneficial fungi like Trichoderma spp. as a biocontrol agent to suppress Colletotrichum falcatum in the soil.



    4. Early Detection: Regular monitoring for early symptoms allows for the removal of infected plants before the disease spreads.



    ### Key Points:

    Sanitation and Hygiene: Maintaining field hygiene and using healthy planting material are essential to prevent the spread of red rot.

    Regular Monitoring: Check for early signs of infection to take prompt action.

    Integrated Management: Combine resistant varieties, fungicide application, and proper cultural practices for effective disease management.


    With proactive measures and timely treatment, sugarcane crops can be protected from red rot, ensuring better yields and healthy growth.
    """)

                elif result=="Sugarcane_Rust":
                    st.header("Sugarcane Rust")
                    st.markdown("""
    Sugarcane Rust is a disease caused by various fungal pathogens, most commonly Puccinia melanocephala (Brown Rust) and Puccinia kuehnii (Orange Rust). These diseases can lead to significant yield losses, especially in susceptible sugarcane varieties. Rust diseases thrive in warm, humid environments.

    Sugarcane Rust is a disease caused by various fungal pathogens, most commonly Puccinia melanocephala (Brown Rust) and Puccinia kuehnii (Orange Rust). These diseases can lead to significant yield losses, especially in susceptible sugarcane varieties. Rust diseases thrive in warm, humid environments.

    ### Symptoms:

    1. Reddish-Brown or Orange Pustules: Small, raised pustules develop on the undersides of leaves, giving them a reddish-brown or orange appearance depending on the rust type.

    2. Yellowing Leaves: Leaves may show chlorosis (yellowing) around the pustules.

    3. Premature Leaf Death: Severe infections can cause leaves to dry out and die prematurely, reducing photosynthesis and yield.

    4.Reduced Cane Growth: Infected plants may experience stunted growth and lower sugar content.


    ### Prevention:

    1. Resistant Varieties: Plant rust-resistant sugarcane varieties to minimize the impact of the disease.


    2. Field Hygiene: Remove and destroy infected plant debris after harvest to reduce the source of infection.


    3. Balanced Fertilization: Avoid excessive nitrogen application, as it can promote susceptibility to rust.


    4. Proper Spacing: Ensure adequate plant spacing to promote airflow and reduce leaf wetness, which can favor rust development.



    ### Cure:

    1. Fungicide Application:

    Use fungicides like Propiconazole, Tebuconazole, or Mancozeb for effective control.

    Apply fungicides as a preventive measure or at the first sign of rust symptoms for maximum efficacy.



    2. Timely Spraying: Apply fungicides in favorable weather conditions (warm and humid) to prevent outbreaks.


    3. Integrated Pest Management (IPM): Combine cultural practices, resistant varieties, and fungicide applications to manage rust effectively.



    ### Key Points:

    1. Monitor Weather Conditions: Rust thrives in warm, moist climates, so adjust prevention measures based on weather forecasts.

    2. Early Detection: Regular field scouting helps identify early symptoms and prevent the disease from spreading.

    3.Rotational Management: Use crop rotation and avoid continuous sugarcane planting in the same fields to reduce disease pressure.


    By adopting these preventive and management strategies, sugarcane rust can be effectively controlled, ensuring healthy crops and optimal yield.""")

                elif result=="Sugarcane_Yellow":
                                st.header("Sugarcane Yellowing")
                                st.markdown("""
    Sugarcane yellowing, also referred to as "yellowing syndrome," is a condition that leads to yellowing of leaves and stunted growth. It can be caused by a variety of factors, including nutrient deficiencies, diseases, and environmental stress. The condition is often linked to multiple underlying causes, but one common cause is the infection by the Sugarcane Yellow Leaf Virus (ScYLV).

    ### Symptoms:

    1.Yellowing of Older Leaves: Leaves, particularly older ones, turn yellow, often starting at the tips and moving inward.

    2.Stunted Growth: Infected plants may show reduced height and leaf development.

    3.Reduced Cane and Sugar Yield: Affected plants often have lower biomass and reduced sucrose content.

    4.Chlorosis: General yellowing of leaves due to chlorophyll degradation.

    5.Vein Yellowing: In some cases, the veins on the leaves may show a more distinct yellowing, with the rest of the leaf remaining green.


    ### Causes:

    1. Sugarcane Yellow Leaf Virus (ScYLV): This virus is primarily transmitted by aphids and is the main cause of yellowing in sugarcane.


    2. Nutrient Deficiencies: Lack of essential nutrients like nitrogen, magnesium, or potassium can lead to yellowing. Nitrogen deficiency is particularly common.


    3. Water Stress: Both waterlogging and drought conditions can stress the plant and lead to yellowing.


    4. Soil-Borne Diseases: Fungal and bacterial infections in the roots or stems can also cause yellowing as the plants struggle to absorb nutrients.



    ### Prevention:

    1. Use of Resistant Varieties: Select varieties that are resistant to the Sugarcane Yellow Leaf Virus (ScYLV) and other diseases linked to yellowing.


    2. Aphid Control: Since ScYLV is aphid-borne, controlling aphid populations using insecticides like Imidacloprid can help reduce virus transmission.


    3. Proper Fertilization: Ensure adequate fertilization with balanced nutrients, particularly nitrogen, magnesium, and potassium, to avoid deficiencies that can cause yellowing.


    4. Water Management: Maintain proper irrigation practices to avoid both waterlogging and drought stress.


    5. Field Sanitation: Remove and destroy infected plants and weeds that can harbor aphids or the virus.


    6. Crop Rotation: Rotate crops with non-host plants to break the cycle of disease transmission.



    ### Cure:

    1.No Direct Cure for Virus: Once plants are infected with the Sugarcane Yellow Leaf Virus, there is no effective cure. Management focuses on prevention.

    2.Remove Infected Plants: Early detection and removal of infected plants can help prevent the spread of the virus to healthy plants.

    3.Fungicide and Bactericide Use: For cases of yellowing caused by fungal or bacterial infections, fungicides or bactericides can be applied as per recommendations for specific diseases.

    4.Nutrient Supplements: Apply foliar sprays with micronutrients to correct deficiencies and improve plant health, particularly for magnesium, potassium, or nitrogen.

    5.Management of Environmental Stress: Provide consistent irrigation and avoid extreme water stress, which could exacerbate yellowing symptoms.


    ### Key Points:

    1.Virus Management: There is no cure for virus-induced yellowing, so managing aphid vectors and using resistant varieties are key.

    2.Nutrient and Environmental Control: Addressing nutrient deficiencies and ensuring good water management can help reduce stress-related yellowing.

    3.Regular Monitoring: Regularly inspect fields for early signs of yellowing and take action to control aphid populations and improve plant health.


    By integrating these preventive and corrective measures, sugarcane yellowing can be managed effectively to ensure healthy growth and optimal yields.""")                                    


                elif result=="Tomato_Early_blight":
                                st.header("Tomato Early Blight")
                                st.markdown("""
    Tomato Early Blight is a fungal disease caused by Alternaria solani. It primarily affects tomato plants, leading to premature leaf drop, reduced fruit quality, and yield loss. Early blight thrives in warm, humid conditions and can spread quickly under favorable environmental conditions.

    ### Symptoms:

    1. Dark Leaf Spots: Small, dark brown or black spots with concentric rings (target-like lesions) appear on older leaves.

    2. Leaf Yellowing: Yellowing typically surrounds the lesions, which can cause the leaves to die and fall off.

    3. Defoliation: Extensive leaf loss can lead to reduced photosynthesis and weak, stunted plants.

    4. Stem and Fruit Lesions: In severe cases, dark lesions may form on stems and fruit, particularly near the soil line, causing rotting and decay.

    5. Reduced Growth: Infected plants often show reduced growth and vigor, and the fruit may ripen unevenly.


    ### Prevention:

    1. Resistant Varieties: Use tomato varieties that are resistant or tolerant to early blight.


    2. Proper Spacing: Space plants adequately to improve air circulation, reducing humidity around the plants, which encourages fungal growth.


    3. Crop Rotation: Rotate crops to avoid planting tomatoes in the same location each year to break the disease cycle.


    4. Sanitation: Remove and destroy infected plant debris at the end of the season to prevent the fungus from overwintering in the field.


    5. Water Management: Water at the base of the plants and avoid overhead irrigation to prevent fungal spores from splashing onto healthy plants.


    6. Weed Control: Keep weeds under control, as they can harbor the fungus and increase the spread of disease.



    ### Cure:

    1. Fungicide Application:

    Chlorothalonil (e.g., Daconil) and Copper-based fungicides are effective for preventing and controlling early blight.

    Mancozeb and Azoxystrobin can also be used to manage the disease.

    Apply fungicides regularly, especially during wet weather conditions when the fungus thrives.



    2. Timely Application: Apply fungicides as a preventive measure or at the first sign of symptoms to minimize spread and protect healthy foliage.


    3. Pruning: Remove and destroy infected plant material, especially lower leaves that are more susceptible to the fungus.


    4. Biological Control: Use beneficial microorganisms such as Trichoderma or Bacillus subtilis as part of an integrated management approach.



    ### Key Points:

    1. Environmental Conditions: Early blight thrives in warm, wet conditions, so managing environmental factors can help limit disease spread.

    2. Early Detection: Regularly inspect plants for early symptoms and take prompt action to apply fungicides or remove infected material.

    3. Integrated Management: Combine cultural practices like crop rotation, spacing, and sanitation with fungicide applications to effectively manage early blight.


    By using these preventive measures, early detection, and timely fungicide application, Tomato Early Blight can be effectively managed to ensure healthy plants and good yields.""")
                    

                elif result=="Tomato_Bacterial_spot":
                    st.header("Tomato Bacterial Spot")
                    st.markdown("""
    Tomato Bacterial Spot is a disease caused by the bacterium Xanthomonas vesicatoria. It affects both tomatoes and peppers, causing significant damage to the leaves, stems, and fruit. This bacterial infection can reduce plant vigor, lower yields, and affect fruit quality.

    ### Symptoms:

    1. Leaf Spots: Small, water-soaked spots appear on the leaves, which turn brown or black as the disease progresses.

    2. Yellow Halos: A yellow ring often surrounds the dark, necrotic spots on the leaves.

    3. Leaf Necrosis: As the disease advances, the affected areas may die, leading to leaf drop.

    4. Fruit Lesions: On tomatoes, the fruit can develop small, sunken, dark spots that affect fruit quality.

    5. Reduced Growth: Infected plants often show stunted growth and a decrease in overall plant vigor.


    ### Prevention:

    1. Resistant Varieties: Plant tomato varieties resistant to bacterial spot.


    2. Use Certified Seed: Start with disease-free, certified seed or transplants to reduce the risk of initial infection.


    3. Crop Rotation: Rotate crops to avoid planting tomatoes in the same location as previous crops that have been affected by bacterial spot.


    4. Field Sanitation: Remove and destroy infected plant material, including fallen leaves and debris, to reduce bacterial spread.


    5. Proper Spacing: Ensure good air circulation between plants to reduce humidity around the foliage, which can favor bacterial growth.


    6. Avoid Overhead Irrigation: Use drip irrigation to minimize water splash that can spread bacteria from infected to healthy plants.


    7. Control Weeds: Weeds can harbor bacteria, so maintaining weed control is crucial to prevent the spread of the disease.



    ### Cure:

    1. No Direct Cure: There is no direct cure for bacterial spot once the infection is established. Management focuses on prevention and controlling the spread.

    2. Copper-based Fungicides: Use copper-based bactericides like Copper hydroxide or Copper oxychloride as a preventive measure or at the first sign of infection. These can help reduce the bacterial load.

    3. Antibiotics: In some cases, antibiotics like Streptomycin may be used, but their use is often restricted in certain areas due to resistance issues.

    4. Biological Control: Biological agents such as Bacillus subtilis or Streptomyces species can help reduce bacterial populations and protect healthy plants.

    5. Remove Infected Plants: Practice "rogueing" by removing and destroying infected plants to reduce the bacterial inoculum in the field.


    ### Key Points:

    1. Preventive Measures: Prevention through resistant varieties, good field hygiene, and proper irrigation practices are key to managing bacterial spot.

    2. Regular Monitoring: Regularly inspect plants for early symptoms and take action to limit the spread of infection.

    3. Fungicide and Bactericide Use: Copper-based products can be effective when used early or preventively in managing bacterial infections.


    By focusing on prevention and using cultural practices, fungicides, and proper sanitation, you can manage Tomato Bacterial Spot and reduce its impact on your crop.""")



                elif result=="Tomato_Late_blight":
                    st.header("Tomato Late Blight")
                    st.markdown("""
    Tomato Late Blight is a severe fungal disease caused by Phytophthora infestans, which affects both tomatoes and potatoes. It is one of the most destructive diseases in tomato production, especially in cool, wet conditions. Late blight can cause rapid damage, leading to complete crop loss if not managed properly.

    ### Symptoms:

    1. Water-soaked Spots on Leaves: Initially, small, dark, water-soaked spots appear on the older leaves, which spread quickly.

    2. Leaf Blighting: The spots expand and turn brown to black, with a characteristic yellow halo around them.

    3. White Fungal Growth: Under humid conditions, a white, fuzzy fungal growth may appear on the undersides of affected leaves.

    4. Stem Lesions: Dark lesions can form on stems, particularly near the soil line, causing the plant to collapse.

    5. Fruit Rot: Infected fruits develop dark, sunken spots that can lead to soft rot, often turning mushy and discolored.

    6. Rapid Decline: The disease can rapidly defoliate the plant, reducing photosynthesis and weakening the plant's ability to produce fruit.


    ### Prevention:

    1. Resistant Varieties: Use tomato varieties that are resistant to late blight or have some level of tolerance to the disease.


    2. Proper Spacing and Airflow: Plant tomatoes with enough spacing to improve air circulation and reduce humidity around the plants, which promotes fungal growth.


    3. Crop Rotation: Rotate tomatoes with non-host crops (e.g., beans, corn) to break the disease cycle and reduce pathogen buildup in the soil.


    4. Sanitation: Remove and destroy infected plant material, including fallen leaves and fruits, to reduce sources of infection.


    5. Avoid Overhead Irrigation: Use drip irrigation to keep foliage dry, as water splashing from overhead irrigation can spread the pathogen.


    6. Monitor Weather: Late blight thrives in cool, wet conditions. Monitoring weather and taking preventive measures when conditions are favorable for disease development can help reduce risk.



    ### Cure:

    1. Fungicide Application:

    Chlorothalonil, Mancozeb, Copper-based fungicides, and Azoxystrobin are effective fungicides for controlling late blight.

    Apply fungicides preventively or at the first sign of symptoms to protect healthy tissue.



    2. Timely and Frequent Spraying: Fungicides should be applied regularly, especially during periods of high humidity and rain, which encourage the spread of the disease.


    3. Remove Infected Plants: Practice "rogueing" by removing and destroying infected plants immediately to prevent further spread within the field.


    4. Biological Control: Biological agents like Trichoderma spp., Bacillus subtilis, or Pseudomonas fluorescens can provide some control over fungal pathogens.



    ### Key Points:

    1. Rapid Spread: Late blight can spread quickly under favorable conditions, so timely action is crucial.

    2. Integrated Management: Use a combination of resistant varieties, fungicide applications, and cultural practices like proper spacing and sanitation for effective management.

    3. Early Detection: Regularly inspect plants for symptoms and act quickly to limit the spread of the disease.


    By using these preventive and curative measures, Tomato Late Blight can be effectively controlled to minimize yield loss and protect your tomato crop.""")



                elif result=="Tomato_Septoria_leaf_spot":
                    st.header("Tomato Septoria Leaf Spot")
                    st.markdown("""

    Tomato Septoria Leaf Spot is a fungal disease caused by Septoria lycopersici. It is one of the most common diseases in tomatoes, particularly in humid and rainy conditions. The disease primarily affects the leaves, reducing photosynthesis and leading to premature leaf drop, which ultimately affects tomato yield and fruit quality.

    ### Symptoms:

    1. Small, Water-soaked Spots: Initially, small, circular, water-soaked spots (1/4 to 1/2 inch in diameter) appear on the lower leaves.

    2. Dark Brown or Gray Centers: The spots enlarge and become dark brown or grayish with a yellow halo around the edges.

    3. Leaf Yellowing: As the disease progresses, the tissue around the spots becomes yellow, leading to chlorosis (yellowing) of the leaves.

    4. Leaf Drop: In severe infections, the affected leaves may die and fall off, causing defoliation and reducing the plant's photosynthetic capacity.

    5. Spread to Upper Leaves: The disease spreads from lower leaves to upper leaves as the fungus continues to infect the plant.

    6. No Fruit Impact: While the disease primarily affects leaves, it does not usually cause direct damage to fruits unless the plant is severely weakened.


    ### Prevention:

    1. Resistant Varieties: Plant tomato varieties resistant to Septoria leaf spot if available in your region.


    2. Proper Spacing: Space plants adequately to improve airflow and reduce humidity around the foliage, which encourages fungal growth.


    3. Crop Rotation: Rotate tomatoes with non-host crops (such as beans or corn) to break the disease cycle and reduce inoculum buildup in the soil.


    4. Field Sanitation: Remove and destroy infected plant material, including fallen leaves, to reduce the source of infection.


    5. Avoid Overhead Irrigation: Use drip irrigation to avoid splashing water onto the foliage, which can spread the spores.


    6. Weed Control: Keep weeds under control, as they can harbor the fungal pathogen and serve as a reservoir for the disease.


    7. Mulching: Use mulch to reduce soil splash, which can spread the spores from the ground to the leaves.



    ### Cure:

    1. Fungicide Application:

    Fungicides such as Chlorothalonil, Mancozeb, and Copper-based fungicides are effective in controlling Septoria leaf spot.

    Azoxystrobin and Tebuconazole are also commonly used fungicides for control.

    Fungicides should be applied preventively or at the first sign of infection and repeated as recommended, especially during wet conditions.



    2. Pruning: Prune infected lower leaves to reduce the spread of the disease and improve air circulation in the plant canopy.


    3. Remove Infected Plants: Remove and destroy severely infected plants to prevent further spread of the disease.


    4. Biological Control: Use biological fungicides like Bacillus subtilis or Trichoderma spp. as part of an integrated pest management approach to suppress fungal growth.



    ### Key Points:

    1. Early Detection: Septoria leaf spot spreads rapidly under humid conditions, so early detection and prompt action are critical to minimizing damage.

    2. Cultural Practices: Cultural controls like proper spacing, crop rotation, and field sanitation are essential to preventing the spread of the disease.

    3. Fungicide Use: Regular fungicide application, especially during wet periods, is effective in controlling Septoria leaf spot.


    By implementing these preventive measures and using fungicides effectively, you can manage Tomato Septoria Leaf Spot and protect your crop from significant yield loss.""")

                elif result=="Tomato__healthy":
                    st.header("Your Tomato Crop is Healthy")
                    st.markdown(""" 
                    Your crop is thriving, and to boost your yield further, applying the right fertilizer is essential.
                    Our fertilizer recommendation system can assist you in selecting the most suitable fertilizer for optimal growth.  """)
            
                elif result=="Wheat_Brown_rust":
                    st.header("Wheat Brown Rust")
                    st.markdown("""
    Wheat Brown Rust, also known as Leaf Rust, is caused by the fungal pathogen Puccinia triticina. It is a significant disease of wheat that can lead to yield losses, particularly in susceptible varieties under favorable conditions for disease development. Warm temperatures and high humidity can accelerate the spread of brown rust.

    ### Symptoms:

    1.Rust Pustules: Small, circular to oval, orange-brown pustules appear on the upper leaf surface. These pustules can also appear on leaf sheaths and stems in severe cases.

    2.Scattered Distribution: The pustules are scattered and may cover large portions of the leaf as the disease progresses.

    3.Leaf Yellowing: Surrounding leaf tissue may turn yellow or brown as the infection progresses, leading to reduced photosynthesis.

    4.Premature Leaf Senescence: Severe infections can cause leaves to die early, reducing the plant's ability to produce grain effectively.

    5.Reduced Grain Quality: Infected plants may produce smaller, shriveled grains due to limited nutrient availability.


    ### Prevention:

    1. Resistant Varieties: Plant wheat varieties that are resistant or moderately resistant to brown rust.


    2. Crop Rotation: Practice crop rotation to break the disease cycle and reduce the pathogen load in the soil.


    3. Timely Planting: Plant wheat at an optimal time to avoid peak conditions favorable for rust development.


    4. Field Monitoring: Regularly inspect fields for early signs of brown rust to take timely action.


    5. Adequate Nutrition: Maintain proper plant nutrition to improve plant vigor and reduce susceptibility to disease.



    ### Cure:

    1. Fungicide Application:

    Triazole fungicides such as Tebuconazole and Propiconazole are effective in controlling brown rust.

    Strobilurin-based fungicides like Azoxystrobin can also help manage the disease.

    Apply fungicides preventively or at the first sign of symptoms to protect healthy foliage.



    2. Integrated Fungicide Management: Use fungicides in rotation or in combination to prevent the development of resistance.


    3. Remove Volunteer Wheat: Eliminate volunteer wheat plants that can harbor the pathogen and serve as a source of infection for new crops.


    4. Avoid Overuse of Nitrogen: Excessive nitrogen can increase the susceptibility of wheat plants to rust diseases, so balanced fertilization is essential.



    ### Key Points:

    1.Weather Monitoring: Since brown rust thrives in warm and humid conditions, monitoring weather forecasts can help predict outbreaks and inform fungicide application timing.

    2.Resistance Management: Rotate fungicides with different modes of action to prevent resistance buildup in the fungal population.

    3.Regular Scouting: Early detection and timely fungicide application can help manage brown rust effectively and prevent significant yield loss.


    Implementing these preventive and management strategies can help control Wheat Brown Rust and maintain healthy, productive crops.
    """)

                elif result=="Wheat_healthy":
                    st.header("Your Wheat Crop is Healthy")
                    st.markdown(""" 
                    Your crop is thriving, and to boost your yield further, applying the right fertilizer is essential.
                    Our fertilizer recommendation system can assist you in selecting the most suitable fertilizer for optimal growth.  """)
                

                elif result=="Wheat_Loose_Smut":
                    st.header("Wheat Loose Smut Disease")
                    st.markdown("""
    Wheat Loose Smut is a fungal disease caused by Ustilago tritici. It infects wheat plants systemically, leading to severe yield losses if not controlled. The fungus replaces the kernels in the wheat heads with black, powdery spore masses.

    ### Symptoms:

    1.Black, Powdery Spikes: Infected wheat spikes appear normal at first but soon develop into black, sooty masses as the grain heads emerge.

    2.No Grain Formation: The fungal spores replace the developing kernels, preventing grain formation.

    3.Early Ripening Appearance: Infected plants may appear to mature faster than healthy plants due to the premature emergence of the smut spores.


    ### Prevention:

    1. Use Disease-Free Seed: Plant certified, disease-free seeds or seeds treated to remove the fungal spores.


    2. Resistant Varieties: Use wheat varieties that are resistant to loose smut.


    3. Crop Rotation: Implement crop rotation with non-host crops to break the disease cycle.


    4. Seed Treatment: Treat seeds with systemic fungicides to prevent infection.



    ### Cure:

    1. Fungicide Seed Treatment:

    Carboxin-based fungicides (e.g., Vitavax)

    Tebuconazole or Triadimenol as effective treatments



    2. Hot Water Seed Treatment: For organic management, soak seeds in hot water (52-54°C) for 10-15 minutes to kill the fungus, though this method must be done carefully to avoid damaging the seed.


    3. Early Intervention: Once the symptoms appear in the field, it is too late to cure affected plants; prevention through seed treatment is essential.



    ### Key Points:

    1.Systemic Nature: The fungus infects wheat at the seedling stage and remains dormant until the plant reaches maturity, so treating seeds is crucial to prevention.

    2.Long-term Control: Regularly using treated seeds and rotating crops can effectively reduce the incidence of loose smut.

    3.Proper Seed Storage: Ensure seeds are stored in a dry, clean environment to prevent contamination.


    By following these preventive measures and treating seeds appropriately, wheat crops can be protected from the damaging effects of loose smut.""")

                elif result=="Wheat_Yellow_rust":
                    st.header("Wheat Yellow Rust")
                    st.markdown(""" 
    Wheat Yellow Rust, also known as Wheat Stripe Rust, is a fungal disease caused by Puccinia striiformis f. sp. tritici. It thrives in cool, moist environments and can lead to significant yield losses if not properly managed.

    ### Symptoms:

    1.Yellow Stripes: Long, narrow yellow pustules form in stripes along the veins on the upper surface of leaves, giving the disease its name.

    2.Leaf Drying: Heavily infected leaves may wither and die prematurely.

    3.Reduced Grain Quality: Severe infections can reduce grain weight and quality.

    4.Pustules on Stems and Glumes: In some cases, pustules can appear on stems and the head’s glumes.


    ### Prevention:

    1. Resistant Varieties: Plant wheat varieties that are resistant or tolerant to yellow rust.


    2. Early Sowing: Sow wheat early to help plants reach a more resistant stage before the weather becomes conducive to the disease.


    3. Field Monitoring: Regularly scout fields, especially during cool, wet periods, to detect early signs of the disease.


    4. Crop Rotation: Rotate crops to reduce the presence of rust spores in the soil.


    5. Remove Volunteer Plants: Eradicate volunteer wheat and other grass hosts that can harbor rust between growing seasons.



    ### Cure:

    1. Fungicide Application:

    Triazole-based fungicides (e.g., Tebuconazole, Propiconazole)

    Strobilurin-based fungicides (e.g., Azoxystrobin)

    Combination products like Tebuconazole + Azoxystrobin for enhanced control



    2. Timely Spraying: Apply fungicides at the first sign of infection or as a preventive measure when conditions are favorable for disease development.


    3. Integrated Disease Management: Combine cultural practices and fungicide applications to manage the disease effectively.



    ### Key Points:

    1.Monitor Weather: Yellow rust thrives in cool (10-15°C) and wet conditions, so monitoring weather can help plan preventive measures.

    2.Resistance Breakdown: The fungus can develop new strains that overcome resistance, so it’s important to use multiple management strategies.

    3.Early Treatment: Prompt fungicide application at the first sign of yellow rust can help limit its spread and protect yields.


    Using a combination of resistant varieties, good cultural practices, and timely fungicide application can help manage Wheat Yellow Rust and minimize its impact on wheat production.""")






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


    elif mode=="🧪Fertilizer recommender":
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
