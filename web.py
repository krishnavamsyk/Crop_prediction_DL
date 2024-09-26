import streamlit as st

#dashboard
st.sidebar.title("DASHBOARD")
mode=st.sidebar.selectbox("Select Page:",["Home","Disease-Recognition","About"])


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
