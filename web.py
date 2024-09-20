import streamlit as st

#dashboard
st.sidebar.title("DASHBOARD")
mode=st.sidebar.selectbox("Select Page:",["Home","Disease-Recognition","About"])

#background image

# Define your custom CSS for the background
page_bg_img = '''
<style>
body {
background-image: url("https://images.unsplash.com/photo-1542281286-9e0a16bb7366");
background-size: cover;
background-repeat: no-repeat;
background-attachment: fixed;
}
</style>
'''


# Apply the CSS using st.markdown
st.markdown(page_bg_img, unsafe_allow_html=True)

st.markdown(page_bg_img, unsafe_allow_html=True)


if (mode=="Home"):
    st.title("Crop-Disease Prediction and Prevention ")
    st.write("By team AI-CRAFT")

    st.markdown('''
                
                ## Welcome.....
                

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


#About page
elif (mode=="About"):
    st.header("About")
    st.markdown('''The Dataset being used for training the model is a comprehensive collection of images depicting various diseases affecting major crops including **wheat, maize, cotton, sugarcane, and rice**.

The dataset contains a diverse range of crop disease images, meticulously curated from multiple sources to ensure completeness and relevance. It encompasses images of common and rare diseases afflicting each crop, captured at different stages of development and severity.")

### Contents:

1. Train dataset contains 15452 images
2. Test set contains 3172 images
3. There are a total of 42 classes in which these images are categorised into.
''')








