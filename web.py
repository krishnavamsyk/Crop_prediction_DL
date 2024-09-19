import streamlit as st


st.sidebar.title("DASHBOARD")
mode=st.sidebar.selectbox("Select Page:",["Home","Disease-Recognition","About"])

if (mode=="Home"):
    st.title("Crop-Disease Prediction and Management ")
    st.write("By team AI-CRAFT")

    st.markdown('''Welcome........
                
                ''')
