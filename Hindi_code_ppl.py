import streamlit as st
import geocoder
import requests
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import numpy as np
import base64
import time
import joblib
from streamlit_option_menu import option_menu
import tempfile
import os
from PIL import Image
from ultralytics import YOLO

# ─── Model cache (load once per session) ──────────────────────────────────────
@st.cache_resource
def load_yolo_model():
    return YOLO('Models/best.pt')

@st.cache_resource
def load_crop_model():
    return joblib.load('Models/Naive_bayes_crp.pkl')

@st.cache_resource
def load_fertilizer_model():
    return joblib.load('Models/svm_model.pkl_2')

# ─── Constants ────────────────────────────────────────────────────────────────
CLASSES = [
    'Apple__Apple_scab', 'Apple_Black_rot', 'Apple_healthy',
    'Corn_Blight', 'Corn_Common_Rust', 'Corn_Gray_Leaf_Spot', 'Corn_Healthy',
    'Cotton_Healthy', 'Cotton_bacterial_blight', 'Cotton_curl_virus',
    'Grape_Black_rot', 'Grape_Leaf_blight(Isariopsis_Leaf_Spot)', 'Grape__healthy',
    'Pepper,_bell_Bacterial_spot', 'Pepper,_bell_healthy',
    'Potato_Early_blight', 'Potato_Late_blight', 'Potato_healthy',
    'Rice_Healthy', 'Rice_bacterial_leaf_blight', 'Rice_brown_spot', 'Rice_leaf_blast',
    'Sugarcan_Mosaic', 'Sugarcane_Healthy', 'Sugarcane_RedRot',
    'Sugarcane_Rust', 'Sugarcane_Yellow',
    'Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Late_blight',
    'Tomato_Septoria_leaf_spot', 'Tomato__healthy',
    'Wheat_Brown_rust', 'Wheat_Healthy', 'Wheat_Loose_Smut', 'Wheat_Yellow_rust'
]

HEALTHY_CLASSES = {c for c in CLASSES if 'healthy' in c.lower() or 'Healthy' in c}

SAMPLE_IMAGES = {
    "🍅 टमाटर अर्ली ब्लाइट":  "samples/tomato_early_blight.jpg",
    "🌽 मक्का सामान्य रस्ट":   "samples/corn_common_rust.jpg",
    "🌾 गेहूं भूरी रस्ट":      "samples/wheat_brown_rust.jpg",
    "🥔 आलू लेट ब्लाइट":      "samples/potato_late_blight.jpg",
}

# ─── Disease info database (Hindi) ───────────────────────────────────────────
DISEASE_INFO_HI = {
    'Apple__Apple_scab': {
        "severity": "मध्यम", "spread": "कवक (Venturia inaequalis)",
        "symptoms": "पत्तियों और फलों पर जैतून-हरे से भूरे मखमली धब्बे; पत्तियाँ मुड़ जाती हैं; फलों पर गहरे उभरे घाव जो फट जाते हैं।",
        "prevention": "रोधी किस्में लगाएं · गिरी पत्तियाँ हटाएं · उचित छंटाई करें · शुरुआती वसंत में निवारक फफूंदनाशक लगाएं।",
        "cure": "फफूंदनाशक: myclobutanil, captan, या mancozeb। संक्रमित सामग्री हटाएं और वायु प्रवाह के लिए छंटाई करें।"
    },
    'Apple_Black_rot': {
        "severity": "उच्च", "spread": "कवक (Botryosphaeria obtusa)",
        "symptoms": "फल पर केंद्रित छल्लों वाले गहरे गोल धंसे घाव; पत्तियों पर बैंगनी किनारे वाले धब्बे; शाखाओं पर कैंकर।",
        "prevention": "गिरा हुआ मलबा हटाएं · संक्रमित शाखाएं काटें · निवारक फफूंदनाशक लगाएं।",
        "cure": "कैंकर से 6–12 इंच परे छंटाई करें। फफूंदनाशक: captan या thiophanate-methyl।"
    },
    'Corn_Blight': {
        "severity": "उच्च", "spread": "कवक (Bipolaris maydis / Exserohilum turcicum)",
        "symptoms": "पत्तियों पर तन, अंडाकार घाव (दक्षिणी) या बड़े सिगार-आकार के भूरे घाव (उत्तरी); पत्तियाँ समय से पहले मर जाती हैं।",
        "prevention": "रोधी संकर किस्में · फसल चक्र · संक्रमित मलबा हटाएं · उचित दूरी बनाए रखें।",
        "cure": "फफूंदनाशक: azoxystrobin या propiconazole लक्षण प्रकट होते ही लगाएं।"
    },
    'Corn_Common_Rust': {
        "severity": "मध्यम", "spread": "कवक (Puccinia sorghi)",
        "symptoms": "पत्तियों के दोनों तरफ छोटे अंडाकार लाल-भूरे फुंसी; पत्तियाँ समय से पहले सूख जाती हैं।",
        "prevention": "रोधी संकर किस्में · फसल चक्र · ठंडे नम मौसम में निगरानी।",
        "cure": "स्ट्रोबिलुरिन या ट्राइजोल युक्त फफूंदनाशक। जल्दी उपयोग करें।"
    },
    'Corn_Gray_Leaf_Spot': {
        "severity": "उच्च", "spread": "कवक (Cercospora zeae-maydis)",
        "symptoms": "पत्ती नसों के समानांतर छोटे आयताकार भूरे-तन घाव जो मिलकर बड़े मृत क्षेत्र बनाते हैं।",
        "prevention": "रोधी संकर किस्में · फसल चक्र · अवशेष जुताई · उचित पौध दूरी।",
        "cure": "फफूंदनाशक: azoxystrobin, pyraclostrobin, propiconazole। VT से R1 वृद्धि चरण में लगाएं।"
    },
    'Cotton_bacterial_blight': {
        "severity": "उच्च", "spread": "जीवाणु (Xanthomonas citri pv. malvacearum)",
        "symptoms": "पत्तियों पर कोणीय पानी-भीगे घाव जो भूरे/काले हो जाते हैं; तनों और टिंडों पर गहरे धंसे धब्बे; पत्ती झड़ना।",
        "prevention": "रोधी किस्में · प्रमाणित रोगमुक्त बीज · फसल चक्र · सिंचाई में सावधानी।",
        "cure": "कॉपर हाइड्रॉक्साइड (Kocide) या कॉपर ऑक्सीक्लोराइड छिड़काव। नियमित खेत निगरानी।"
    },
    'Cotton_curl_virus': {
        "severity": "उच्च", "spread": "विषाणु — सफेद मक्खी (Bemisia tabaci) द्वारा",
        "symptoms": "पत्तियों का ऊपर की ओर मुड़ना; पीलापन; पौधों का बौना होना; मोज़ेक पैटर्न; पत्तियों का समय से पहले गिरना।",
        "prevention": "सफेद मक्खी नियंत्रण (imidacloprid, thiamethoxam) · रोधी किस्में · संक्रमित पौधे हटाएं।",
        "cure": "एक बार संक्रमित होने पर कोई इलाज नहीं। सफेद मक्खी नियंत्रण पर ध्यान दें। संक्रमित पौधे उखाड़ें।"
    },
    'Grape_Black_rot': {
        "severity": "उच्च", "spread": "कवक (Guignardia bidwellii)",
        "symptoms": "अंगूर पर गहरे गोल धंसे घाव; पत्तियों पर पीले प्रभामंडल वाले छोटे गहरे घाव; फल सिकुड़कर काला हो जाता है।",
        "prevention": "रोधी किस्में · संक्रमित सामग्री हटाएं · वायु प्रवाह सुनिश्चित करें · फूल आने से पहले फफूंदनाशक।",
        "cure": "फफूंदनाशक: captan, chlorothalonil, या mancozeb। संक्रमित क्षेत्रों की छंटाई।"
    },
    'Grape_Leaf_blight(Isariopsis_Leaf_Spot)': {
        "severity": "मध्यम", "spread": "कवक (Phomopsis viticola)",
        "symptoms": "पत्तियों पर हल्के भूरे केंद्र वाले छोटे गहरे गोल घाव; अंकुरों पर भूरे धंसे घाव; फल गुच्छों पर पानी-भीगे धब्बे।",
        "prevention": "रोधी किस्में · संक्रमित बेलों की छंटाई · पर्याप्त वायु प्रवाह · शुरुआती मौसम में फफूंदनाशक।",
        "cure": "फफूंदनाशक: captan, chlorothalonil, myclobutanil। कली फूटने से पहले छिड़काव शुरू करें।"
    },
    'Pepper,_bell_Bacterial_spot': {
        "severity": "उच्च", "spread": "जीवाणु (Xanthomonas euvesicatoria)",
        "symptoms": "पत्तियों पर पीले प्रभामंडल वाले कोणीय पानी-भीगे घाव; फल पर गहरे धंसे घाव; पत्ती झड़ना; पौधे का बौना होना।",
        "prevention": "रोधी किस्में · रोगमुक्त बीज · फसल चक्र · ड्रिप सिंचाई · उचित दूरी।",
        "cure": "कॉपर हाइड्रॉक्साइड या सल्फेट जीवाणुनाशक। जैविक नियंत्रण: Bacillus subtilis। संक्रमित हिस्से हटाएं।"
    },
    'Potato_Early_blight': {
        "severity": "मध्यम", "spread": "कवक (Alternaria solani)",
        "symptoms": "पुरानी पत्तियों पर केंद्रित छल्लों वाले गहरे भूरे-काले धब्बे (निशाने जैसे); घावों के आसपास पीलापन; कंद संक्रमण।",
        "prevention": "फसल चक्र · रोधी किस्में · संक्रमित मलबा हटाएं · ऊपरी सिंचाई से बचें।",
        "cure": "फफूंदनाशक: chlorothalonil, mancozeb, azoxystrobin। गर्म नम मौसम में निवारक उपचार।"
    },
    'Potato_Late_blight': {
        "severity": "अति-गंभीर", "spread": "ओमाइसीट (Phytophthora infestans)",
        "symptoms": "पत्तियों पर पानी-भीगे हल्के हरे से गहरे भूरे घाव; पत्ती की निचली सतह पर सफेद रोएँदार फफूंद; कंद सड़न।",
        "prevention": "रोधी किस्में · प्रमाणित बीज आलू · फसल चक्र · उचित दूरी · ऊपरी सिंचाई से बचें।",
        "cure": "फफूंदनाशक: chlorothalonil, mancozeb, metalaxyl (Ridomil Gold)। जैविक प्रबंधन में कॉपर स्प्रे।"
    },
    'Rice_Healthy': {
        "severity": "कोई नहीं", "spread": "लागू नहीं",
        "symptoms": "कोई रोग नहीं पाया गया।",
        "prevention": "अच्छी प्रथाएं जारी रखें।",
        "cure": "लागू नहीं"
    },
    'Rice_bacterial_leaf_blight': {
        "severity": "उच्च", "spread": "जीवाणु (Xanthomonas oryzae pv. oryzae)",
        "symptoms": "पत्ती के किनारों पर पानी-भीगी धारियाँ जो पीली फिर भूरी होती हैं; V-आकार के घाव; पत्तियाँ सूखती हैं; पौधों का बौना होना।",
        "prevention": "रोधी किस्में · प्रमाणित बीज · फसल अवशेष हटाएं · संतुलित उर्वरक (नाइट्रोजन अधिक न करें)।",
        "cure": "कॉपर-आधारित जीवाणुनाशक। स्ट्रेप्टोमाइसिन/ऑक्सीटेट्रासाइक्लिन छिड़काव। जैविक: Pseudomonas fluorescens।"
    },
    'Rice_brown_spot': {
        "severity": "मध्यम", "spread": "कवक (Bipolaris oryzae)",
        "symptoms": "पत्तियों पर भूरे गोल धब्बे जिनका केंद्र भूरा-स्लेटी और किनारे गहरे भूरे; अनाज का रंग बदलना।",
        "prevention": "पर्याप्त NPK पोषण · प्रमाणित बीज · उचित सिंचाई · फसल चक्र।",
        "cure": "फफूंदनाशक: mancozeb, tricyclazole, propiconazole। बीज उपचार: carbendazim या thiram।"
    },
    'Rice_leaf_blast': {
        "severity": "अति-गंभीर", "spread": "कवक (Magnaporthe oryzae)",
        "symptoms": "पत्तियों पर हीरे/धुरी आकार के घाव; कॉलर रॉट; गंभीर मामलों में गांठ और पुष्पगुच्छ ब्लास्ट।",
        "prevention": "रोधी किस्में · संतुलित नाइट्रोजन · स्थिर जल स्तर · भीड़ से बचें।",
        "cure": "फफूंदनाशक: tricyclazole, isoprothiolane (Fuji-one), edifenphos (Hinosan)। सिलिकॉन उर्वरण।"
    },
    'Sugarcan_Mosaic': {
        "severity": "मध्यम", "spread": "विषाणु (SCMV) — एफिड द्वारा",
        "symptoms": "पत्तियों पर हल्के और गहरे हरे मोज़ेक पैटर्न; पौधों का बौना होना; पत्ती विकृति; गन्ने की उपज में कमी।",
        "prevention": "प्रमाणित रोगमुक्त बीज गन्ना · रोधी किस्में · एफिड नियंत्रण (imidacloprid) · संक्रमित पौधे हटाएं।",
        "cure": "कोई रासायनिक इलाज नहीं। एफिड वाहक प्रबंधन। संक्रमित पौधे उखाड़ें (रोगिंग)।"
    },
    'Sugarcane_RedRot': {
        "severity": "अति-गंभीर", "spread": "कवक (Colletotrichum falcatum)",
        "symptoms": "चिरे तने में सफेद धब्बों के साथ लाल रंग; बदबूदार स्पंजी तने; ऊपर से नीचे की ओर पत्तियों का पीला पड़ना।",
        "prevention": "रोधी किस्में · संक्रमित सामग्री हटाएं · प्रमाणित बीज गन्ना · पर्याप्त जल निकासी।",
        "cure": "फफूंदनाशक: carbendazim, thiophanate-methyl। गर्म पानी उपचार: 52°C पर 30 मिनट। जैविक: Trichoderma spp।"
    },
    'Sugarcane_Rust': {
        "severity": "मध्यम", "spread": "कवक (Puccinia melanocephala / P. kuehnii)",
        "symptoms": "पत्तियों की निचली सतह पर लाल-भूरी या नारंगी फुंसियाँ; पीलापन; समय से पहले पत्तियों का मरना।",
        "prevention": "रोधी किस्में · संक्रमित मलबा हटाएं · संतुलित उर्वरक · पर्याप्त दूरी।",
        "cure": "फफूंदनाशक: propiconazole, tebuconazole, या mancozeb। पहले लक्षण पर लगाएं।"
    },
    'Sugarcane_Yellow': {
        "severity": "मध्यम", "spread": "विषाणु (ScYLV) — एफिड से या पोषण की कमी",
        "symptoms": "पुरानी पत्तियों का सिरों से पीला पड़ना; बौना विकास; नस का पीलापन; सुक्रोज में कमी।",
        "prevention": "रोधी किस्में · एफिड नियंत्रण · संतुलित NPK पोषण · उचित सिंचाई।",
        "cure": "विषाणु का कोई इलाज नहीं। एफिड प्रबंधन, पत्तेदार पोषक छिड़काव, संक्रमित पौधे हटाएं।"
    },
    'Tomato_Bacterial_spot': {
        "severity": "उच्च", "spread": "जीवाणु (Xanthomonas vesicatoria)",
        "symptoms": "पत्तियों पर पीले प्रभामंडल के साथ पानी-भीगे धब्बे; पत्ती नेक्रोसिस; फल पर छोटे गहरे धंसे धब्बे।",
        "prevention": "रोधी किस्में · प्रमाणित बीज · फसल चक्र · ड्रिप सिंचाई · खेत स्वच्छता।",
        "cure": "कॉपर हाइड्रॉक्साइड जीवाणुनाशक। जैविक: Bacillus subtilis। संक्रमित पौधे हटाएं।"
    },
    'Tomato_Early_blight': {
        "severity": "मध्यम", "spread": "कवक (Alternaria solani)",
        "symptoms": "पुरानी पत्तियों पर केंद्रित छल्लों वाले गहरे धब्बे; पीलापन; पत्ती झड़ना; तने और फल पर गहरे घाव।",
        "prevention": "रोधी किस्में · उचित दूरी · फसल चक्र · संक्रमित मलबा हटाएं · ड्रिप सिंचाई।",
        "cure": "फफूंदनाशक: chlorothalonil, कॉपर-आधारित, mancozeb, azoxystrobin। निचली पत्तियों की छंटाई।"
    },
    'Tomato_Late_blight': {
        "severity": "अति-गंभीर", "spread": "ओमाइसीट (Phytophthora infestans)",
        "symptoms": "पत्तियों पर तेजी से फैलते पानी-भीगे काले धब्बे; आर्द्रता में सफेद रोएँदार फफूंद; तने के काले घाव; नरम फल सड़न।",
        "prevention": "रोधी किस्में · उचित दूरी · फसल चक्र · खेत स्वच्छता · ड्रिप सिंचाई।",
        "cure": "फफूंदनाशक: chlorothalonil, mancozeb, कॉपर-आधारित, azoxystrobin। नम मौसम में बार-बार लगाएं।"
    },
    'Tomato_Septoria_leaf_spot': {
        "severity": "मध्यम", "spread": "कवक (Septoria lycopersici)",
        "symptoms": "निचली पत्तियों पर गहरे भूरे/स्लेटी केंद्र और पीले प्रभामंडल वाले छोटे गोल धब्बे; पत्ती पीलापन; पत्ती झड़ना।",
        "prevention": "रोधी किस्में · उचित दूरी · फसल चक्र · खेत स्वच्छता · ड्रिप सिंचाई · मल्चिंग।",
        "cure": "फफूंदनाशक: chlorothalonil, mancozeb, azoxystrobin, tebuconazole। निचली पत्तियों की छंटाई।"
    },
    'Wheat_Brown_rust': {
        "severity": "उच्च", "spread": "कवक (Puccinia triticina)",
        "symptoms": "पत्तियों की ऊपरी सतह पर छोटी गोल नारंगी-भूरी फुंसियाँ; आसपास पीलापन; समय से पहले पत्ती मृत्यु; सिकुड़ा अनाज।",
        "prevention": "रोधी किस्में · फसल चक्र · समय पर बुवाई · संतुलित पोषण।",
        "cure": "ट्राइजोल: tebuconazole, propiconazole। स्ट्रोबिलुरिन: azoxystrobin। पहले लक्षण पर लगाएं।"
    },
    'Wheat_Loose_Smut': {
        "severity": "उच्च", "spread": "कवक (Ustilago tritici) — बीज-जनित",
        "symptoms": "बालियों में अनाज की जगह काला पाउडर; अनाज नहीं बनता; पौधे जल्दी पकते दिखते हैं।",
        "prevention": "प्रमाणित रोगमुक्त बीज · रोधी किस्में · प्रणालीगत फफूंदनाशक से बीज उपचार।",
        "cure": "बीज उपचार: carboxin (Vitavax), tebuconazole, या triadimenol। गर्म पानी उपचार: 52–54°C पर 10–15 मिनट।"
    },
    'Wheat_Yellow_rust': {
        "severity": "उच्च", "spread": "कवक (Puccinia striiformis)",
        "symptoms": "पत्ती नसों के साथ लंबी संकरी पीली फुंसी धारियाँ; पत्तियाँ सूख जाती हैं; अनाज का भार कम होता है।",
        "prevention": "रोधी किस्में · जल्दी बुवाई · ठंडे नम मौसम में नियमित निरीक्षण · फसल चक्र।",
        "cure": "ट्राइजोल: tebuconazole, propiconazole। स्ट्रोबिलुरिन: azoxystrobin। पहले लक्षण पर लगाएं।"
    },
}

SEVERITY_COLORS = {
    "कोई नहीं":    ("#1b5e20", "#66bb6a", "✅"),
    "मध्यम":      ("#e65100", "#ffb74d", "⚠️"),
    "उच्च":       ("#b71c1c", "#ef9a9a", "🔴"),
    "अति-गंभीर":  ("#4a148c", "#ce93d8", "🚨"),
}

# ─── Helper functions ─────────────────────────────────────────────────────────
def get_base64_image(image_path):
    norm_path = image_path.replace("\\", "/")
    if not os.path.exists(norm_path):
        return None
    with open(norm_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def model_prediction(test_image):
    model = load_yolo_model()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        image = Image.open(test_image).convert('RGB')
        image.save(tmp.name)
        tmp_path = tmp.name
    results = model(tmp_path)
    probs = results[0].probs.data.tolist()
    top_idx = int(np.argmax(probs))
    return CLASSES[top_idx], probs[top_idx], probs

def render_confidence_bar(confidence: float):
    pct   = int(confidence * 100)
    color = "#66bb6a" if confidence >= 0.7 else "#ffa726" if confidence >= 0.4 else "#ef5350"
    st.markdown(f"""
    <div style="margin-bottom:4px;">
        <span style="color:#a5d6a7;font-size:0.9rem;">विश्वास स्तर: <b style="color:{color};">{pct}%</b></span>
    </div>
    <div style="background:#1a3a1a;border-radius:10px;height:18px;width:100%;margin:6px 0 14px;">
        <div style="background:linear-gradient(90deg,{color},{color}aa);border-radius:10px;height:18px;width:{pct}%;"></div>
    </div>
    """, unsafe_allow_html=True)

def render_disease_card(result: str, confidence: float):
    info     = DISEASE_INFO_HI.get(result, {})
    severity = info.get("severity", "अज्ञात")
    bg, accent, icon = SEVERITY_COLORS.get(severity, ("#1a3a1a", "#4caf50", "ℹ️"))
    is_healthy = result in HEALTHY_CLASSES

    if is_healthy:
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,#1b5e20,#2e7d32);border-left:5px solid #66bb6a;
                    border-radius:12px;padding:20px 24px;margin:12px 0;">
            <h2 style="color:#a5d6a7;margin:0;">✅ आपकी फसल स्वस्थ है!</h2>
            <p style="color:#c8e6c9;margin-top:8px;">
                कोई रोग नहीं पाया गया। फसल की निगरानी जारी रखें और उपज बढ़ाने के लिए
                हमारे <b>उर्वरक अनुशंसा</b> टूल का उपयोग करें।
            </p>
        </div>
        """, unsafe_allow_html=True)
        return

    display_name = result.replace("_", " ").replace(",", ", ")
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,{bg},{bg}cc);border-left:5px solid {accent};
                border-radius:12px;padding:20px 24px;margin:12px 0;">
        <h2 style="color:{accent};margin:0;">{icon} {display_name}</h2>
        <span style="background:{accent}33;color:{accent};padding:3px 10px;border-radius:12px;
                     font-size:0.8rem;font-weight:600;">गंभीरता: {severity}</span>
        <span style="background:#1a3a1a;color:#81c784;padding:3px 10px;border-radius:12px;
                     font-size:0.8rem;margin-left:8px;">{info.get('spread','')}</span>
    </div>
    """, unsafe_allow_html=True)

    if info:
        tab1, tab2, tab3 = st.tabs(["🔍 लक्षण", "🛡️ रोकथाम", "💊 उपचार"])
        with tab1:
            st.markdown(f"<p style='color:#c8e6c9;'>{info['symptoms']}</p>", unsafe_allow_html=True)
        with tab2:
            for step in info['prevention'].split(' · '):
                st.markdown(f"- {step}")
        with tab3:
            st.markdown(f"<p style='color:#c8e6c9;'>{info['cure']}</p>", unsafe_allow_html=True)

def get_weather_forecast(lat, lon):
    API_KEY = st.secrets["OPENWEATHER_API_KEY"]
    URL = f"http://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={API_KEY}&units=metric"
    response = requests.get(URL)
    if response.status_code == 200:
        data = response.json()
        return pd.DataFrame([{
            'datetime':      datetime.fromtimestamp(f['dt']),
            'temperature':   f['main']['temp'],
            'humidity':      f['main']['humidity'],
            'precipitation': f.get('rain', {}).get('3h', 0),
            'windspeed':     f['wind']['speed']
        } for f in data.get('list', [])])
    st.error("मौसम डेटा प्राप्त करने में विफल।")
    return pd.DataFrame()

def plot_weather_forecast(forecast_df):
    fig, axes = plt.subplots(4, 1, figsize=(10, 12))
    fig.patch.set_facecolor('#0f1a0f')
    configs = [
        ('temperature',   'tab:red',    'तापमान (°C)'),
        ('humidity',      'tab:blue',   'आर्द्रता (%)'),
        ('precipitation', 'tab:green',  'वर्षा (mm)'),
        ('windspeed',     'tab:orange', 'हवा की गति (m/s)'),
    ]
    for ax, (col, color, label) in zip(axes, configs):
        ax.set_facecolor('#1a3a1a')
        ax.plot(forecast_df['datetime'], forecast_df[col], color=color, linewidth=2)
        ax.fill_between(forecast_df['datetime'], forecast_df[col], alpha=0.15, color=color)
        ax.set_ylabel(label, color=color, fontsize=10)
        ax.tick_params(axis='y', labelcolor=color)
        ax.tick_params(axis='x', colors='#81c784', rotation=30)
        ax.grid(True, alpha=0.2, color='#4caf50')
        ax.spines[:].set_color('#2d5a2d')
        ax.set_title(f"{label} पूर्वानुमान", color='#81c784', fontsize=11, fontweight='bold', pad=6)
    plt.tight_layout(pad=2.0)
    st.pyplot(fig)

# ─── Main app ─────────────────────────────────────────────────────────────────
def Hindi():

    with st.sidebar:
        mode = option_menu(
            menu_title=None,
            options=["🏡 होम", "👤 परिचय", "🦠 रोग पहचान",
                     "🌥️ 5-दिन पूर्वानुमान", "🌱 फसल सुझाव",
                     "🧪 उर्वरक सुझाव", "👥 टीम"],
            styles={
                "container":         {"background-color": "transparent"},
                "icon":              {"color": "#81c784", "font-size": "16px"},
                "nav-link":          {"color": "#c8e6c9", "font-size": "14px", "--hover-color": "#2d5a2d"},
                "nav-link-selected": {"background-color": "#2e7d32", "color": "white"},
            }
        )
        st.markdown("<hr style='border-color:#2d5a2d;margin:20px 0;'>", unsafe_allow_html=True)
        st.markdown("""
        <div style='text-align:center;padding:10px;'>
            <h2 style='color:#66bb6a;margin:0;'>🌾 GreenWatch</h2>
            <p style='color:#81c784;font-size:0.8rem;margin:4px 0;'>AI कृषि स्वास्थ्य सलाहकार</p>
            <p style='color:#4caf50;font-size:0.75rem;'>by AI-Craft</p>
        </div>
        """, unsafe_allow_html=True)

        if mode == "🦠 रोग पहचान":
            st.markdown("<hr style='border-color:#2d5a2d;'>", unsafe_allow_html=True)
            st.markdown("<p style='color:#81c784;font-weight:600;'>🌿 नमूना छवि आज़माएं</p>", unsafe_allow_html=True)
            sample_choice = st.selectbox("नमूना चुनें:", list(SAMPLE_IMAGES.keys()), label_visibility="collapsed")
            if st.button("नमूना लोड करें"):
                sample_path = SAMPLE_IMAGES[sample_choice]
                if os.path.exists(sample_path):
                    st.session_state['sample_image_hi'] = sample_path
                    st.session_state['sample_name_hi']  = sample_choice
                else:
                    st.warning("नमूना छवि उपलब्ध नहीं है।")

    # ── होम ───────────────────────────────────────────────────────────────────
    if mode == "🏡 होम":
        st.markdown("<h1 style='text-align:center;'>🌾 GreenWatch: AI कृषि स्वास्थ्य सलाहकार</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align:center;color:#81c784;font-size:1.1rem;'>अपनी फसल की रक्षा करें। अपनी उपज सुरक्षित करें।</p>", unsafe_allow_html=True)
        st.markdown("<hr>", unsafe_allow_html=True)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown('<div class="metric-card"><h3>36</h3><p>रोग वर्ग</p></div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="metric-card"><h3>10+</h3><p>फसल प्रकार</p></div>', unsafe_allow_html=True)
        with col3:
            st.markdown('<div class="metric-card"><h3>98%+</h3><p>मॉडल सटीकता</p></div>', unsafe_allow_html=True)
        with col4:
            st.markdown('<div class="metric-card"><h3>2</h3><p>भाषाएं</p></div>', unsafe_allow_html=True)

        st.markdown("### 🌱 यह कैसे काम करता है")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**📤 1. छवि अपलोड करें**\n\nअपनी फसल की बीमार पत्ती की फोटो लें और अपलोड करें।")
        with c2:
            st.markdown("**🧠 2. AI विश्लेषण**\n\nहमारा YOLOv8 मॉडल छवि का विश्लेषण करके रोग की पहचान करता है।")
        with c3:
            st.markdown("**📋 3. मार्गदर्शन पाएं**\n\nलक्षण, रोकथाम और उपचार के साथ तत्काल निदान पाएं।")

        st.markdown("### 🌾 समर्थित फसलें")
        crops = ["🍎 सेब", "🌽 मक्का", "🧶 कपास", "🍇 अंगूर",
                 "🫑 शिमला मिर्च", "🥔 आलू", "🌾 चावल", "🎋 गन्ना",
                 "🍅 टमाटर", "🌾 गेहूं"]
        cols = st.columns(5)
        for i, crop in enumerate(crops):
            cols[i % 5].markdown(
                f"<div style='background:#1e3a1e;border-radius:8px;padding:8px;"
                f"text-align:center;margin:4px;color:#c8e6c9;'>{crop}</div>",
                unsafe_allow_html=True
            )

    # ── परिचय ────────────────────────────────────────────────────────────────
    elif mode == "👤 परिचय":
        st.markdown("## 👤 GreenWatch AI के बारे में")
        st.markdown("""
GreenWatch AI एक AI-संचालित कृषि मंच है जो एकल सुलभ टूल में फसल रोग पहचान,
मौसम पूर्वानुमान, फसल अनुशंसा और उर्वरक अनुशंसा को जोड़ता है।

**निर्मित: ** YOLOv8 · MobileNetV2 · Custom CNN · Naive Bayes · SVM · Streamlit · OpenWeatherMap API

**मॉडल पाइपलाइन:** कच्ची छवि → YOLOv8n-cls → 36-वर्ग softmax → रोग + विश्वास स्तर + कार्रवाई योग्य मार्गदर्शन
        """)
        tab1, tab2, tab3, tab4 = st.tabs(["🦠 रोग डिटेक्टर", "🌱 फसल सुझाव", "🧪 उर्वरक सुझाव", "🌥️ मौसम पूर्वानुमान"])
        with tab1:
            st.markdown("कस्टम CNN, फिर MobileNetV2 ट्रांसफर लर्निंग, फिर अंतिम तैनाती के लिए YOLOv8n-cls प्रशिक्षित किया। YOLO ने 10 फसल प्रकारों के 36 रोग वर्गों में सर्वश्रेष्ठ सटीकता प्राप्त की।")
        with tab2:
            st.markdown("मिट्टी NPK, तापमान, आर्द्रता, pH और वर्षा पर प्रशिक्षित Naive Bayes वर्गीकरणकर्ता सर्वोत्तम फसल की सिफारिश करता है।")
        with tab3:
            st.markdown("फसल प्रकार, मिट्टी प्रकार, मिट्टी NPK, तापमान, आर्द्रता और नमी पर प्रशिक्षित SVM मॉडल इष्टतम उर्वरक की सिफारिश करता है।")
        with tab4:
            st.markdown("OpenWeatherMap API का उपयोग करके लाइव 5-दिन मौसम पूर्वानुमान और तापमान, आर्द्रता, वर्षा और हवा पर आधारित कृषि सावधानियां।")

    # ── रोग पहचान ────────────────────────────────────────────────────────────
    elif mode == "🦠 रोग पहचान":
        st.markdown("## 🦠 फसल रोग पहचान")
        st.markdown("रोगों को तुरंत पहचानने के लिए अपनी फसल की पत्ती की स्पष्ट छवि अपलोड करें।")

        uploaded_image = st.file_uploader(
            "एक छवि चुनें (JPG, JPEG, PNG)",
            type=["jpg", "jpeg", "png"],
            help="सर्वोत्तम परिणाम के लिए: अच्छी रोशनी, प्रभावित पत्ती के करीब से, स्पष्ट फोकस में"
        )

        active_image = uploaded_image
        if not active_image and 'sample_image_hi' in st.session_state:
            active_image = st.session_state['sample_image_hi']
            st.info(f"नमूना उपयोग हो रहा है: **{st.session_state.get('sample_name_hi', '')}**")

        if active_image is not None:
            col1, col2 = st.columns([1, 1])
            with col1:
                st.markdown("**📷 अपलोड की गई छवि**")
                if isinstance(active_image, str):
                    st.image(active_image, use_column_width=True)
                else:
                    st.image(active_image, use_column_width=True)
            with col2:
                st.markdown("**🔬 विश्लेषण**")
                if st.button("🔍 रोग पहचानें", use_container_width=True):
                    with st.spinner("फसल छवि का विश्लेषण किया जा रहा है..."):
                        result, confidence, all_probs = model_prediction(active_image)

                    render_confidence_bar(confidence)

                    if confidence < 0.4:
                        st.warning("⚠️ कम विश्वास स्तर — स्पष्ट, करीबी, अच्छी रोशनी वाली छवि आज़माएं।")

                    render_disease_card(result, confidence)

                    st.markdown("**📊 शीर्ष 3 पूर्वानुमान**")
                    top3_idx = np.argsort(all_probs)[::-1][:3]
                    for idx in top3_idx:
                        name = CLASSES[idx].replace("_", " ")
                        pct  = int(all_probs[idx] * 100)
                        bar_color = "#66bb6a" if idx == top3_idx[0] else "#4a6741"
                        st.markdown(f"""
                        <div style="margin:4px 0;">
                            <span style="color:#c8e6c9;font-size:0.85rem;">{name}</span>
                            <div style="background:#1a3a1a;border-radius:6px;height:12px;margin-top:2px;">
                                <div style="background:{bar_color};border-radius:6px;height:12px;width:{pct}%;"></div>
                            </div>
                            <span style="color:#81c784;font-size:0.8rem;">{pct}%</span>
                        </div>
                        """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background:#1a3a1a;border:2px dashed #4caf50;border-radius:12px;
                        padding:40px;text-align:center;">
                <h3 style="color:#66bb6a;">📷 शुरू करने के लिए फसल की छवि अपलोड करें</h3>
                <p style="color:#81c784;">या साइडबार से नमूना छवि चुनें →</p>
            </div>
            """, unsafe_allow_html=True)

    # ── 5-दिन पूर्वानुमान ────────────────────────────────────────────────────
    elif mode == "🌥️ 5-दिन पूर्वानुमान":
        st.markdown("## 🌥️ मौसम पूर्वानुमान और फसल सावधानियां")

        city_input = st.text_input("🌍 अपने शहर का नाम दर्ज करें", value="नई दिल्ली", key="city_input_hindi")

        if st.button("मौसम पूर्वानुमान प्राप्त करें", key="weather_btn_hindi"):
            with st.spinner("मौसम डेटा प्राप्त किया जा रहा है..."):
                g = geocoder.arcgis(city_input)
            if g.ok:
                lat, lng = g.lat, g.lng
                st.success(f"📍 **{city_input}** (अक्षांश: {round(lat,4)}, देशांतर: {round(lng,4)})")

                forecast_df = get_weather_forecast(lat, lng)
                if not forecast_df.empty:
                    plot_weather_forecast(forecast_df)

                    forecast_df['date'] = forecast_df['datetime'].dt.date
                    avgs      = forecast_df.groupby('date').mean().head(5)
                    avg_temp  = avgs['temperature'].mean()
                    avg_hum   = avgs['humidity'].mean()
                    avg_prec  = avgs['precipitation'].mean()
                    avg_wind  = avgs['windspeed'].mean()

                    st.markdown("### 📊 5-दिन औसत")
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("🌡️ औसत तापमान", f"{avg_temp:.1f}°C")
                    c2.metric("💧 औसत आर्द्रता", f"{avg_hum:.1f}%")
                    c3.metric("🌧️ औसत वर्षा",   f"{avg_prec:.2f} mm")
                    c4.metric("💨 औसत हवा",      f"{avg_wind:.1f} m/s")

                    st.markdown("### ⚠️ फसल सावधानियां")
                    precautions = []
                    if avg_temp > 35:
                        precautions.append("🌡️ **उच्च तापमान** — गर्मी के तनाव से बचाने के लिए फसलों की सिंचाई करें।")
                    elif avg_temp < 15:
                        precautions.append("❄️ **कम तापमान** — ठंड से नुकसान से बचाने के लिए संवेदनशील फसलों को ढकें।")
                    if avg_hum > 80:
                        precautions.append("💧 **उच्च आर्द्रता** — फफूंद रोगों (फफूंदी, रस्ट) के लिए निगरानी करें।")
                    if avg_prec > 5:
                        precautions.append("🌧️ **भारी वर्षा** — जलभराव और जड़ सड़न से बचाने के लिए जल निकासी सुनिश्चित करें।")
                    elif avg_prec == 0:
                        precautions.append("☀️ **कोई वर्षा नहीं** — मिट्टी की नमी बनाए रखने के लिए सिंचाई करें।")
                    if avg_wind > 15:
                        precautions.append("💨 **तेज हवाएं** — ढीले पौधों को सुरक्षित करें और भौतिक क्षति की जांच करें।")
                    if not precautions:
                        precautions.append("✅ **स्थितियां अनुकूल लगती हैं** — नियमित निगरानी जारी रखें।")

                    for p in precautions:
                        st.markdown(f"- {p}")
            else:
                st.error("शहर नहीं मिला। कृपया वर्तनी जांचें और पुनः प्रयास करें।")

    # ── फसल सुझाव ────────────────────────────────────────────────────────────
    elif mode == "🌱 फसल सुझाव":
        st.markdown("## 🌱 फसल अनुशंसा")
        st.markdown("अपनी मिट्टी और जलवायु विवरण दर्ज करें ताकि आपकी परिस्थितियों के लिए सबसे अच्छी फसल खोजी जा सके।")

        with st.form("crop_form_hi"):
            col1, col2 = st.columns(2)
            with col1:
                N_input        = st.number_input("मिट्टी में नाइट्रोजन (%)", min_value=0, max_value=200, value=50)
                P_input        = st.number_input("मिट्टी में फॉस्फोरस (%)", min_value=0, max_value=200, value=50)
                K_input        = st.number_input("मिट्टी में पोटेशियम (%)", min_value=0, max_value=200, value=50)
                temp_input     = st.number_input("तापमान (°C)", min_value=-10.0, max_value=60.0, value=25.0)
            with col2:
                humid_input    = st.number_input("आर्द्रता (%)", min_value=0.0, max_value=100.0, value=60.0)
                ph_input       = st.number_input("मिट्टी का pH", min_value=0.0, max_value=14.0, value=6.5)
                rainfall_input = st.number_input("वार्षिक वर्षा (mm)", min_value=0.0, max_value=5000.0, value=200.0)
            submit = st.form_submit_button("🌾 फसल की सिफारिश करें", use_container_width=True)

        if submit:
            try:
                model      = load_crop_model()
                features   = np.array([[N_input, P_input, K_input, temp_input, humid_input, ph_input, rainfall_input]])
                prediction = model.predict(features)
                st.markdown(f"""
                <div style="background:linear-gradient(135deg,#1b5e20,#2e7d32);border-left:5px solid #66bb6a;
                            border-radius:12px;padding:20px 24px;margin:12px 0;text-align:center;">
                    <h2 style="color:#a5d6a7;">🌾 अनुशंसित फसल</h2>
                    <h1 style="color:#66bb6a;font-size:2.5rem;">{prediction[0].upper()}</h1>
                    <p style="color:#c8e6c9;">आपकी मिट्टी और जलवायु परिस्थितियों के लिए सबसे उपयुक्त।</p>
                </div>
                """, unsafe_allow_html=True)
                st.markdown("**समर्थित फसलें:** सेब, केला, चावल, अनार, अरहर, पपीता, संतरा, खरबूज, मूंग, मोठ, आम, मक्का, मसूर, राजमा, जूट, अंगूर, कपास, कॉफी, नारियल, चना, उड़द, तरबूज")
            except Exception as e:
                st.error(f"पूर्वानुमान त्रुटि: {e}")

    # ── उर्वरक सुझाव ─────────────────────────────────────────────────────────
    elif mode == "🧪 उर्वरक सुझाव":
        st.markdown("## 🧪 उर्वरक अनुशंसा")
        st.markdown("अपने निकटतम मृदा परीक्षण केंद्र पर मिट्टी की जांच करवाएं और नीचे विवरण भरें।")

        crop_mapping = {
            'चावल':0,'गेहूं':1,'तंबाकू':2,'गन्ना':3,'दालें':4,'अनार':5,
            'धान':6,'तेल बीज':7,'बाजरा':8,'मक्का':9,'मूंगफली':10,
            'कपास':11,'कॉफी':12,'तरबूज':13,'जौ':14,'राजमा':15,'संतरा':16
        }
        crop_mapping_en = {
            'चावल':'rice','गेहूं':'Wheat','तंबाकू':'Tobacco','गन्ना':'Sugarcane',
            'दालें':'Pulses','अनार':'pomegranate','धान':'Paddy','तेल बीज':'Oil seeds',
            'बाजरा':'Millets','मक्का':'Maize','मूंगफली':'Ground Nuts','कपास':'Cotton',
            'कॉफी':'coffee','तरबूज':'watermelon','जौ':'Barley','राजमा':'kidneybeans','संतरा':'orange'
        }
        crop_mapping_idx = {
            'rice':0,'Wheat':1,'Tobacco':2,'Sugarcane':3,'Pulses':4,'pomegranate':5,
            'Paddy':6,'Oil seeds':7,'Millets':8,'Maize':9,'Ground Nuts':10,
            'Cotton':11,'coffee':12,'watermelon':13,'Barley':14,'kidneybeans':15,'orange':16
        }
        soil_mapping    = {'चिकनी मिट्टी':0,'दोमट':1,'लाल':2,'काली':3,'बलुई':4}
        soil_mapping_en = {'चिकनी मिट्टी':'Clayey','दोमट':'Loamy','लाल':'Red','काली':'Black','बलुई':'Sandy'}
        soil_mapping_idx = {'Clayey':0,'Loamy':1,'Red':2,'Black':3,'Sandy':4}

        with st.form("fert_form_hi"):
            col1, col2 = st.columns(2)
            with col1:
                temp_input  = st.number_input("तापमान (°C)",         min_value=0, max_value=60,  value=25)
                humid_input = st.number_input("आर्द्रता (%)",         min_value=0, max_value=100, value=60)
                moist_input = st.number_input("मिट्टी की नमी (%)",   min_value=0, max_value=100, value=40)
                soil_input  = st.selectbox("मिट्टी का प्रकार", list(soil_mapping.keys()))
            with col2:
                crop_input  = st.selectbox("फसल का प्रकार", list(crop_mapping_en.keys()))
                N_input     = st.number_input("नाइट्रोजन (N)",        min_value=0, max_value=200, value=40)
                P_input     = st.number_input("फॉस्फोरस (P)",         min_value=0, max_value=200, value=40)
                K_input     = st.number_input("पोटेशियम (K)",         min_value=0, max_value=200, value=40)
            submit = st.form_submit_button("🧪 उर्वरक की सिफारिश करें", use_container_width=True)

        if submit:
            try:
                model      = load_fertilizer_model()
                soil_en    = soil_mapping_en[soil_input]
                crop_en    = crop_mapping_en[crop_input]
                data       = np.array([temp_input, humid_input, moist_input,
                                       soil_mapping_idx[soil_en], crop_mapping_idx[crop_en],
                                       N_input, K_input, P_input])
                prediction = model.predict([data])
                st.markdown(f"""
                <div style="background:linear-gradient(135deg,#1b5e20,#2e7d32);border-left:5px solid #66bb6a;
                            border-radius:12px;padding:20px 24px;margin:12px 0;text-align:center;">
                    <h2 style="color:#a5d6a7;">🧪 अनुशंसित उर्वरक</h2>
                    <h1 style="color:#66bb6a;font-size:2.5rem;">{prediction[0].upper()}</h1>
                    <p style="color:#c8e6c9;">आपकी फसल और मिट्टी की परिस्थितियों के लिए इष्टतम।</p>
                </div>
                """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"पूर्वानुमान त्रुटि: {e}")

    # ── टीम ──────────────────────────────────────────────────────────────────
    elif mode == "👥 टीम":
        st.markdown("## 👥 टीम से मिलें — AI-Craft")
        st.markdown("GreenWatch AI को JNU के CSE छात्रों की एक टीम ने एक पूर्ण-स्टैक AI कृषि परियोजना के रूप में बनाया।")
        st.markdown("<hr>", unsafe_allow_html=True)

        team_members = [
            {"name": "Krishna Vamsy K",      "roll": "23/11/EC/002", "role": "ML लीड और फुल स्टैक",   "image_path": "team/krishna.jpg"},
            {"name": "M. Pradeep",            "roll": "23/11/EC/063", "role": "ML इंजीनियर",           "image_path": "team/pradeep.jpg"},
            {"name": "A. Sampath Dev",        "roll": "23/11/EC/029", "role": "बैकएंड और डेटा",        "image_path": "team/sampath.jpg"},
            {"name": "Vignesh Thangabalan B", "roll": "23/11/EC/020", "role": "फ्रंटएंड और UI",        "image_path": "team/vignesh.jpg"},
            {"name": "M. Jai Ram Chandra",    "roll": "23/11/EC/071", "role": "मॉडल प्रशिक्षण और QA", "image_path": "team/jairam.jpg"},
        ]

        for member in team_members:
            col1, col2 = st.columns([1, 4])
            with col1:
                img_b64 = get_base64_image(member["image_path"])
                if img_b64:
                    st.markdown(f"""
                    <div style="border:2px solid #4caf50;border-radius:12px;padding:5px;
                                display:inline-block;text-align:center;max-width:120px;">
                        <img src="data:image/jpg;base64,{img_b64}"
                             style="max-width:100%;border-radius:8px;">
                    </div>""", unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div style="background:#1e3a1e;border:2px solid #4caf50;border-radius:12px;
                                width:100px;height:100px;display:flex;align-items:center;
                                justify-content:center;font-size:2rem;">👤</div>
                    """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"**{member['name']}**")
                st.markdown(f"<span style='color:#81c784;'>{member['role']}</span>", unsafe_allow_html=True)
                st.markdown(f"<span style='color:#a5d6a7;font-size:0.85rem;'>रोल नंबर: {member['roll']}</span>", unsafe_allow_html=True)
            st.markdown("<hr style='border-color:#1e3a1e;margin:8px 0;'>", unsafe_allow_html=True)

