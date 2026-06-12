import streamlit as st
import geocoder
import requests
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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

# ─── Page config (must be first Streamlit call) ───────────────────────────────
st.set_page_config(
    page_title="GreenWatch AI",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)



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
    "🍅 Tomato Early Blight": "samples/tomato_early_blight.jpg",
    "🌽 Corn Common Rust":    "samples/corn_common_rust.jpg",
    "🌾 Wheat Brown Rust":    "samples/wheat_brown_rust.jpg",
    "🥔 Potato Late Blight":  "samples/potato_late_blight.jpg",
}

# ─── Disease info database ────────────────────────────────────────────────────
DISEASE_INFO = {
    'Apple__Apple_scab': {
        "severity": "Moderate", "spread": "Fungal (Venturia inaequalis)",
        "symptoms": "Olive-green to brown velvety spots on leaves and fruits; distorted leaves; dark raised lesions on fruits that crack as they enlarge.",
        "prevention": "Plant resistant varieties · Remove fallen leaves · Prune for air circulation · Apply preventive fungicides during early spring.",
        "cure": "Fungicides: myclobutanil, captan, or mancozeb at first sign of infection. Remove infected material and prune for airflow."
    },
    'Apple_Black_rot': {
        "severity": "High", "spread": "Fungal (Botryosphaeria obtusa)",
        "symptoms": "Dark circular sunken lesions on fruit with concentric rings; purple-bordered leaf spots; cankers on branches.",
        "prevention": "Remove fallen debris · Prune infected branches · Apply fungicides preventively.",
        "cure": "Prune cankers 6–12 inches beyond damage. Fungicides: captan or thiophanate-methyl."
    },
    'Corn_Blight': {
        "severity": "High", "spread": "Fungal (Bipolaris maydis / Exserohilum turcicum)",
        "symptoms": "Tan oval lesions (Southern) or large cigar-shaped gray lesions (Northern) on leaves; premature leaf death.",
        "prevention": "Plant resistant hybrids · Crop rotation · Remove infected debris · Ensure proper plant spacing.",
        "cure": "Fungicides: azoxystrobin or propiconazole at first appearance of symptoms."
    },
    'Corn_Common_Rust': {
        "severity": "Moderate", "spread": "Fungal (Puccinia sorghi)",
        "symptoms": "Small oval reddish-brown pustules on both sides of leaves; leaves dry out prematurely.",
        "prevention": "Plant resistant hybrids · Crop rotation · Monitor during cool humid weather.",
        "cure": "Fungicides containing strobilurins or triazoles. Apply early for best results."
    },
    'Corn_Gray_Leaf_Spot': {
        "severity": "High", "spread": "Fungal (Cercospora zeae-maydis)",
        "symptoms": "Small rectangular grayish-tan lesions parallel to leaf veins that merge causing large dead areas.",
        "prevention": "Resistant hybrids · Crop rotation · Plow under residue · Proper plant spacing.",
        "cure": "Fungicides: azoxystrobin (Quadris), pyraclostrobin (Headline), propiconazole. Apply at VT to R1 growth stage."
    },
    'Cotton_bacterial_blight': {
        "severity": "High", "spread": "Bacterial (Xanthomonas citri pv. malvacearum)",
        "symptoms": "Water-soaked angular lesions on leaves turning brown/black; dark sunken spots on stems and bolls; defoliation.",
        "prevention": "Resistant varieties · Certified disease-free seeds · Crop rotation · Avoid overhead irrigation.",
        "cure": "Copper hydroxide (Kocide) or copper oxychloride foliar sprays. Regular field monitoring."
    },
    'Cotton_curl_virus': {
        "severity": "High", "spread": "Viral — transmitted by whitefly (Bemisia tabaci)",
        "symptoms": "Upward leaf curling; yellowing; stunted plants; mosaic patterns; premature leaf drop.",
        "prevention": "Control whiteflies (imidacloprid, thiamethoxam) · Plant resistant varieties · Remove infected plants.",
        "cure": "No cure once infected. Focus on whitefly vector control and removal of infected plants."
    },
    'Grape_Black_rot': {
        "severity": "High", "spread": "Fungal (Guignardia bidwellii)",
        "symptoms": "Dark circular sunken lesions on grapes; small dark lesions with yellow halos on leaves; infected fruit shrivels and turns black.",
        "prevention": "Resistant varieties · Remove infected material · Good air circulation · Fungicides before flowering.",
        "cure": "Fungicides: captan, chlorothalonil, or mancozeb. Prune infected areas. Continue spraying late in season."
    },
    'Grape_Leaf_blight(Isariopsis_Leaf_Spot)': {
        "severity": "Moderate", "spread": "Fungal (Phomopsis viticola)",
        "symptoms": "Small dark round lesions with light gray center on leaves; brown sunken shoot lesions; water-soaked spots on fruit clusters.",
        "prevention": "Resistant varieties · Prune infected vines · Adequate air circulation · Fungicides early season.",
        "cure": "Fungicides: captan, chlorothalonil, myclobutanil. Begin spraying before bud break."
    },
    'Pepper,_bell_Bacterial_spot': {
        "severity": "High", "spread": "Bacterial (Xanthomonas euvesicatoria)",
        "symptoms": "Small water-soaked angular lesions with yellow halos on leaves; dark sunken lesions on fruit; defoliation; stunted growth.",
        "prevention": "Resistant varieties · Disease-free seeds · Crop rotation · Drip irrigation · Proper spacing.",
        "cure": "Copper hydroxide or copper sulfate bactericides. Biological control with Bacillus subtilis. Prune infected parts."
    },
    'Potato_Early_blight': {
        "severity": "Moderate", "spread": "Fungal (Alternaria solani)",
        "symptoms": "Dark brown to black concentric ring spots (target-like) on older leaves; yellowing around lesions; stem lesions; tuber infection in severe cases.",
        "prevention": "Crop rotation · Resistant varieties · Remove infected debris · Avoid overhead watering.",
        "cure": "Fungicides: chlorothalonil, mancozeb, azoxystrobin. Begin at first sign or preventively in warm wet weather."
    },
    'Potato_Late_blight': {
        "severity": "Critical", "spread": "Oomycete (Phytophthora infestans)",
        "symptoms": "Water-soaked pale green to dark brown lesions on leaves; white fuzzy mold on leaf undersides; dark stem lesions; tuber rot.",
        "prevention": "Resistant varieties · Certified seed potatoes · Crop rotation · Proper spacing · Avoid overhead watering.",
        "cure": "Fungicides: chlorothalonil, mancozeb, metalaxyl (Ridomil Gold). Apply preventively. Copper sprays for organic management."
    },
    'Rice_Healthy': {"severity": "None", "spread": "N/A", "symptoms": "No disease detected.", "prevention": "Continue good practices.", "cure": "N/A"},
    'Rice_bacterial_leaf_blight': {
        "severity": "High", "spread": "Bacterial (Xanthomonas oryzae pv. oryzae)",
        "symptoms": "Water-soaked streaks on leaf margins turning yellow then brown; characteristic V-shaped lesions; leaf drying; stunted growth.",
        "prevention": "Resistant varieties · Certified seeds · Remove crop residues · Balanced fertilization (avoid excess nitrogen).",
        "cure": "Copper-based bactericides. Streptomycin/oxytetracycline sprays. Biological control: Pseudomonas fluorescens."
    },
    'Rice_brown_spot': {
        "severity": "Moderate", "spread": "Fungal (Bipolaris oryzae)",
        "symptoms": "Circular to oval brown lesions with grayish center and dark margins on leaves; grain discoloration; stunted growth.",
        "prevention": "Adequate NPK nutrition · Certified seeds · Proper water management · Crop rotation · Field sanitation.",
        "cure": "Fungicides: mancozeb, tricyclazole, propiconazole. Seed treatment with carbendazim or thiram."
    },
    'Rice_leaf_blast': {
        "severity": "Critical", "spread": "Fungal (Magnaporthe oryzae)",
        "symptoms": "Diamond-shaped spindle lesions with gray centers and brown borders; collar rot; node and panicle blast in severe cases.",
        "prevention": "Resistant varieties · Balanced nitrogen · Consistent water levels · Avoid overcrowding.",
        "cure": "Fungicides: tricyclazole, isoprothiolane (Fuji-one), edifenphos (Hinosan). Silicon fertilization strengthens cell walls."
    },
    'Sugarcan_Mosaic': {
        "severity": "Moderate", "spread": "Viral (SCMV) — transmitted by aphids",
        "symptoms": "Light and dark green mosaic patches on leaves; stunted growth; leaf distortion; reduced cane and sugar yield.",
        "prevention": "Certified virus-free seed cane · Resistant varieties · Aphid control (imidacloprid) · Remove infected plants.",
        "cure": "No chemical cure. Manage aphid vectors. Remove and destroy infected plants (rogueing)."
    },
    'Sugarcane_RedRot': {
        "severity": "Critical", "spread": "Fungal (Colletotrichum falcatum)",
        "symptoms": "Red discoloration with white patches inside split stalks; spongy stalks with foul odor; yellowing and drying leaves from top down.",
        "prevention": "Resistant varieties · Remove infected material · Certified disease-free seed canes · Adequate drainage.",
        "cure": "Fungicides: carbendazim, thiophanate-methyl. Hot water treatment: 52°C for 30 min. Biological control: Trichoderma spp."
    },
    'Sugarcane_Rust': {
        "severity": "Moderate", "spread": "Fungal (Puccinia melanocephala / P. kuehnii)",
        "symptoms": "Reddish-brown or orange pustules on leaf undersides; chlorosis around pustules; premature leaf death; reduced cane growth.",
        "prevention": "Resistant varieties · Remove infected debris · Balanced fertilization · Adequate plant spacing.",
        "cure": "Fungicides: propiconazole, tebuconazole, or mancozeb. Apply preventively or at first rust symptoms."
    },
    'Sugarcane_Yellow': {
        "severity": "Moderate", "spread": "Viral (ScYLV) — aphid-transmitted or nutritional",
        "symptoms": "Yellowing of older leaves from tips inward; stunted growth; vein yellowing; reduced sucrose content.",
        "prevention": "Resistant varieties · Aphid control · Balanced NPK nutrition · Proper irrigation · Field sanitation.",
        "cure": "No cure for viral cause. Aphid management, nutrient foliar sprays, remove infected plants."
    },
    'Tomato_Bacterial_spot': {
        "severity": "High", "spread": "Bacterial (Xanthomonas vesicatoria)",
        "symptoms": "Small water-soaked spots turning brown/black with yellow halos; leaf necrosis; small dark sunken spots on fruit; stunted growth.",
        "prevention": "Resistant varieties · Certified seeds · Crop rotation · Drip irrigation · Field sanitation · Weed control.",
        "cure": "Copper hydroxide or copper oxychloride bactericides. Biological: Bacillus subtilis. Remove infected plants (rogueing)."
    },
    'Tomato_Early_blight': {
        "severity": "Moderate", "spread": "Fungal (Alternaria solani)",
        "symptoms": "Dark brown/black concentric ring spots on older leaves; yellowing around lesions; defoliation; dark lesions on stems and fruit near soil.",
        "prevention": "Resistant varieties · Proper spacing · Crop rotation · Remove infected debris · Drip irrigation.",
        "cure": "Fungicides: chlorothalonil (Daconil), copper-based, mancozeb, azoxystrobin. Prune infected lower leaves."
    },
    'Tomato_Late_blight': {
        "severity": "Critical", "spread": "Oomycete (Phytophthora infestans)",
        "symptoms": "Water-soaked dark spots on leaves that spread rapidly; white fuzzy fungal growth in humidity; dark stem lesions; soft fruit rot.",
        "prevention": "Resistant varieties · Good plant spacing · Crop rotation · Field sanitation · Drip irrigation.",
        "cure": "Fungicides: chlorothalonil, mancozeb, copper-based, azoxystrobin. Apply frequently during wet periods. Remove infected plants immediately."
    },
    'Tomato_Septoria_leaf_spot': {
        "severity": "Moderate", "spread": "Fungal (Septoria lycopersici)",
        "symptoms": "Small circular water-soaked spots with dark brown/gray centers and yellow halos on lower leaves; leaf yellowing; defoliation spreading upward.",
        "prevention": "Resistant varieties · Adequate spacing · Crop rotation · Field sanitation · Drip irrigation · Mulching.",
        "cure": "Fungicides: chlorothalonil, mancozeb, copper-based, azoxystrobin, tebuconazole. Prune infected lower leaves. Biological: Bacillus subtilis."
    },
    'Wheat_Brown_rust': {
        "severity": "High", "spread": "Fungal (Puccinia triticina)",
        "symptoms": "Small circular orange-brown pustules on upper leaf surface; surrounding leaf yellowing; premature leaf senescence; shriveled grain.",
        "prevention": "Resistant varieties · Crop rotation · Timely planting · Balanced nutrition.",
        "cure": "Triazole fungicides: tebuconazole, propiconazole. Strobilurin: azoxystrobin. Apply preventively or at first sign."
    },
    'Wheat_Loose_Smut': {
        "severity": "High", "spread": "Fungal (Ustilago tritici) — seed-borne",
        "symptoms": "Black powdery spore masses replacing grain heads; no grain formation; plants appear to mature early.",
        "prevention": "Certified disease-free seeds · Resistant varieties · Seed treatment with systemic fungicides.",
        "cure": "Seed treatment: carboxin (Vitavax), tebuconazole, or triadimenol. Hot water treatment: 52–54°C for 10–15 min."
    },
    'Wheat_Yellow_rust': {
        "severity": "High", "spread": "Fungal (Puccinia striiformis)",
        "symptoms": "Long narrow yellow pustule stripes along leaf veins; leaf drying; reduced grain weight; pustules may appear on stems and glumes.",
        "prevention": "Resistant varieties · Early sowing · Regular scouting in cool wet periods · Crop rotation.",
        "cure": "Triazoles: tebuconazole, propiconazole. Strobilurins: azoxystrobin. Combination products for enhanced control."
    },
}

SEVERITY_COLORS = {
    "None":     ("#1b5e20", "#66bb6a", "✅"),
    "Moderate": ("#e65100", "#ffb74d", "⚠️"),
    "High":     ("#b71c1c", "#ef9a9a", "🔴"),
    "Critical": ("#4a148c", "#ce93d8", "🚨"),
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
    pct = int(confidence * 100)
    color = "#66bb6a" if confidence >= 0.7 else "#ffa726" if confidence >= 0.4 else "#ef5350"
    st.markdown(f"""
    <div style="margin-bottom:4px;">
        <span style="color:#a5d6a7;font-size:0.9rem;">Confidence: <b style="color:{color};">{pct}%</b></span>
    </div>
    <div class="conf-bar-bg">
        <div class="conf-bar-fill" style="width:{pct}%;background:linear-gradient(90deg,{color},{color}aa);"></div>
    </div>
    """, unsafe_allow_html=True)

def render_disease_card(result: str, confidence: float):
    info = DISEASE_INFO.get(result, {})
    severity = info.get("severity", "Unknown")
    bg, accent, icon = SEVERITY_COLORS.get(severity, ("#1a3a1a", "#4caf50", "ℹ️"))
    is_healthy = result in HEALTHY_CLASSES

    if is_healthy:
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,#1b5e20,#2e7d32);border-left:5px solid #66bb6a;
                    border-radius:12px;padding:20px 24px;margin:12px 0;">
            <h2 style="color:#a5d6a7;margin:0;">✅ Your crop appears healthy!</h2>
            <p style="color:#c8e6c9;margin-top:8px;">
                No disease detected. Keep monitoring your crops and use our 
                <b>Fertilizer Recommender</b> to optimise yield further.
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
                     font-size:0.8rem;font-weight:600;">Severity: {severity}</span>
        <span style="background:#1a3a1a;color:#81c784;padding:3px 10px;border-radius:12px;
                     font-size:0.8rem;margin-left:8px;">{info.get('spread','')}</span>
    </div>
    """, unsafe_allow_html=True)

    if info:
        tab1, tab2, tab3 = st.tabs(["🔍 Symptoms", "🛡️ Prevention", "💊 Cure"])
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
    st.error("Failed to fetch weather data.")
    return pd.DataFrame()

def plot_weather_forecast(forecast_df):
    fig, axes = plt.subplots(4, 1, figsize=(10, 12))
    fig.patch.set_facecolor('#0f1a0f')
    configs = [
        ('temperature',   'tab:red',    'Temperature (°C)'),
        ('humidity',      'tab:blue',   'Humidity (%)'),
        ('precipitation', 'tab:green',  'Precipitation (mm)'),
        ('windspeed',     'tab:orange', 'Wind Speed (m/s)'),
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
        ax.set_title(f"{label} Forecast", color='#81c784', fontsize=11, fontweight='bold', pad=6)
    plt.tight_layout(pad=2.0)
    st.pyplot(fig)

# ─── Main app ─────────────────────────────────────────────────────────────────
def English():

    with st.sidebar:
        mode = option_menu(
            menu_title=None,
            options=["🏡 Home", "👤 About", "🦠 Disease Recognition",
                     "🌥️ 5-Day Forecast", "🌱 Crop Recommender",
                     "🧪 Fertilizer Recommender", "👥 Team"],
            styles={
                "container":    {"background-color": "transparent"},
                "icon":         {"color": "#81c784", "font-size": "16px"},
                "nav-link":     {"color": "#c8e6c9", "font-size": "14px", "--hover-color": "#2d5a2d"},
                "nav-link-selected": {"background-color": "#2e7d32", "color": "white"},
            }
        )
        st.markdown("<hr style='border-color:#2d5a2d;margin:20px 0;'>", unsafe_allow_html=True)
        st.markdown("""
        <div style='text-align:center;padding:10px;'>
            <h2 style='color:#66bb6a;margin:0;'>🌾 GreenWatch</h2>
            <p style='color:#81c784;font-size:0.8rem;margin:4px 0;'>AI Crop Health Advisor</p>
            <p style='color:#4caf50;font-size:0.75rem;'>by AI-Craft</p>
        </div>
        """, unsafe_allow_html=True)

        # ── Sidebar sample images (visible on Disease Recognition page) ──
        if mode == "🦠 Disease Recognition":
            st.markdown("<hr style='border-color:#2d5a2d;'>", unsafe_allow_html=True)
            st.markdown("<p style='color:#81c784;font-weight:600;'>🌿 Try a Sample Image</p>", unsafe_allow_html=True)
            sample_choice = st.selectbox("Choose sample:", list(SAMPLE_IMAGES.keys()), label_visibility="collapsed")
            if st.button("Load Sample"):
                sample_path = SAMPLE_IMAGES[sample_choice]
                if os.path.exists(sample_path):
                    st.session_state['sample_image'] = sample_path
                    st.session_state['sample_name'] = sample_choice
                else:
                    st.warning("Sample image not found in repo.")

    # ── HOME ──────────────────────────────────────────────────────────────────
    if mode == "🏡 Home":
        st.markdown("<h1 style='text-align:center;'>🌾 GreenWatch: AI Crop Health Advisor</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align:center;color:#81c784;font-size:1.1rem;'>Protect Your Crops. Secure Your Yield.</p>", unsafe_allow_html=True)
        st.markdown("<hr>", unsafe_allow_html=True)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown('<div class="metric-card"><h3>36</h3><p>Disease Classes</p></div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="metric-card"><h3>10+</h3><p>Crop Types</p></div>', unsafe_allow_html=True)
        with col3:
            st.markdown('<div class="metric-card"><h3>98%+</h3><p>Model Accuracy</p></div>', unsafe_allow_html=True)
        with col4:
            st.markdown('<div class="metric-card"><h3>2</h3><p>Languages</p></div>', unsafe_allow_html=True)

        st.markdown("### 🌱 How It Works")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**📤 1. Upload Image**\n\nTake a photo of your crop showing signs of illness and upload it.")
        with c2:
            st.markdown("**🧠 2. AI Analysis**\n\nOur YOLOv8-based model analyses the image and identifies the disease.")
        with c3:
            st.markdown("**📋 3. Get Guidance**\n\nReceive an instant diagnosis with symptoms, prevention, and cure.")

        st.markdown("### 🌾 Supported Crops")
        crops = ["🍎 Apple", "🌽 Corn", "🧶 Cotton", "🍇 Grape",
                 "🫑 Pepper", "🥔 Potato", "🌾 Rice", "🎋 Sugarcane",
                 "🍅 Tomato", "🌾 Wheat"]
        cols = st.columns(5)
        for i, crop in enumerate(crops):
            cols[i % 5].markdown(f"<div style='background:#1e3a1e;border-radius:8px;padding:8px;text-align:center;margin:4px;color:#c8e6c9;'>{crop}</div>", unsafe_allow_html=True)

    # ── ABOUT ────────────────────────────────────────────────────────────────
    elif mode == "👤 About":
        st.markdown("## 👤 About GreenWatch AI")
        st.markdown("""
GreenWatch AI is an AI-powered agricultural platform combining crop disease detection, 
weather forecasting, crop recommendation, and fertilizer recommendation into a single 
accessible tool for farmers.

**Built with:** YOLOv8 · MobileNetV2 · Custom CNN · Naive Bayes · SVM · Streamlit · OpenWeatherMap API

**Model pipeline:** Raw image → YOLOv8n-cls → 36-class softmax → Disease + confidence + actionable guidance
        """)
        tab1, tab2, tab3, tab4 = st.tabs(["🦠 Disease Detector", "🌱 Crop Recommender", "🧪 Fertilizer Recommender", "🌥️ Weather Forecast"])
        with tab1:
            st.markdown("Trained a custom CNN, then MobileNetV2 transfer learning, then YOLOv8n-cls for final deployment. YOLO achieved the best accuracy across 36 disease classes covering 10 crop types.")
        with tab2:
            st.markdown("Naive Bayes classifier trained on soil NPK, temperature, humidity, pH, and rainfall to recommend the best crop to plant.")
        with tab3:
            st.markdown("SVM model trained on crop type, soil type, soil NPK, temperature, humidity, and moisture to recommend the optimal fertilizer.")
        with tab4:
            st.markdown("Live 5-day weather forecast using OpenWeatherMap API with intelligent farming precautions based on temperature, humidity, precipitation, and wind speed.")

    # ── DISEASE RECOGNITION ───────────────────────────────────────────────────
    elif mode == "🦠 Disease Recognition":
        st.markdown("## 🦠 Crop Disease Recognition")
        st.markdown("Upload a clear image of your crop leaf to detect diseases instantly.")

        uploaded_image = st.file_uploader(
            "Choose an image (JPG, JPEG, PNG)",
            type=["jpg", "jpeg", "png"],
            help="For best results: good lighting, close-up of affected leaf, in-focus image"
        )

        # Load sample if selected from sidebar
        active_image = uploaded_image
        if not active_image and 'sample_image' in st.session_state:
            active_image = st.session_state['sample_image']
            st.info(f"Using sample: **{st.session_state.get('sample_name', '')}**")

        if active_image is not None:
            col1, col2 = st.columns([1, 1])
            with col1:
                st.markdown("**📷 Uploaded Image**")
                if isinstance(active_image, str):
                    st.image(active_image, use_column_width=True)
                else:
                    st.image(active_image, use_column_width=True)

            with col2:
                st.markdown("**🔬 Analysis**")
                if st.button("🔍 Predict Disease", use_container_width=True):
                    with st.spinner("Analysing crop image..."):
                        result, confidence, all_probs = model_prediction(active_image)

                    render_confidence_bar(confidence)

                    if confidence < 0.4:
                        st.warning("⚠️ Low confidence — try a clearer, closer, well-lit image.")

                    render_disease_card(result, confidence)

                    # Top-3 predictions
                    st.markdown("**📊 Top 3 Predictions**")
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
                <h3 style="color:#66bb6a;">📷 Upload a crop image to get started</h3>
                <p style="color:#81c784;">Or select a sample image from the sidebar →</p>
            </div>
            """, unsafe_allow_html=True)

    # ── 5-DAY FORECAST ────────────────────────────────────────────────────────
    elif mode == "🌥️ 5-Day Forecast":
        st.markdown("## 🌥️ Weather Forecast & Crop Precautions")

        city_input = st.text_input("🌍 Enter your city name", value="New Delhi", key="city_input_english")

        if st.button("Get Weather Forecast", key="weather_btn_english"):
            with st.spinner("Fetching weather data..."):
                g = geocoder.arcgis(city_input)
            if g.ok:
                lat, lng = g.lat, g.lng
                st.success(f"📍 **{city_input}** (Lat: {round(lat,4)}, Lon: {round(lng,4)})")

                forecast_df = get_weather_forecast(lat, lng)
                if not forecast_df.empty:
                    plot_weather_forecast(forecast_df)

                    forecast_df['date'] = forecast_df['datetime'].dt.date
                    avgs = forecast_df.groupby('date').mean().head(5)
                    avg_temp  = avgs['temperature'].mean()
                    avg_hum   = avgs['humidity'].mean()
                    avg_prec  = avgs['precipitation'].mean()
                    avg_wind  = avgs['windspeed'].mean()

                    st.markdown("### 📊 5-Day Averages")
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("🌡️ Avg Temp",  f"{avg_temp:.1f}°C")
                    c2.metric("💧 Avg Humidity", f"{avg_hum:.1f}%")
                    c3.metric("🌧️ Avg Precip",  f"{avg_prec:.2f} mm")
                    c4.metric("💨 Avg Wind",    f"{avg_wind:.1f} m/s")

                    st.markdown("### ⚠️ Crop Precautions")
                    precautions = []
                    if avg_temp > 35:   precautions.append("🌡️ **High temperature** — irrigate crops to prevent heat stress.")
                    elif avg_temp < 15: precautions.append("❄️ **Low temperature** — cover sensitive crops from cold damage.")
                    if avg_hum > 80:    precautions.append("💧 **High humidity** — monitor for fungal diseases (mildew, rust).")
                    if avg_prec > 5:    precautions.append("🌧️ **Heavy rainfall** — ensure drainage to avoid waterlogging and root rot.")
                    elif avg_prec == 0: precautions.append("☀️ **No rainfall** — irrigate to maintain soil moisture.")
                    if avg_wind > 15:   precautions.append("💨 **High winds** — secure loose plants and check for physical damage.")
                    if not precautions: precautions.append("✅ **Conditions look favourable** — continue regular monitoring.")

                    for p in precautions:
                        st.markdown(f"- {p}")
            else:
                st.error("City not found. Please check spelling and try again.")

    # ── CROP RECOMMENDER ──────────────────────────────────────────────────────
    elif mode == "🌱 Crop Recommender":
        st.markdown("## 🌱 Crop Recommender")
        st.markdown("Enter your soil and climate details to find the best crop for your conditions.")

        with st.form("crop_form"):
            col1, col2 = st.columns(2)
            with col1:
                N_input       = st.number_input("Nitrogen content in soil (%)", min_value=0, max_value=200, value=50)
                P_input       = st.number_input("Phosphorous content in soil (%)", min_value=0, max_value=200, value=50)
                K_input       = st.number_input("Potassium content in soil (%)", min_value=0, max_value=200, value=50)
                temp_input    = st.number_input("Temperature (°C)", min_value=-10.0, max_value=60.0, value=25.0)
            with col2:
                humid_input   = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=60.0)
                ph_input      = st.number_input("Soil pH", min_value=0.0, max_value=14.0, value=6.5)
                rainfall_input = st.number_input("Annual Rainfall (mm)", min_value=0.0, max_value=5000.0, value=200.0)
            submit = st.form_submit_button("🌾 Recommend Crop", use_container_width=True)

        if submit:
            try:
                model = load_crop_model()
                features = np.array([[N_input, P_input, K_input, temp_input, humid_input, ph_input, rainfall_input]])
                prediction = model.predict(features)
                st.markdown(f"""
                <div style="background:linear-gradient(135deg,#1b5e20,#2e7d32);border-left:5px solid #66bb6a;
                            border-radius:12px;padding:20px 24px;margin:12px 0;text-align:center;">
                    <h2 style="color:#a5d6a7;">🌾 Recommended Crop</h2>
                    <h1 style="color:#66bb6a;font-size:2.5rem;">{prediction[0].upper()}</h1>
                    <p style="color:#c8e6c9;">Best suited for your soil and climate conditions.</p>
                </div>
                """, unsafe_allow_html=True)
                st.markdown("**Supported crops:** Apple, Banana, Rice, Pomegranate, Pigeonpeas, Papaya, Orange, Muskmelon, Mungbean, Mothbeans, Mango, Maize, Lentil, Kidneybeans, Jute, Grapes, Cotton, Coffee, Coconut, Chickpea, Blackgram, Watermelon")
            except Exception as e:
                st.error(f"Prediction error: {e}")

    # ── FERTILIZER RECOMMENDER ────────────────────────────────────────────────
    elif mode == "🧪 Fertilizer Recommender":
        st.markdown("## 🧪 Fertilizer Recommender")
        st.markdown("Test your soil at your nearest soil testing centre and fill in the details below.")

        crop_mapping = {
            'rice':0,'Wheat':1,'Tobacco':2,'Sugarcane':3,'Pulses':4,'pomegranate':5,
            'Paddy':6,'Oil seeds':7,'Millets':8,'Maize':9,'Ground Nuts':10,
            'Cotton':11,'coffee':12,'watermelon':13,'Barley':14,'kidneybeans':15,'orange':16
        }
        soil_mapping = {'Clayey':0,'Loamy':1,'Red':2,'Black':3,'Sandy':4}

        with st.form("fert_form"):
            col1, col2 = st.columns(2)
            with col1:
                temp_input  = st.number_input("Temperature (°C)", min_value=0, max_value=60, value=25)
                humid_input = st.number_input("Humidity (%)", min_value=0, max_value=100, value=60)
                moist_input = st.number_input("Soil Moisture (%)", min_value=0, max_value=100, value=40)
                soil_input  = st.selectbox("Soil Type", list(soil_mapping.keys()))
            with col2:
                crop_input  = st.selectbox("Crop Type", list(crop_mapping.keys()))
                N_input     = st.number_input("Nitrogen (N)", min_value=0, max_value=200, value=40)
                P_input     = st.number_input("Phosphorous (P)", min_value=0, max_value=200, value=40)
                K_input     = st.number_input("Potassium (K)", min_value=0, max_value=200, value=40)
            submit = st.form_submit_button("🧪 Recommend Fertilizer", use_container_width=True)

        if submit:
            try:
                model = load_fertilizer_model()
                data = np.array([temp_input, humid_input, moist_input,
                                 soil_mapping[soil_input], crop_mapping[crop_input],
                                 N_input, K_input, P_input])
                prediction = model.predict([data])
                st.markdown(f"""
                <div style="background:linear-gradient(135deg,#1b5e20,#2e7d32);border-left:5px solid #66bb6a;
                            border-radius:12px;padding:20px 24px;margin:12px 0;text-align:center;">
                    <h2 style="color:#a5d6a7;">🧪 Recommended Fertilizer</h2>
                    <h1 style="color:#66bb6a;font-size:2.5rem;">{prediction[0].upper()}</h1>
                    <p style="color:#c8e6c9;">Optimal for your crop and soil conditions.</p>
                </div>
                """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Prediction error: {e}")

    # ── TEAM ──────────────────────────────────────────────────────────────────
    elif mode == "👥 Team":
        st.markdown("## 👥 Meet the Team — AI-Craft")
        st.markdown("GreenWatch AI was built by a team of CSE students from JNU as a full-stack AI agriculture project.")
        st.markdown("<hr>", unsafe_allow_html=True)

        team_members = [
            {"name": "Krishna Vamsy K",       "roll": "23/11/EC/002", "role": "ML Lead & Full Stack",   "image_path": "team/krishna.jpg"},
            {"name": "M. Pradeep",             "roll": "23/11/EC/063", "role": "ML Engineer",            "image_path": "team/pradeep.jpg"},
            {"name": "A. Sampath Dev",         "roll": "23/11/EC/029", "role": "Backend & Data",         "image_path": "team/sampath.jpg"},
            {"name": "Vignesh Thangabalan B",  "roll": "23/11/EC/020", "role": "Frontend & UI",          "image_path": "team/vignesh.jpg"},
            {"name": "M. Jai Ram Chandra",     "roll": "23/11/EC/071", "role": "Model Training & QA",    "image_path": "team/jairam.jpg"},
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
                    st.markdown(f"""
                    <div style="background:#1e3a1e;border:2px solid #4caf50;border-radius:12px;
                                width:100px;height:100px;display:flex;align-items:center;
                                justify-content:center;font-size:2rem;">👤</div>
                    """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"**{member['name']}**")
                st.markdown(f"<span style='color:#81c784;'>{member['role']}</span>", unsafe_allow_html=True)
                st.markdown(f"<span style='color:#a5d6a7;font-size:0.85rem;'>Roll: {member['roll']}</span>", unsafe_allow_html=True)
            st.markdown("<hr style='border-color:#1e3a1e;margin:8px 0;'>", unsafe_allow_html=True)