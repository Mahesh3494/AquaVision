import streamlit as st
import numpy as np
from PIL import Image
import onnxruntime as ort
import plotly.graph_objects as go

# ================================
# CONFIGURATION
# ================================
SHRIMP_IMG_SIZE = 224
FISH_IMG_SIZE = 300

SHRIMP_CLASSES = ['BG', 'Healthy', 'WSSV', 'WSSV_BG']
FISH_CLASSES = [
    'Bacterial Red disease',
    'Bacterial diseases - Aeromoniasis',
    'Bacterial gill disease',
    'EUS Disease',
    'Fungal diseases Saprolegniasis',
    'Healthy Fish',
    'Parasitic diseases',
    'Viral diseases White tail disease'
]

SHRIMP_INFO = {
    'Healthy': {
        'status': 'Healthy',
        'description': 'Your shrimp appears healthy with no visible signs of disease.',
        'recommendation': 'Continue current feeding schedule. Monitor water quality parameters regularly.',
        'severity': 'None'
    },
    'BG': {
        'status': 'Black Gill Disease (BG)',
        'description': 'Black Gill disease caused by environmental stress or bacterial infection.',
        'recommendation': 'Improve water quality immediately. Check ammonia and nitrite levels. Reduce stocking density.',
        'severity': 'Moderate'
    },
    'WSSV': {
        'status': 'White Spot Syndrome Virus (WSSV)',
        'description': 'Most devastating shrimp disease globally. White spots visible on shell and body.',
        'recommendation': 'IMMEDIATE ACTION REQUIRED. Isolate affected shrimp. Contact veterinarian urgently. Consider emergency harvest.',
        'severity': 'Critical'
    },
    'WSSV_BG': {
        'status': 'WSSV + Black Gill (Combined)',
        'description': 'Infected with both WSSV and Black Gill simultaneously. High mortality risk.',
        'recommendation': 'CRITICAL: Immediate isolation required. Emergency harvest recommended. Contact veterinarian immediately.',
        'severity': 'Critical'
    }
}

FISH_INFO = {
    'Healthy Fish': {
        'status': 'Healthy Fish',
        'description': 'Your fish appears healthy with no visible signs of disease.',
        'recommendation': 'Continue current feeding schedule. Monitor water quality parameters regularly.',
        'severity': 'None'
    },
    'Bacterial Red disease': {
        'status': 'Bacterial Red Disease',
        'description': 'Bacterial infection causing reddening of fins, skin and body.',
        'recommendation': 'Isolate affected fish. Improve water quality. Consult veterinarian for antibiotic treatment.',
        'severity': 'Moderate'
    },
    'Bacterial diseases - Aeromoniasis': {
        'status': 'Aeromoniasis',
        'description': 'Bacterial disease caused by Aeromonas species. Causes hemorrhaging and ulcers.',
        'recommendation': 'Isolate affected fish. Reduce stocking density. Consult veterinarian for treatment.',
        'severity': 'Moderate'
    },
    'Bacterial gill disease': {
        'status': 'Bacterial Gill Disease',
        'description': 'Bacterial infection affecting gill tissue. Causes breathing difficulty.',
        'recommendation': 'Improve water oxygenation. Reduce organic matter. Consult veterinarian.',
        'severity': 'Moderate'
    },
    'EUS Disease': {
        'status': 'Epizootic Ulcerative Syndrome (EUS)',
        'description': 'Serious fungal disease causing deep ulcers. Spreads rapidly in ponds.',
        'recommendation': 'IMMEDIATE ACTION: Isolate affected fish. Report to fisheries department. Do not move water between ponds.',
        'severity': 'Critical'
    },
    'Fungal diseases Saprolegniasis': {
        'status': 'Saprolegniasis (Fungal)',
        'description': 'Fungal infection appearing as white cotton-like growth on skin.',
        'recommendation': 'Improve water quality. Reduce stress factors. Consult veterinarian for antifungal treatment.',
        'severity': 'Moderate'
    },
    'Parasitic diseases': {
        'status': 'Parasitic Disease',
        'description': 'Parasitic infection affecting skin, gills or internal organs.',
        'recommendation': 'Identify parasite type. Consult veterinarian for appropriate antiparasitic treatment.',
        'severity': 'Moderate'
    },
    'Viral diseases White tail disease': {
        'status': 'White Tail Disease (Viral)',
        'description': 'Viral disease causing white discoloration of tail muscle. High mortality in juveniles.',
        'recommendation': 'IMMEDIATE ACTION: No cure available. Isolate affected fish. Prevent spread to other ponds.',
        'severity': 'Critical'
    }
}

# ================================
# LOAD MODEL
# ================================
@st.cache_resource
def load_shrimp_model():
    return ort.InferenceSession("shrimp_model.onnx")

@st.cache_resource
def load_fish_model():
    return ort.InferenceSession("fish_model_b3.onnx")

# ================================
# PREPROCESS
# ================================
def preprocess(image, img_size):
    image = image.convert('RGB')
    img = image.resize((img_size, img_size))
    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# ================================
# PREDICT
# ================================
def predict(image, session, class_names, img_size):
    img_array = preprocess(image, img_size)
    input_name = session.get_inputs()[0].name
    predictions = session.run(None, {input_name: img_array})[0]
    predicted_class = class_names[np.argmax(predictions)]
    confidence = float(np.max(predictions)) * 100
    return predicted_class, confidence, predictions[0]

# ================================
# PAGE CONFIG
# ================================
st.set_page_config(
    page_title="AquaVision — Disease Detection",
    page_icon="🐟",
    layout="wide"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root { color-scheme: light only !important; }

html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
    font-family: 'Inter', sans-serif !important;
    background-color: #F0F4F8 !important;
    color: #1A202C !important;
}

/* Force light mode everywhere */
* { color-scheme: light only; }

.block-container {
    max-width: 1000px;
    padding-top: 1.5rem;
    padding-left: 1.5rem;
    padding-right: 1.5rem;
    padding-bottom: 2rem;
}

/* Hide default Streamlit chrome */
#MainMenu, footer { visibility: hidden; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #1A202C;
}
section[data-testid="stSidebar"] * {
    color: #E2E8F0 !important;
}
section[data-testid="stSidebar"] hr {
    border-color: #2D3748;
}

/* Header */
.av-header {
    background: #FFFFFF;
    border-radius: 16px;
    padding: 28px 32px;
    margin-bottom: 24px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    border-bottom: 3px solid #3182CE;
}
.av-header-top {
    display: flex;
    align-items: center;
    gap: 14px;
    margin-bottom: 20px;
}
.av-header-text h1 {
    color: #1A202C;
    font-size: 2rem;
    font-weight: 800;
    margin: 0;
    letter-spacing: -0.5px;
}
.av-header-text p {
    color: #718096;
    font-size: 0.9rem;
    margin: 3px 0 0 0;
}
.av-hero-stats {
    display: flex;
    gap: 24px;
    flex-wrap: wrap;
    padding-top: 16px;
    border-top: 1px solid #EDF2F7;
}
.av-hero-stat {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 0.88rem;
    font-weight: 600;
    color: #4A5568;
}
.av-hero-stat-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #38A169;
    flex-shrink: 0;
}

/* Stats row */
.av-stats {
    display: flex;
    gap: 12px;
    margin-bottom: 24px;
    flex-wrap: wrap;
}
.av-stat-card {
    background: #FFFFFF;
    border-radius: 12px;
    padding: 14px 20px;
    flex: 1;
    min-width: 120px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    border-left: 4px solid #3182CE;
}
.av-stat-label {
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #718096;
    margin-bottom: 4px;
}
.av-stat-value {
    font-size: 1.35rem;
    font-weight: 800;
    color: #1A202C;
}

/* Species toggle */
.av-species-row {
    background: #FFFFFF;
    border-radius: 12px;
    padding: 16px 20px;
    margin-bottom: 20px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08);
}

/* Hide the default radio and make big pill buttons */
div[data-testid="stRadio"] > label {
    display: none !important;
}
div[data-testid="stRadio"] > div {
    display: flex !important;
    gap: 12px !important;
    flex-direction: row !important;
}
div[data-testid="stRadio"] > div > label {
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    padding: 14px 32px !important;
    border-radius: 50px !important;
    border: 2px solid #CBD5E0 !important;
    background: #FFFFFF !important;
    cursor: pointer !important;
    font-size: 1.1rem !important;
    font-weight: 700 !important;
    color: #1A202C !important;
    transition: all 0.15s !important;
    min-width: 130px !important;
}
div[data-testid="stRadio"] > div > label:hover {
    border-color: #3182CE !important;
    color: #3182CE !important;
    background: #EBF8FF !important;
}
div[data-testid="stRadio"] > div > label[data-checked="true"],
div[data-testid="stRadio"] > div > label:has(input:checked) {
    background: #1A202C !important;
    border-color: #1A202C !important;
    color: #FFFFFF !important;
}
div[data-testid="stRadio"] > div > label > div:first-child {
    display: none !important;
}
div[data-testid="stRadio"] > div > label p {
    font-size: 1.1rem !important;
    font-weight: 700 !important;
    margin: 0 !important;
    color: #1A202C !important;
}

/* Upload button text fix */
div[data-testid="stFileUploader"] button {
    color: #1A202C !important;
    background: #FFFFFF !important;
    border: 1px solid #CBD5E0 !important;
    font-weight: 600 !important;
}
div[data-testid="stFileUploader"] small,
div[data-testid="stFileUploader"] span,
div[data-testid="stFileUploader"] p {
    color: #4A5568 !important;
}

/* Upload area */
.av-upload-wrap {
    background: #FFFFFF;
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 24px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08);
}
div[data-testid="stFileUploader"] {
    border: 2px dashed #BEE3F8;
    border-radius: 10px;
    padding: 8px;
    background: #EBF8FF;
}
div[data-testid="stFileUploader"]:hover {
    border-color: #3182CE;
    background: #E0F0FF;
}

/* Result banner */
.av-result-none {
    background: linear-gradient(135deg, #276749, #38A169);
    border-radius: 14px;
    padding: 20px 24px;
    margin-bottom: 16px;
    color: white;
}
.av-result-moderate {
    background: linear-gradient(135deg, #975A16, #DD6B20);
    border-radius: 14px;
    padding: 20px 24px;
    margin-bottom: 16px;
    color: white;
}
.av-result-critical {
    background: linear-gradient(135deg, #742A2A, #C53030);
    border-radius: 14px;
    padding: 20px 24px;
    margin-bottom: 16px;
    color: white;
}
.av-result-banner-label {
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    opacity: 0.85;
    margin-bottom: 6px;
}
.av-result-banner-title {
    font-size: 1.6rem;
    font-weight: 800;
    margin: 0;
    letter-spacing: -0.3px;
}
.av-result-banner-conf {
    font-size: 0.9rem;
    opacity: 0.9;
    margin-top: 6px;
}

/* Info cards */
.av-info-card {
    background: #FFFFFF;
    border-radius: 12px;
    padding: 16px 20px;
    margin-bottom: 12px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.07);
}
.av-info-card-label {
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #718096;
    margin-bottom: 6px;
}
.av-info-card-text {
    font-size: 0.95rem;
    color: #2D3748;
    line-height: 1.5;
}

/* Action card */
.av-action-none {
    background: #F0FFF4;
    border-left: 4px solid #38A169;
    border-radius: 0 10px 10px 0;
    padding: 14px 18px;
    margin-bottom: 12px;
}
.av-action-moderate {
    background: #FFFAF0;
    border-left: 4px solid #DD6B20;
    border-radius: 0 10px 10px 0;
    padding: 14px 18px;
    margin-bottom: 12px;
}
.av-action-critical {
    background: #FFF5F5;
    border-left: 4px solid #C53030;
    border-radius: 0 10px 10px 0;
    padding: 14px 18px;
    margin-bottom: 12px;
}
.av-action-text {
    font-size: 0.93rem;
    color: #2D3748;
    line-height: 1.55;
    font-weight: 500;
}

/* Confidence metrics */
.av-metrics-row {
    display: flex;
    gap: 12px;
    margin-bottom: 12px;
}
.av-metric {
    background: #FFFFFF;
    border-radius: 10px;
    padding: 12px 16px;
    flex: 1;
    box-shadow: 0 1px 3px rgba(0,0,0,0.07);
    text-align: center;
}
.av-metric-label {
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #718096;
}
.av-metric-value {
    font-size: 1.5rem;
    font-weight: 800;
    color: #1A202C;
    margin-top: 2px;
}

/* Reliability badge */
.av-badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.04em;
}
.av-badge-high { background: #C6F6D5; color: #276749; }
.av-badge-medium { background: #FEEBC8; color: #975A16; }
.av-badge-low { background: #FED7D7; color: #742A2A; }

/* Section heading */
.av-section-heading {
    font-size: 0.8rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #718096;
    margin: 20px 0 10px 0;
}

/* Top predictions */
.av-pred-row {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 10px;
    background: #FFFFFF;
    border-radius: 10px;
    padding: 10px 14px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
}
.av-pred-rank {
    font-size: 0.75rem;
    font-weight: 700;
    color: #A0AEC0;
    width: 20px;
}
.av-pred-name {
    font-size: 0.9rem;
    font-weight: 600;
    color: #2D3748;
    flex: 1;
}
.av-pred-pct {
    font-size: 0.9rem;
    font-weight: 700;
    color: #3182CE;
    width: 52px;
    text-align: right;
}
.av-pred-bar-wrap {
    width: 100%;
    background: #EDF2F7;
    border-radius: 4px;
    height: 6px;
    margin-top: 4px;
}
.av-pred-bar-fill {
    height: 6px;
    border-radius: 4px;
    background: #3182CE;
}

/* Divider */
.av-divider {
    border: none;
    border-top: 1px solid #E2E8F0;
    margin: 20px 0;
}

/* Disclaimer */
.av-disclaimer {
    background: #EDF2F7;
    border-radius: 10px;
    padding: 12px 16px;
    font-size: 0.8rem;
    color: #718096;
    margin-top: 20px;
    line-height: 1.5;
}

/* Low confidence warning */
.av-warn {
    background: #FFFFF0;
    border: 1px solid #ECC94B;
    border-radius: 10px;
    padding: 12px 16px;
    font-size: 0.88rem;
    color: #744210;
    margin-bottom: 12px;
    font-weight: 500;
}

/* Radio buttons */
div[data-testid="stRadio"] label {
    font-size: 1rem !important;
    font-weight: 600 !important;
    color: #2D3748 !important;
}
            
div[data-testid="stRadio"] > div > label:has(input:checked) p {
    color: #FFFFFF !important;
}            

/* Plotly chart container */
.av-chart-wrap {
    background: #FFFFFF;
    border-radius: 12px;
    padding: 4px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.07);
    margin-top: 4px;
}
</style>
""", unsafe_allow_html=True)

# ================================
# HEADER
# ================================
st.markdown(f"""
<div class="av-header">
    <div class="av-header-top">
        <div style="font-size:2.6rem;line-height:1">🐟</div>
        <div class="av-header-text">
            <h1>AquaVision</h1>
            <p>AI-powered disease detection for aquaculture farmers · VIT-AP University</p>
        </div>
    </div>
    <div class="av-hero-stats">
        <div class="av-hero-stat"><div class="av-hero-stat-dot"></div>85% Shrimp Model Accuracy</div>
        <div class="av-hero-stat"><div class="av-hero-stat-dot"></div>82% Fish Model Accuracy</div>
        <div class="av-hero-stat"><div class="av-hero-stat-dot" style="background:#3182CE"></div>12 Disease Classes</div>
        <div class="av-hero-stat"><div class="av-hero-stat-dot" style="background:#3182CE"></div>4,000+ Training Images</div>
        <div class="av-hero-stat"><div class="av-hero-stat-dot" style="background:#805AD5"></div>Fish + Shrimp Detection</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ================================
# SPECIES TOGGLE
# ================================
st.markdown("""
<div class="av-species-row">
    <div style="font-size:0.8rem;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;color:#718096;margin-bottom:12px">Select Species</div>
""", unsafe_allow_html=True)
species = st.radio("", ["🦐 Shrimp", "🐟 Fish"], horizontal=True, label_visibility="collapsed")
st.markdown('</div>', unsafe_allow_html=True)

is_shrimp = species == "🦐 Shrimp"

if is_shrimp:
    session = load_shrimp_model()
    class_names = SHRIMP_CLASSES
    disease_info = SHRIMP_INFO
    img_size = SHRIMP_IMG_SIZE
    model_arch = "EfficientNetB0"
    num_classes = 4
    val_accuracy = "85%"
    dataset_size = "2,000"
else:
    session = load_fish_model()
    class_names = FISH_CLASSES
    disease_info = FISH_INFO
    img_size = FISH_IMG_SIZE
    model_arch = "EfficientNetB3"
    num_classes = 8
    val_accuracy = "82.2%"
    dataset_size = "2,000"

# ================================
# STATS CARDS
# ================================
st.markdown(f"""
<div class="av-stats">
    <div class="av-stat-card">
        <div class="av-stat-label">Model</div>
        <div class="av-stat-value" style="font-size:1.05rem">{model_arch}</div>
    </div>
    <div class="av-stat-card" style="border-left-color:#38A169">
        <div class="av-stat-label">Val Accuracy</div>
        <div class="av-stat-value">{val_accuracy}</div>
    </div>
    <div class="av-stat-card" style="border-left-color:#805AD5">
        <div class="av-stat-label">Classes</div>
        <div class="av-stat-value">{num_classes}</div>
    </div>
    <div class="av-stat-card" style="border-left-color:#DD6B20">
        <div class="av-stat-label">Training Images</div>
        <div class="av-stat-value">{dataset_size}</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ================================
# SIDEBAR
# ================================
with st.sidebar:
    st.markdown("## 🌊 AquaVision")
    st.markdown("AI-powered disease screening for fish and shrimp farming.")
    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
Built by **Mahesh Penubothu**  
VIT-AP University, CSE

---

### How to use
1. Select species (Shrimp / Fish)
2. Upload a clear photo
3. Get instant diagnosis

---

### Grad-CAM
The model highlights the exact region it focused on to make each prediction.
""")
    try:
        st.image("assets/gradcam_all_classes.png", caption="Model attention map", use_container_width=True)
    except:
        pass

# ================================
# UPLOAD
# ================================
st.markdown('<div class="av-upload-wrap">', unsafe_allow_html=True)
st.markdown("**📷 Upload Image**")
st.caption("Upload a clear, well-lit photo of a single fish or shrimp. JPG, PNG, or WebP.")
uploaded_file = st.file_uploader("", type=['jpg', 'jpeg', 'png', 'webp'], label_visibility="collapsed")
st.markdown('</div>', unsafe_allow_html=True)

# ================================
# RESULTS
# ================================
if uploaded_file is not None:
    image = Image.open(uploaded_file)

    col1, col2 = st.columns([1.05, 0.95])

    with col1:
        st.image(image, use_container_width=True)

    with col2:
        with st.spinner("Analyzing image..."):
            predicted_class, confidence, all_probs = predict(image, session, class_names, img_size)

        info = disease_info[predicted_class]
        severity = info['severity']

        # Low confidence warning
        if confidence < 60:
            st.markdown('<div class="av-warn">⚠️ Low confidence — upload a clearer, well-lit photo for a more reliable result.</div>', unsafe_allow_html=True)

        # Result banner
        banner_class = {
            'None': 'av-result-none',
            'Moderate': 'av-result-moderate',
            'Critical': 'av-result-critical'
        }[severity]

        severity_icon = {'None': '✅', 'Moderate': '⚠️', 'Critical': '🚨'}[severity]

        # Reliability
        if confidence >= 95:
            rel_label = "Very High"
            rel_badge = "av-badge-high"
        elif confidence >= 80:
            rel_label = "High"
            rel_badge = "av-badge-high"
        elif confidence >= 60:
            rel_label = "Medium"
            rel_badge = "av-badge-medium"
        else:
            rel_label = "Low"
            rel_badge = "av-badge-low"

        st.markdown(f"""
<div class="{banner_class}">
    <div class="av-result-banner-label">{severity_icon} {severity} severity</div>
    <div class="av-result-banner-title">{info['status']}</div>
    <div class="av-result-banner-conf">
        Confidence: <strong>{confidence:.1f}%</strong> &nbsp;·&nbsp;
        <span class="av-badge {rel_badge}">{rel_label} reliability</span>
    </div>
</div>
""", unsafe_allow_html=True)

        # Metrics
        st.markdown(f"""
<div class="av-metrics-row">
    <div class="av-metric">
        <div class="av-metric-label">Confidence</div>
        <div class="av-metric-value">{confidence:.1f}%</div>
    </div>
    <div class="av-metric">
        <div class="av-metric-label">Class</div>
        <div class="av-metric-value" style="font-size:1.1rem">{predicted_class}</div>
    </div>
</div>
""", unsafe_allow_html=True)

        # Disease info
        st.markdown('<div class="av-info-card"><div class="av-info-card-label">🔬 Disease Information</div><div class="av-info-card-text">' + info['description'] + '</div></div>', unsafe_allow_html=True)

        # Action
        action_class = {
            'None': 'av-action-none',
            'Moderate': 'av-action-moderate',
            'Critical': 'av-action-critical'
        }[severity]
        st.markdown(f'<div class="{action_class}"><div class="av-info-card-label">🚑 Recommended Action</div><div class="av-action-text">{info["recommendation"]}</div></div>', unsafe_allow_html=True)

    # ================================
    # TOP PREDICTIONS
    # ================================
    st.markdown('<div class="av-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="av-section-heading">🎯 Top Predictions</div>', unsafe_allow_html=True)

    top3_indices = np.argsort(all_probs)[::-1][:3]
    for i, idx in enumerate(top3_indices):
        prob = float(all_probs[idx]) * 100
        bar_width = int(prob)
        rank_colors = ["#3182CE", "#718096", "#A0AEC0"]
        st.markdown(f"""
<div class="av-pred-row">
    <div class="av-pred-rank">#{i+1}</div>
    <div style="flex:1">
        <div style="display:flex;justify-content:space-between;align-items:center">
            <span class="av-pred-name">{class_names[idx]}</span>
            <span class="av-pred-pct">{prob:.1f}%</span>
        </div>
        <div class="av-pred-bar-wrap">
            <div class="av-pred-bar-fill" style="width:{bar_width}%;background:{rank_colors[i]}"></div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

    # ================================
    # PLOTLY CHART
    # ================================
    st.markdown('<div class="av-section-heading">📊 Detection Breakdown</div>', unsafe_allow_html=True)

    sorted_indices = np.argsort(all_probs)
    sorted_classes = [class_names[i] for i in sorted_indices]
    sorted_probs = [float(all_probs[i]) * 100 for i in sorted_indices]

    bar_colors = ['#C53030' if p == max(sorted_probs) else '#BEE3F8' for p in sorted_probs]
    text_colors = ['white' if p == max(sorted_probs) else '#2D3748' for p in sorted_probs]

    fig = go.Figure(go.Bar(
        x=sorted_probs,
        y=sorted_classes,
        orientation='h',
        marker_color=bar_colors,
        text=[f'{p:.1f}%' for p in sorted_probs],
        textposition='outside',
        textfont=dict(size=12, color='#2D3748'),
        hovertemplate='%{y}: %{x:.1f}%<extra></extra>'
    ))

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=60, t=10, b=10),
        xaxis=dict(
            range=[0, 110],
            ticksuffix='%',
            gridcolor='#EDF2F7',
            tickfont=dict(color='#718096', size=11),
            title=dict(text='Confidence (%)', font=dict(color='#718096', size=11))
        ),
        yaxis=dict(
            tickfont=dict(color='#2D3748', size=12),
            gridcolor='rgba(0,0,0,0)'
        ),
        height=max(180, len(sorted_classes) * 42),
        showlegend=False
    )

    st.markdown('<div class="av-chart-wrap">', unsafe_allow_html=True)
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    st.markdown('</div>', unsafe_allow_html=True)

    # ================================
    # DISCLAIMER
    # ================================
    st.markdown("""
<div class="av-disclaimer">
    <div style="display:flex;align-items:center;gap:8px;margin-bottom:8px">
        <span style="font-size:1.1rem">🐟</span>
        <span style="font-weight:800;font-size:1rem;color:#2D3748">AquaVision</span>
        <span style="color:#CBD5E0;font-size:0.8rem">· AI-Powered Aquaculture Disease Detection</span>
    </div>
    <div style="color:#718096;font-size:0.82rem;line-height:1.6">
        ⚕️ <strong>Disclaimer:</strong> For preliminary screening only. Not a substitute for professional veterinary diagnosis.
        Always consult a qualified aquaculture specialist for treatment decisions.<br>
        Built by <strong>Mahesh Penubothu</strong> · VIT-AP University · For research and educational purposes.
    </div>
</div>
""", unsafe_allow_html=True)



