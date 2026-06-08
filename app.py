import streamlit as st
import numpy as np
from PIL import Image
import onnxruntime as ort

# ================================
# CONFIGURATION
# ================================
IMG_SIZE = 224

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
        'status': '✅ Healthy',
        'description': 'Your shrimp appears healthy with no visible signs of disease.',
        'recommendation': 'Continue current feeding schedule. Monitor water quality parameters regularly.',
        'severity': 'None'
    },
    'BG': {
        'status': '⚠️ Black Gill Disease (BG)',
        'description': 'Black Gill disease caused by environmental stress or bacterial infection.',
        'recommendation': 'Improve water quality immediately. Check ammonia and nitrite levels. Reduce stocking density.',
        'severity': 'Moderate'
    },
    'WSSV': {
        'status': '🚨 White Spot Syndrome Virus (WSSV)',
        'description': 'Most devastating shrimp disease globally. White spots visible on shell and body.',
        'recommendation': 'IMMEDIATE ACTION REQUIRED. Isolate affected shrimp. Contact veterinarian urgently. Consider emergency harvest.',
        'severity': 'Critical'
    },
    'WSSV_BG': {
        'status': '🚨 WSSV + Black Gill (Combined Infection)',
        'description': 'Infected with both WSSV and Black Gill simultaneously. High mortality risk.',
        'recommendation': 'CRITICAL: Immediate isolation required. Emergency harvest recommended. Contact veterinarian immediately.',
        'severity': 'Critical'
    }
}

FISH_INFO = {
    'Healthy Fish': {
        'status': '✅ Healthy Fish',
        'description': 'Your fish appears healthy with no visible signs of disease.',
        'recommendation': 'Continue current feeding schedule. Monitor water quality parameters regularly.',
        'severity': 'None'
    },
    'Bacterial Red disease': {
        'status': '⚠️ Bacterial Red Disease',
        'description': 'Bacterial infection causing reddening of fins, skin and body.',
        'recommendation': 'Isolate affected fish. Improve water quality. Consult veterinarian for antibiotic treatment.',
        'severity': 'Moderate'
    },
    'Bacterial diseases - Aeromoniasis': {
        'status': '⚠️ Aeromoniasis',
        'description': 'Bacterial disease caused by Aeromonas species. Causes hemorrhaging and ulcers.',
        'recommendation': 'Isolate affected fish. Reduce stocking density. Consult veterinarian for treatment.',
        'severity': 'Moderate'
    },
    'Bacterial gill disease': {
        'status': '⚠️ Bacterial Gill Disease',
        'description': 'Bacterial infection affecting gill tissue. Causes breathing difficulty.',
        'recommendation': 'Improve water oxygenation. Reduce organic matter. Consult veterinarian.',
        'severity': 'Moderate'
    },
    'EUS Disease': {
        'status': '🚨 Epizootic Ulcerative Syndrome (EUS)',
        'description': 'Serious fungal disease causing deep ulcers. Spreads rapidly in ponds.',
        'recommendation': 'IMMEDIATE ACTION: Isolate affected fish. Report to fisheries department. Do not move water between ponds.',
        'severity': 'Critical'
    },
    'Fungal diseases Saprolegniasis': {
        'status': '⚠️ Saprolegniasis (Fungal)',
        'description': 'Fungal infection appearing as white cotton-like growth on skin.',
        'recommendation': 'Improve water quality. Reduce stress factors. Consult veterinarian for antifungal treatment.',
        'severity': 'Moderate'
    },
    'Parasitic diseases': {
        'status': '⚠️ Parasitic Disease',
        'description': 'Parasitic infection affecting skin, gills or internal organs.',
        'recommendation': 'Identify parasite type. Consult veterinarian for appropriate antiparasitic treatment.',
        'severity': 'Moderate'
    },
    'Viral diseases White tail disease': {
        'status': '🚨 White Tail Disease (Viral)',
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
    return ort.InferenceSession("fish_model.onnx")

# ================================
# PREPROCESS
# ================================
def preprocess(image):
    image = image.convert('RGB')
    img = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img, dtype=np.float32)
    mean = np.array([103.939, 116.779, 123.68], dtype=np.float32)
    img_array = img_array[..., ::-1]
    img_array = img_array - mean
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# ================================
# PREDICT
# ================================
def predict(image, session, class_names):
    img_array = preprocess(image)
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
    layout="centered"
)

# ================================
# HEADER
# ================================
st.title("🐟 AquaVision")
st.subheader("Aquaculture Disease Detection Tool")
st.write("Upload a photo of your shrimp or fish to detect diseases instantly.")
st.markdown("---")

# ================================
# SPECIES TOGGLE
# ================================
species = st.radio(
    "Select species:",
    ["🦐 Shrimp", "🐟 Fish"],
    horizontal=True
)

is_shrimp = species == "🦐 Shrimp"

if is_shrimp:
    session = load_shrimp_model()
    class_names = SHRIMP_CLASSES
    disease_info = SHRIMP_INFO
    upload_label = "Upload Shrimp Image"
else:
    session = load_fish_model()
    class_names = FISH_CLASSES
    disease_info = FISH_INFO
    upload_label = "Upload Fish Image"

# ================================
# UPLOAD
# ================================
uploaded_file = st.file_uploader(
    upload_label,
    type=['jpg', 'jpeg', 'png', 'webp'],
    help="Take a clear photo and upload it here"
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    col1, col2 = st.columns([1, 1])
    with col1:
        st.image(image, caption="Uploaded Image", use_container_width=True)

    with col2:
        with st.spinner("Analyzing..."):
            predicted_class, confidence, all_probs = predict(image, session, class_names)

        info = disease_info[predicted_class]

        st.markdown(f"### {info['status']}")
        st.metric("Confidence", f"{confidence:.1f}%")
        st.progress(confidence / 100)

        severity = info['severity']
        if severity == 'None':
            st.success(f"Severity: {severity}")
        elif severity == 'Moderate':
            st.warning(f"Severity: {severity}")
        else:
            st.error(f"Severity: {severity}")

    st.markdown("---")
    st.markdown("### What This Means")
    st.write(info['description'])

    st.markdown("### Recommended Action")
    st.info(info['recommendation'])

    st.markdown("### Detection Breakdown")
    for i, class_name in enumerate(class_names):
        prob = float(all_probs[i]) * 100
        st.write(f"{class_name}: {prob:.1f}%")
        st.progress(prob / 100)

    st.markdown("---")
    st.caption("AquaVision — Built for aquaculture farmers | VIT-AP University")