import streamlit as st
import numpy as np
from PIL import Image
import onnxruntime as ort

# ================================
# CONFIGURATION
# ================================
MODEL_PATH = "shrimp_model.onnx"
IMG_SIZE = 224

CLASS_NAMES = ['BG', 'Healthy', 'WSSV', 'WSSV_BG']

DISEASE_INFO = {
    'Healthy': {
        'status': '✅ Healthy',
        'description': 'Your shrimp appears healthy with no visible signs of disease.',
        'recommendation': 'Continue current feeding schedule. Monitor water quality parameters regularly. Maintain proper stocking density.',
        'severity': 'None'
    },
    'BG': {
        'status': '⚠️ Black Gill Disease (BG)',
        'description': 'Black Gill disease is caused by environmental stress, parasites, or bacterial infection. Gills appear darkened or blackened.',
        'recommendation': 'Improve water quality immediately. Check ammonia and nitrite levels. Reduce stocking density. Consult a veterinarian for treatment options.',
        'severity': 'Moderate'
    },
    'WSSV': {
        'status': '🚨 White Spot Syndrome Virus (WSSV)',
        'description': 'WSSV is the most devastating shrimp disease globally. White spots visible on shell and body. Spreads rapidly through pond.',
        'recommendation': 'IMMEDIATE ACTION REQUIRED. Isolate affected shrimp immediately. Do not share equipment between ponds. Contact veterinarian urgently. Consider emergency harvest to minimize losses.',
        'severity': 'Critical'
    },
    'WSSV_BG': {
        'status': '🚨 WSSV + Black Gill (Combined Infection)',
        'description': 'Shrimp is infected with both White Spot Syndrome Virus and Black Gill disease simultaneously. High mortality risk.',
        'recommendation': 'CRITICAL: Multiple infections detected. Immediate isolation required. Emergency harvest recommended. Contact veterinarian immediately. Do not transfer water or equipment to other ponds.',
        'severity': 'Critical'
    }
}

# ================================
# LOAD MODEL
# ================================
@st.cache_resource
def load_disease_model():
    session = ort.InferenceSession(MODEL_PATH)
    return session

def preprocess(image):
    image = image.convert('RGB')
    img = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img, dtype=np.float32)
    # EfficientNet preprocessing — same as training
    mean = np.array([103.939, 116.779, 123.68], dtype=np.float32)
    img_array = img_array[..., ::-1]  # RGB to BGR
    img_array = img_array - mean
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict(image, session):
    img_array = preprocess(image)
    input_name = session.get_inputs()[0].name
    predictions = session.run(None, {input_name: img_array})[0]
    predicted_class = CLASS_NAMES[np.argmax(predictions)]
    confidence = float(np.max(predictions)) * 100
    return predicted_class, confidence, predictions[0]

# ================================
# PAGE CONFIG
# ================================
st.set_page_config(
    page_title="AquaVision — Shrimp Disease Detection",
    page_icon="🦐",
    layout="centered"
)

# ================================
# HEADER
# ================================
st.title("🦐 AquaVision")
st.subheader("Shrimp Disease Detection Tool")
st.write("Upload a photo of your shrimp to detect diseases instantly.")
st.markdown("---")

# ================================
# MODEL LOAD
# ================================
model = load_disease_model()

# ================================
# UPLOAD
# ================================
uploaded_file = st.file_uploader(
    "Upload Shrimp Image",
    type=['jpg', 'jpeg', 'png', 'webp'],
    help="Take a clear photo of the shrimp and upload it here"
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    col1, col2 = st.columns([1, 1])
    with col1:
        st.image(image, caption="Uploaded Image", use_container_width=True)

    with col2:
        with st.spinner("Analyzing..."):
            predicted_class, confidence, all_probs = predict(image, model)

        info = DISEASE_INFO[predicted_class]

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
    for i, class_name in enumerate(CLASS_NAMES):
        prob = float(all_probs[i]) * 100
        st.write(f"{class_name}: {prob:.1f}%")
        st.progress(prob / 100)

    st.markdown("---")
    st.caption("AquaVision — Built for Indian shrimp farmers | VIT-AP University")