import base64
import io
import os
import time

import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import onnxruntime as ort

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
# UI — a single custom component (ui/aquavision.*) that shares
# AquaManage's palette, type and severity bands.
# ================================
st.set_page_config(
    page_title="AquaVision — Disease check for shrimp and fish",
    page_icon="🦐",
    layout="wide",
    initial_sidebar_state="collapsed",
)

UI_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ui")
STATIC = "app/static"  # served from ./static (server.enableStaticServing)


def read_ui(name):
    with open(os.path.join(UI_DIR, name), encoding="utf-8") as f:
        return f.read()


aquavision = st.components.v2.component(
    "aquavision",
    css=read_ui("aquavision.css"),
    js=read_ui("aquavision.js"),
)

SHORT_NAMES = {
    'Healthy': 'Healthy', 'BG': 'Black gill', 'WSSV': 'White spot (WSSV)', 'WSSV_BG': 'WSSV + black gill',
    'Healthy Fish': 'Healthy', 'Bacterial Red disease': 'Red disease',
    'Bacterial diseases - Aeromoniasis': 'Aeromoniasis', 'Bacterial gill disease': 'Gill disease',
    'EUS Disease': 'EUS', 'Fungal diseases Saprolegniasis': 'Saprolegniasis',
    'Parasitic diseases': 'Parasites', 'Viral diseases White tail disease': 'White tail',
}
SEVERITY_ORDER = {'none': 0, 'moderate': 1, 'critical': 2}

SPECIES = {
    'shrimp': dict(
        name="Shrimp", load=load_shrimp_model, classes=SHRIMP_CLASSES, info=SHRIMP_INFO, size=SHRIMP_IMG_SIZE,
        arch="EfficientNet-B0", accuracy="85%",
        # Specimen placement on the plate (desktop, then phone), as CSS lengths.
        stage=dict(image=f"{STATIC}/stage/shrimp.webp", mask=f"{STATIC}/stage/shrimp.webp", blend="normal",
                   x="44%", y="12%", w="53%", xm="6%", ym="34%", wm="92%", shadow="-2%",
                   caption="Specimen — Indian white shrimp · photo Chan T. Y. & Lin C. W. (MNHN), CC BY 4.0"),
        samples=[("shrimp_healthy", "Healthy"), ("shrimp_bg", "Black gill"),
                 ("shrimp_wssv", "White spot"), ("shrimp_wssv_bg", "WSSV + black gill")],
    ),
    'fish': dict(
        name="Fish", load=load_fish_model, classes=FISH_CLASSES, info=FISH_INFO, size=FISH_IMG_SIZE,
        arch="EfficientNet-B3", accuracy="82%",
        stage=dict(image=f"{STATIC}/stage/tilapia.webp", mask=f"{STATIC}/stage/tilapia.webp", blend="normal",
                   x="42%", y="15%", w="55%", xm="16%", ym="43%", wm="80%", shadow="-2%",
                   caption="Specimen — Nile tilapia · photo G. R. Schüür, CC BY-SA 4.0"),
        samples=[("fish_healthy", "Healthy"), ("fish_gill", "Gill disease"), ("fish_eus", "EUS")],
    ),
}
SAMPLE_FILES = {sid: os.path.join("static", "samples", f"{sid}.jpg")
                for sp in SPECIES.values() for sid, _ in sp['samples']}


def severity_key(info):
    return info['severity'].lower()


def build_config():
    species = []
    for key, sp in SPECIES.items():
        conditions = sorted(
            ({'short': SHORT_NAMES[c], 'severity': severity_key(sp['info'][c])} for c in sp['classes']),
            key=lambda c: SEVERITY_ORDER[c['severity']],
        )
        species.append({
            'key': key, 'name': sp['name'], 'arch': sp['arch'], 'accuracy': sp['accuracy'], 'input': sp['size'],
            'conditions': conditions, 'stage': sp['stage'],
            'samples': [{'id': sid, 'label': label, 'src': f"{STATIC}/samples/{sid}.jpg"} for sid, label in sp['samples']],
        })
    return {
        'species': species,
        'stats': [{'value': '12', 'label': 'Conditions'}, {'value': '2', 'label': 'Species'},
                  {'value': '4k+', 'label': 'Training photos'}, {'value': '<1 s', 'label': 'Per check'}],
        'gradcam': f"{STATIC}/gradcam.webp",
        'credits': [{'what': 'Shrimp photo', 'who': 'Chan T. Y. & Lin C. W. (MNHN)', 'license': 'CC BY 4.0',
                     'url': 'https://commons.wikimedia.org/wiki/File:Fenneropenaeus_indicus_(MNHN-IU-2011-5728).jpeg'},
                    {'what': 'Tilapia photo', 'who': 'Germano Roberto Schüür', 'license': 'CC BY-SA 4.0',
                     'url': 'https://commons.wikimedia.org/wiki/File:Til%C3%A1pia_ou_Sarotherodon_niloticus.jpg'}],
    }


def to_data_uri(image, max_side=1200):
    im = image.convert('RGB')
    im.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    im.save(buf, format='JPEG', quality=85)
    return 'data:image/jpeg;base64,' + base64.b64encode(buf.getvalue()).decode()


def analyse(req):
    """Run one check requested by the front end and return a JSON-safe result."""
    rid = str(req.get('id', ''))[:32]
    sp = SPECIES.get(req.get('species'))
    if sp is None:
        return {'id': rid, 'error': "Pick shrimp or fish first."}
    try:
        if req.get('sample') in SAMPLE_FILES:
            image = Image.open(SAMPLE_FILES[req['sample']])
            label = "Sample · " + dict(sp['samples']).get(req['sample'], "photo")
            preview = f"{STATIC}/samples/{req['sample']}.jpg"
        else:
            raw = req.get('image') or ''
            if len(raw) > 20_000_000:
                return {'id': rid, 'error': "That photo is too large. Use one under 15 MB."}
            image = Image.open(io.BytesIO(base64.b64decode(raw)))
            label, preview = "Your photo", None
        # Phone photos carry their rotation in EXIF; apply it so the model sees the animal upright.
        image = ImageOps.exif_transpose(image)
        image.load()
    except Exception:
        return {'id': rid, 'error': "That file couldn't be read as a photo. Try a JPG or PNG."}

    t0 = time.perf_counter()
    predicted, confidence, probs = predict(image, sp['load'](), sp['classes'], sp['size'])
    ms = (time.perf_counter() - t0) * 1000

    info = sp['info'][predicted]
    ranked = sorted(
        ({'status': sp['info'][c]['status'], 'severity': severity_key(sp['info'][c]), 'p': float(probs[i]) * 100}
         for i, c in enumerate(sp['classes'])),
        key=lambda r: r['p'], reverse=True,
    )
    return {
        'id': rid, 'species': req['species'], 'status': info['status'], 'description': info['description'],
        'recommendation': info['recommendation'], 'severity': severity_key(info),
        'confidence': confidence, 'ms': ms, 'probs': ranked,
        'label': label, 'preview': preview or to_data_uri(image),
    }


# Page chrome: let the component own the whole viewport.
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wdth,wght@12..96,75..100,400..800&family=Noto+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');
html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] { background: #0a1418 !important; }
header[data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDecoration"], #MainMenu, footer { display: none !important; }
.block-container, [data-testid="stMainBlockContainer"] { max-width: none !important; padding: 0 !important; }
[data-testid="stVerticalBlock"] { gap: 0 !important; }
[data-testid="stElementContainer"]:has(style) { display: none; }
</style>
""", unsafe_allow_html=True)

if "av_result" not in st.session_state:
    st.session_state.av_result = None

ui = aquavision(
    key="aquavision",
    data={'config': build_config(), 'result': st.session_state.av_result},
    on_analyze_change=lambda: None,
)

request = ui.analyze
if request and request.get('id') != st.session_state.get('av_last_id'):
    st.session_state.av_last_id = request.get('id')
    st.session_state.av_result = analyse(request)
    st.rerun()
