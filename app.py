import base64
import io
import os
import time
from urllib.parse import quote

import streamlit as st
import numpy as np
from PIL import Image, ImageOps
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
    page_title="AquaVision — Disease check",
    page_icon="🦐",
    layout="wide",
    initial_sidebar_state="collapsed",
)

SPECIES = {
    'Shrimp': dict(load=load_shrimp_model, classes=SHRIMP_CLASSES, info=SHRIMP_INFO,
                   size=SHRIMP_IMG_SIZE, arch="EfficientNetB0"),
    'Fish': dict(load=load_fish_model, classes=FISH_CLASSES, info=FISH_INFO,
                 size=FISH_IMG_SIZE, arch="EfficientNetB3"),
}

SAMPLES = {
    'Shrimp': [("samples/shrimp_healthy.jpg.jpg", "Healthy shrimp"),
               ("samples/shrimp_BlackGill.jpg.jpg", "Black gill")],
    'Fish': [("samples/fish_healthy.jpg.png", "Healthy fish"),
             ("samples/fish_bacterial_red.jpg.jpg", "Red disease")],
}

# Same three bands and words as the AquaManage landing page.
SEVERITY = {
    'None': dict(key='none', label='Healthy', rank='Lowest of the three levels', band='No disease seen', bar='#7fb35a'),
    'Moderate': dict(key='moderate', label='Moderate', rank='Middle of the three levels', band='Watch closely', bar='#e0a430'),
    'Critical': dict(key='critical', label='Critical', rank='Highest of the three levels', band='Act today', bar='#e0574b'),
}

# Lucide icon paths, inlined so the page needs no icon font.
ICONS = {
    'check': '<circle cx="12" cy="12" r="10"/><path d="m9 12 2 2 4-4"/>',
    'triangle': '<path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3"/><path d="M12 9v4"/><path d="M12 17h.01"/>',
    'octagon': '<path d="M12 16h.01"/><path d="M12 8v4"/><path d="M15.312 2a2 2 0 0 1 1.414.586l4.688 4.688A2 2 0 0 1 22 8.688v6.624a2 2 0 0 1-.586 1.414l-4.688 4.688a2 2 0 0 1-1.414.586H8.688a2 2 0 0 1-1.414-.586l-4.688-4.688A2 2 0 0 1 2 15.312V8.688a2 2 0 0 1 .586-1.414l4.688-4.688A2 2 0 0 1 8.688 2z"/>',
    'camera': '<path d="M14.5 4h-5L7 7H4a2 2 0 0 0-2 2v9a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2V9a2 2 0 0 0-2-2h-3l-2.5-3z"/><circle cx="12" cy="13" r="3"/>',
    'scan': '<path d="M3 7V5a2 2 0 0 1 2-2h2"/><path d="M17 3h2a2 2 0 0 1 2 2v2"/><path d="M21 17v2a2 2 0 0 1-2 2h-2"/><path d="M7 21H5a2 2 0 0 1-2-2v-2"/><circle cx="12" cy="12" r="1"/><path d="M18.944 12.33a1 1 0 0 0 0-.66 7.5 7.5 0 0 0-13.888 0 1 1 0 0 0 0 .66 7.5 7.5 0 0 0 13.888 0"/>',
}
SEVERITY_ICON = {'None': 'check', 'Moderate': 'triangle', 'Critical': 'octagon'}


def icon(name, size=20):
    # st.html strips inline <svg>, so icons are CSS masks painted in currentColor (see ICON_CSS).
    return f'<span class="ic ic-{name}" style="width:{size}px;height:{size}px" aria-hidden="true"></span>'


ICON_CSS = "".join(
    '.ic-%s { --m: url("data:image/svg+xml,%s"); }' % (name, quote(
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="black" '
        'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">' + paths + '</svg>'))
    for name, paths in ICONS.items()
)


def to_data_uri(image):
    im = image.convert('RGB')
    im.thumbnail((1400, 1400))
    buf = io.BytesIO()
    im.save(buf, format='JPEG', quality=85)
    return 'data:image/jpeg;base64,' + base64.b64encode(buf.getvalue()).decode()


# ================================
# STYLES — tokens from AquaManage (CLAUDE.md §6)
# ================================
st.markdown("<style>" + ICON_CSS + "</style>", unsafe_allow_html=True)
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wdth,wght@12..96,75..100,400..800&family=Noto+Sans:wght@400;500;600;700&display=swap');

:root {
    --ink: #12211c; --brackish: #0f5257; --bloom: #4c7a34; --saline: #e0a430;
    --alarm: #b3271e; --dusk: #0a1418; --foam: #f4efe4; --panel: #0f1e22;
    --line: rgba(244,239,228,0.12); --muted: rgba(244,239,228,0.66);
    --display: 'Bricolage Grotesque', system-ui, sans-serif;
    --sans: 'Noto Sans', system-ui, sans-serif;
}

.stApp {
    background:
        radial-gradient(1100px 560px at 85% -8%, rgba(15,82,87,0.55), transparent 62%),
        radial-gradient(800px 480px at -12% 6%, rgba(224,164,48,0.07), transparent 60%),
        var(--dusk) !important;
    color: var(--foam);
    -webkit-font-smoothing: antialiased;
}
.stApp, .stApp button, .stApp input, .stApp p, .stApp label { font-family: var(--sans); }

header[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer, [data-testid="stDecoration"], [data-testid="stToolbar"] { display: none !important; }
.block-container, [data-testid="stMainBlockContainer"] {
    max-width: 1180px; padding: 1.25rem 1.5rem 4rem !important;
}

/* ---------- Nav ---------- */
.av-nav { display: flex; align-items: center; justify-content: space-between; gap: 16px;
          padding: 8px 0 24px; border-bottom: 1px solid var(--line); }
.av-brand { display: flex; align-items: center; gap: 10px; font-family: var(--display);
            font-weight: 700; font-size: 1.3rem; letter-spacing: -0.02em; color: var(--foam); }
.ic { display: inline-block; flex-shrink: 0; background: currentColor; vertical-align: middle;
      -webkit-mask: var(--m) center / contain no-repeat; mask: var(--m) center / contain no-repeat; }
.av-brand .ic { color: var(--saline); }
.av-brand small { font-family: var(--sans); font-weight: 500; font-size: 0.8rem; color: var(--muted); letter-spacing: 0; }
.av-chip { display: inline-flex; align-items: center; gap: 8px; border: 1px solid var(--line);
           border-radius: 999px; padding: 6px 14px; font-size: 0.8rem; color: var(--muted); white-space: nowrap; }
.av-dot { width: 8px; height: 8px; border-radius: 50%; background: #7fb35a; box-shadow: 0 0 0 4px rgba(127,179,90,0.18); }

/* ---------- Hero ---------- */
.av-hero { padding: 64px 0 8px; }
.av-eyebrow { font-size: 0.75rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.22em;
              color: var(--saline); margin: 0; }
.av-display { font-family: var(--display); font-variation-settings: "wdth" 75; font-weight: 700;
              text-transform: uppercase; line-height: 0.88; letter-spacing: -0.02em;
              font-size: clamp(3.2rem, 9vw, 7.5rem); margin: 20px 0 0; padding: 0; color: var(--foam); }
.av-display span { color: var(--saline); }
.av-lede { max-width: 36rem; font-size: 1.125rem; line-height: 1.65; color: var(--muted); margin: 24px 0 0; }
.av-stats { display: grid; grid-template-columns: repeat(4, 1fr); margin-top: 56px; border-top: 1px solid var(--line); }
.av-stat { padding: 20px 20px 0 0; }
.av-stat + .av-stat { padding-left: 20px; border-left: 1px solid var(--line); }
.av-stat-v { font-family: var(--display); font-variation-settings: "wdth" 75; font-weight: 700;
             font-size: 3rem; line-height: 1; font-variant-numeric: tabular-nums; color: var(--foam); }
.av-stat-l { font-size: 0.85rem; color: var(--muted); margin-top: 6px; }
@media (max-width: 640px) {
    .av-stats { grid-template-columns: repeat(2, 1fr); }
    .av-stat:nth-child(3) { border-left: 0; padding-left: 0; }
    .av-stat-v { font-size: 2.4rem; }
    .av-chip { display: none; }
}

/* ---------- Steps ---------- */
.av-step { display: flex; align-items: baseline; gap: 12px; margin: 56px 0 6px; }
.av-step-n { font-family: var(--display); font-size: 0.8rem; font-weight: 700; letter-spacing: 0.12em; color: var(--saline); }
.av-step-t { font-family: var(--display); font-weight: 600; font-size: 1.6rem; letter-spacing: -0.03em; color: var(--foam); }
.av-hint { color: var(--muted); font-size: 0.92rem; margin: 0 0 14px; }
.av-tips { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 4px; }
.av-tips span { border: 1px solid var(--line); border-radius: 999px; padding: 5px 12px; font-size: 0.8rem; color: var(--muted); }

/* Species toggle: radio restyled as big pills (48px+ tap targets) */
div[data-testid="stRadio"] > label { display: none !important; }
div[data-testid="stRadio"] > div { display: flex !important; flex-direction: row !important; gap: 10px !important; }
div[data-testid="stRadio"] > div > label {
    display: flex !important; align-items: center !important; justify-content: center !important;
    min-height: 52px; min-width: 140px; padding: 0 28px !important; margin: 0 !important;
    border-radius: 999px !important; border: 1px solid var(--line) !important;
    background: rgba(244,239,228,0.03) !important; cursor: pointer !important;
    transition: border-color .15s, background .15s !important;
}
div[data-testid="stRadio"] > div > label:hover { border-color: rgba(224,164,48,0.6) !important; }
div[data-testid="stRadio"] > div > label > div:first-child { display: none !important; }
div[data-testid="stRadio"] > div > label p { font-size: 1.05rem !important; font-weight: 600 !important; margin: 0 !important; color: var(--foam) !important; }
div[data-testid="stRadio"] > div > label:has(input:checked) { background: var(--foam) !important; border-color: var(--foam) !important; }
div[data-testid="stRadio"] > div > label:has(input:checked) p { color: var(--ink) !important; }

/* Sample thumbnails */
/* Keep the two samples side by side on phones instead of stacking full-width */
.st-key-samples [data-testid="stHorizontalBlock"] { flex-wrap: nowrap !important; gap: 12px; }
.st-key-samples [data-testid="stColumn"] { min-width: 0 !important; flex: 1 1 0 !important; width: auto !important; }
.st-key-samples [data-testid="stImage"] img { aspect-ratio: 4 / 3; object-fit: cover; border-radius: 16px; }
.stApp [data-testid="stBaseButton-secondary"] {
    background: transparent !important; color: var(--foam) !important; border: 1px solid var(--line) !important;
    border-radius: 999px !important; min-height: 44px; font-weight: 600 !important;
}
.stApp [data-testid="stBaseButton-secondary"]:hover { border-color: var(--saline) !important; color: var(--saline) !important; }

/* Uploader */
[data-testid="stFileUploaderDropzone"] {
    background: rgba(15,30,34,0.85) !important; border: 1.5px dashed rgba(224,164,48,0.45) !important;
    border-radius: 24px !important; padding: 36px 28px !important; min-height: 180px;
}
[data-testid="stFileUploaderDropzone"]:hover { border-color: var(--saline) !important; }
[data-testid="stFileUploaderDropzone"] * { color: var(--muted) !important; }
.stApp [data-testid="stFileUploaderDropzone"] button {
    background: var(--saline) !important; color: var(--ink) !important; border: 0 !important;
    border-radius: 999px !important; font-weight: 700 !important; min-height: 48px; padding: 0 22px !important;
}
[data-testid="stFileUploaderDropzone"] button * { color: var(--ink) !important; }
[data-testid="stFileUploaderFile"] * { color: var(--foam) !important; }

/* ---------- Result ---------- */
.av-figure { position: relative; aspect-ratio: 4 / 3; overflow: hidden; border-radius: 28px; background: #1b2b2a; margin: 0; }
.av-figure img { width: 100%; height: 100%; object-fit: cover; display: block; }
.av-vignette { position: absolute; inset: 0; background: radial-gradient(120% 90% at 50% 45%, transparent 55%, rgba(10,20,24,0.6) 100%); }
.av-corner { position: absolute; width: 38px; height: 38px; border: 0 solid var(--saline); }
.av-corner.tl { top: 18px; left: 18px; border-top-width: 3px; border-left-width: 3px; border-top-left-radius: 12px; }
.av-corner.tr { top: 18px; right: 18px; border-top-width: 3px; border-right-width: 3px; border-top-right-radius: 12px; }
.av-corner.bl { bottom: 18px; left: 18px; border-bottom-width: 3px; border-left-width: 3px; border-bottom-left-radius: 12px; }
.av-corner.br { bottom: 18px; right: 18px; border-bottom-width: 3px; border-right-width: 3px; border-bottom-right-radius: 12px; }
.av-scan { position: absolute; left: 0; right: 0; top: 0; height: 2px; opacity: 0;
           background: linear-gradient(90deg, transparent, var(--saline), transparent);
           box-shadow: 0 0 18px 3px rgba(224,164,48,0.55); animation: av-scan 1.5s ease-in-out 1; }
@keyframes av-scan { 0% { top: 0; opacity: 1; } 90% { opacity: 1; } 100% { top: 100%; opacity: 0; } }
.av-figcap { position: absolute; bottom: 18px; left: 72px; display: flex; align-items: center; gap: 8px;
             border-radius: 999px; background: rgba(10,20,24,0.72); backdrop-filter: blur(8px);
             padding: 8px 14px; font-size: 0.875rem; color: var(--foam); }
.av-meta { margin-top: 14px; font-size: 0.82rem; color: var(--muted); font-variant-numeric: tabular-nums; }

.av-card { background: var(--foam); color: var(--ink); border-radius: 28px; padding: 32px; }
.av-card * { color: inherit; }
.av-sev-row { display: flex; flex-wrap: wrap; align-items: center; gap: 12px; }
.av-sev { display: inline-flex; align-items: center; gap: 8px; border-radius: 999px; padding: 8px 16px;
          font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; font-size: 0.95rem; }
.av-sev.none { background: var(--bloom); color: #fff !important; }
.av-sev.moderate { background: var(--saline); color: var(--ink) !important; }
.av-sev.critical { background: var(--alarm); color: #fff !important; }
.av-sev * { color: inherit !important; }
.av-rank { font-size: 0.875rem; font-weight: 600; }
.av-rank.none { color: var(--bloom) !important; } .av-rank.moderate { color: #9a6b0f !important; } .av-rank.critical { color: var(--alarm) !important; }
.av-bands { display: grid; grid-template-columns: repeat(3, 1fr); gap: 6px; margin-top: 16px; }
.av-bands div { height: 8px; border-radius: 99px; }
.av-bands .b-none { background: rgba(76,122,52,0.2); } .av-bands .b-moderate { background: rgba(224,164,48,0.25); } .av-bands .b-critical { background: rgba(179,39,30,0.15); }
.av-bands .b-none.on { background: var(--bloom); } .av-bands .b-moderate.on { background: var(--saline); } .av-bands .b-critical.on { background: var(--alarm); }
.av-title { font-family: var(--display); letter-spacing: -0.035em; font-weight: 600;
            font-size: clamp(1.8rem, 3vw, 2.4rem); line-height: 1.05; margin: 24px 0 0; padding: 0; color: var(--ink) !important; }
.av-desc { color: rgba(18,33,28,0.72) !important; margin: 10px 0 0; line-height: 1.55; }
.av-conf { display: flex; justify-content: space-between; align-items: baseline; margin-top: 24px; font-size: 0.875rem; }
.av-conf-l { color: rgba(18,33,28,0.62) !important; }
.av-conf-v { font-size: 1.6rem; font-weight: 600; font-variant-numeric: tabular-nums; }
.av-track { height: 10px; border-radius: 99px; background: rgba(18,33,28,0.1); overflow: hidden; margin-top: 8px; }
.av-fill { height: 100%; border-radius: 99px; background: var(--ink); transform-origin: left; animation: av-grow 900ms cubic-bezier(.2,.7,.2,1) both; }
@keyframes av-grow { from { transform: scaleX(0); } }
.av-rel { font-size: 0.72rem; font-weight: 700; padding: 3px 10px; border-radius: 99px; margin-left: 8px; }
.av-rel.high { background: rgba(76,122,52,0.15); color: var(--bloom) !important; }
.av-rel.medium { background: rgba(224,164,48,0.22); color: #7a540a !important; }
.av-rel.low { background: rgba(179,39,30,0.12); color: var(--alarm) !important; }
.av-action { margin: 24px 0 0; border-left: 4px solid; padding-left: 16px; font-size: 1.05rem; font-weight: 500; line-height: 1.45; }
.av-action.none { border-color: var(--bloom); } .av-action.moderate { border-color: var(--saline); } .av-action.critical { border-color: var(--alarm); }
.av-warn { display: flex; gap: 10px; align-items: flex-start; background: rgba(224,164,48,0.2); color: #5f4108 !important;
           border-radius: 14px; padding: 12px 14px; margin-bottom: 20px; font-size: 0.9rem; font-weight: 500; line-height: 1.45; }
.av-warn .ic { flex-shrink: 0; margin-top: 1px; }
.av-sub { font-size: 0.72rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.18em; color: rgba(18,33,28,0.5) !important; margin: 32px 0 0; }
.av-alt { margin-top: 4px; }
.av-alt-row { display: flex; justify-content: space-between; gap: 12px; font-size: 0.9rem; margin-top: 10px; }
.av-alt-row b { font-variant-numeric: tabular-nums; font-weight: 600; }
.av-alt .av-track { height: 5px; margin-top: 5px; }
.av-alt .av-fill { background: rgba(18,33,28,0.45); }
.av-legend { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; margin: 12px 0 0 !important; padding: 0 !important; list-style: none; }
.av-legend li { display: flex; flex-direction: column; gap: 6px; border-radius: 14px; padding: 12px; font-size: 0.85rem; margin: 0;
                border: 2px solid transparent; list-style: none; }
.av-legend li span:last-child { color: rgba(18,33,28,0.62) !important; }
.av-legend .l-none { background: rgba(76,122,52,0.12); } .av-legend .l-none .ic { color: var(--bloom) !important; }
.av-legend .l-moderate { background: rgba(224,164,48,0.16); } .av-legend .l-moderate .ic { color: #9a6b0f !important; }
.av-legend .l-critical { background: rgba(179,39,30,0.1); } .av-legend .l-critical .ic { color: var(--alarm) !important; }
.av-legend .l-none.on { border-color: var(--bloom); } .av-legend .l-moderate.on { border-color: var(--saline); } .av-legend .l-critical.on { border-color: var(--alarm); }
.av-disc { font-size: 0.75rem; color: rgba(18,33,28,0.55) !important; margin: 16px 0 0; }
@media (max-width: 640px) { .av-card { padding: 24px; border-radius: 24px; } .av-figure { border-radius: 24px; } }

/* Empty state */
.av-empty { border: 1px solid var(--line); border-radius: 28px; padding: 40px 32px; display: flex; gap: 20px; align-items: center; color: var(--muted); }
.av-empty .ic { color: var(--saline); flex-shrink: 0; }
.av-empty b { color: var(--foam); font-family: var(--display); font-size: 1.25rem; font-weight: 600; letter-spacing: -0.02em; display: block; margin-bottom: 4px; }

/* Breakdown expander */
[data-testid="stExpander"] details { border: 1px solid var(--line) !important; border-radius: 20px !important; background: rgba(15,30,34,0.6); }
[data-testid="stExpander"] summary p { font-weight: 600; color: var(--foam); }

/* Grad-CAM */
.av-h2 { font-family: var(--display); font-weight: 600; letter-spacing: -0.035em; line-height: 0.98;
         font-size: clamp(2rem, 4.5vw, 3.5rem); margin: 18px 0 0; padding: 0; color: var(--foam); }
.st-key-gradcam [data-testid="stImage"] img { border-radius: 24px; background: #fff; }

/* Footer */
.av-footer { margin-top: 96px; padding-top: 28px; border-top: 1px solid var(--line); display: flex;
             justify-content: space-between; gap: 24px; flex-wrap: wrap; color: var(--muted); font-size: 0.85rem; line-height: 1.6; }
.av-footer b { color: var(--foam); }

@media (prefers-reduced-motion: reduce) {
    .av-scan { display: none; }
    .av-fill { animation: none; }
}
</style>
""", unsafe_allow_html=True)

# ================================
# NAV + HERO
# ================================
st.html(f"""
<nav class="av-nav">
    <div class="av-brand">{icon('scan', 26)} AquaVision <small>by AquaManage</small></div>
    <div class="av-chip"><span class="av-dot"></span>Models loaded · shrimp &amp; fish</div>
</nav>
<section class="av-hero">
    <p class="av-eyebrow">Disease check</p>
    <h1 class="av-display">Photo in.<br><span>Severity out.</span></h1>
    <p class="av-lede">Net a few shrimp or fish, take a photo in daylight, and get the likely disease
    and how serious it is — before it spreads to the next pond.</p>
    <div class="av-stats">
        <div class="av-stat"><div class="av-stat-v">12</div><div class="av-stat-l">Classes, shrimp and fish</div></div>
        <div class="av-stat"><div class="av-stat-v">85%</div><div class="av-stat-l">Shrimp model accuracy</div></div>
        <div class="av-stat"><div class="av-stat-v">82%</div><div class="av-stat-l">Fish model accuracy</div></div>
        <div class="av-stat"><div class="av-stat-v">4,000+</div><div class="av-stat-l">Training photos</div></div>
    </div>
</section>
""")

# ================================
# INPUT: species, sample, upload
# ================================
if 'sample' not in st.session_state:
    st.session_state.sample = None  # (path, label) of a chosen sample photo


def clear_sample():
    st.session_state.sample = None


def use_sample(path, label):
    st.session_state.sample = (path, label)


left, right = st.columns([1, 1.35], gap="large")

with left:
    st.html('<div class="av-step"><span class="av-step-n">01</span><span class="av-step-t">Pick the animal</span></div>')
    species = st.radio(
        "Species", ["Shrimp", "Fish"], horizontal=True, label_visibility="collapsed",
        format_func=lambda s: "🦐  Shrimp" if s == "Shrimp" else "🐟  Fish",
        key="species", on_change=clear_sample,
    )

    st.html('<div class="av-step" style="margin-top:36px"><span class="av-step-n">OR</span><span class="av-step-t">Try a sample</span></div>')
    with st.container(key="samples"):
        cols = st.columns(len(SAMPLES[species]))
        for col, (path, label) in zip(cols, SAMPLES[species]):
            with col:
                st.image(path, width="stretch")
                st.button(label, key=f"sample_{path}", on_click=use_sample, args=(path, label), width="stretch")

with right:
    st.html('<div class="av-step"><span class="av-step-n">02</span><span class="av-step-t">Add a photo</span></div>'
            '<p class="av-hint">One animal, well lit, filling most of the frame. JPG, PNG or WebP.</p>')
    uploaded_file = st.file_uploader(
        "Photo", type=['jpg', 'jpeg', 'png', 'webp'],
        label_visibility="collapsed", on_change=clear_sample,
    )
    st.html('<div class="av-tips"><span>☀️ Daylight, no flash</span><span>🔍 Close-up of gills, shell or skin</span><span>🖐️ One animal per photo</span></div>')

cfg = SPECIES[species]
session = cfg['load']()
class_names = cfg['classes']
disease_info = cfg['info']

if st.session_state.sample:
    sample_path, sample_label = st.session_state.sample
    image = Image.open(sample_path)
    source_label = f"Sample · {sample_label}"
elif uploaded_file is not None:
    image = Image.open(uploaded_file)
    source_label = "Your photo"
else:
    image = None

# ================================
# RESULT
# ================================
st.html('<div class="av-step" style="margin-top:72px"><span class="av-step-n">03</span><span class="av-step-t">Result</span></div>')

if image is None:
    st.html(f"""
<div class="av-empty">{icon('camera', 36)}
    <div><b>No photo yet</b>Add a photo above, or tap a sample, and the result shows up here.</div>
</div>""")
else:
    # Phone photos carry their rotation in EXIF; apply it so the model sees the animal upright.
    image = ImageOps.exif_transpose(image)

    with st.spinner("Checking photo…"):
        t0 = time.perf_counter()
        predicted_class, confidence, all_probs = predict(image, session, class_names, cfg['size'])
        elapsed_ms = (time.perf_counter() - t0) * 1000

    info = disease_info[predicted_class]
    severity = info['severity']
    sev = SEVERITY[severity]
    key = sev['key']

    if confidence >= 95:
        rel_label, rel_cls = "Very high", "high"
    elif confidence >= 80:
        rel_label, rel_cls = "High", "high"
    elif confidence >= 60:
        rel_label, rel_cls = "Medium", "medium"
    else:
        rel_label, rel_cls = "Low", "low"

    warn = ""
    if confidence < 60:
        warn = (f'<div class="av-warn">{icon("triangle", 18)}<span>The model is unsure about this photo. '
                'Take another in daylight, closer to the animal, and check again.</span></div>')

    order = np.argsort(all_probs)[::-1]
    alts = ""
    for idx in order[1:3]:
        p = float(all_probs[idx]) * 100
        alts += (f'<div class="av-alt-row"><span>{disease_info[class_names[idx]]["status"]}</span><b>{p:.1f}%</b></div>'
                 f'<div class="av-track"><div class="av-fill" style="width:{max(p, 0.5):.1f}%"></div></div>')

    bands = "".join(f'<div class="b-{s["key"]}{" on" if s["key"] == key else ""}"></div>' for s in SEVERITY.values())
    legend = "".join(
        f'<li class="l-{s["key"]}{" on" if s["key"] == key else ""}">{icon(SEVERITY_ICON[name], 20)}'
        f'<span style="font-weight:600">{s["label"]}</span><span>{s["band"]}</span></li>'
        for name, s in SEVERITY.items()
    )

    col_photo, col_card = st.columns([1.15, 1], gap="medium")

    with col_photo:
        st.html(f"""
<figure class="av-figure">
    <img src="{to_data_uri(image)}" alt="{source_label}">
    <div class="av-vignette"></div>
    <div class="av-scan"></div>
    <div class="av-corner tl"></div><div class="av-corner tr"></div>
    <div class="av-corner bl"></div><div class="av-corner br"></div>
    <figcaption class="av-figcap">{icon('camera', 16)} {source_label}</figcaption>
</figure>
<p class="av-meta">{species} model · {cfg['arch']} · checked in {elapsed_ms:.0f} ms</p>
""")

    with col_card:
        st.html(f"""
<div class="av-card">
    {warn}
    <div class="av-sev-row">
        <div class="av-sev {key}">{icon(SEVERITY_ICON[severity], 20)} {sev['label']}</div>
        <span class="av-rank {key}">{sev['rank']}</span>
    </div>
    <div class="av-bands">{bands}</div>

    <h2 class="av-title">{info['status']}</h2>
    <p class="av-desc">{info['description']}</p>

    <div class="av-conf">
        <span class="av-conf-l">Confidence <span class="av-rel {rel_cls}">{rel_label}</span></span>
        <span class="av-conf-v">{confidence:.1f}%</span>
    </div>
    <div class="av-track"><div class="av-fill" style="width:{confidence:.1f}%"></div></div>

    <p class="av-action {key}">{info['recommendation']}</p>

    <p class="av-sub">Could also be</p>
    <div class="av-alt">{alts}</div>

    <p class="av-sub">Three bands, always with a word next to the colour</p>
    <ul class="av-legend">{legend}</ul>
    <p class="av-disc">A screening tool, not a lab test. Confirm with your technician or a fisheries officer.</p>
</div>
""")

    # Full breakdown across every class
    with st.expander(f"See all {len(class_names)} classes"):
        asc = order[::-1]
        names = [disease_info[class_names[i]]['status'] for i in asc]
        probs = [float(all_probs[i]) * 100 for i in asc]
        colors = [sev['bar'] if i == order[0] else 'rgba(244,239,228,0.28)' for i in asc]

        fig = go.Figure(go.Bar(
            x=probs, y=names, orientation='h', marker_color=colors, marker_line_width=0,
            text=[f'{p:.1f}%' for p in probs], textposition='outside',
            textfont=dict(size=12, color='#f4efe4'),
            hovertemplate='%{y}: %{x:.1f}%<extra></extra>',
        ))
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family='Noto Sans, sans-serif', color='#f4efe4'),
            margin=dict(l=8, r=48, t=8, b=36), height=max(200, len(names) * 44),
            xaxis=dict(range=[0, 112], ticksuffix='%', gridcolor='rgba(244,239,228,0.08)', zeroline=False,
                       tickfont=dict(color='rgba(244,239,228,0.55)', size=11)),
            yaxis=dict(tickfont=dict(size=13, color='#f4efe4'), automargin=True, ticksuffix='  '), bargap=0.35, showlegend=False,
        )
        st.plotly_chart(fig, theme=None, config={'displayModeBar': False})

# ================================
# HOW THE MODEL LOOKS (Grad-CAM)
# ================================
if os.path.exists("assets/gradcam_all_classes.png"):
    st.html("<div style='height:88px'></div>")
    g_text, g_img = st.columns([1, 1.4], gap="large")
    with g_text:
        st.html("""
<p class="av-eyebrow">Under the hood</p>
<h2 class="av-h2">It shows where it looked.</h2>
<p class="av-lede" style="font-size:1rem">Grad-CAM heat maps mark the part of the photo that pushed the
model to its answer — gills for black gill, shell spots for WSSV. If the heat is on the background,
the photo needs retaking.</p>
<p class="av-lede" style="font-size:1rem">Two EfficientNet models, fine-tuned on about 2,000 photos each,
exported to ONNX so a check runs on a CPU in well under a second.</p>
""")
    with g_img:
        with st.container(key="gradcam"):
            st.image("assets/gradcam_all_classes.png", width="stretch")

# ================================
# FOOTER
# ================================
st.html("""
<footer class="av-footer">
    <div><b>AquaVision</b> · part of AquaManage, the pond notebook that reads itself.<br>
    For preliminary screening only — not a substitute for a veterinary diagnosis.</div>
    <div>Built by <b>Mahesh Penubothu</b> · VIT-AP University</div>
</footer>
""")
