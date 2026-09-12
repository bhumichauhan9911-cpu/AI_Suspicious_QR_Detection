
import streamlit as st
from PIL import Image
import cv2
import numpy as np
import pandas as pd
import joblib
from urllib.parse import urlparse
from pathlib import Path

st.set_page_config(page_title="AI Suspicious QR Detector", page_icon="QR", layout="centered")
MODEL_PATH = Path("models/qr_suspicious_model.joblib")

st.title("AI-Based Suspicious QR Code Detection")
st.caption("CNN-based image analysis + Random Forest ML + OpenCV QR decoding")

@st.cache_resource
def load_models():
    model = joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None
    try:
        from tensorflow.keras.models import load_model
        cnn = load_model("models/qr_cnn.keras")
    except Exception:
        cnn = None
    return model, cnn

def decode_qr(image):
    img = np.array(image.convert("RGB"))
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    data, _, _ = cv2.QRCodeDetector().detectAndDecode(bgr)
    return data.strip() if data else ""

def extract_features(text):
    lower = text.lower()
    p = urlparse(text)
    suspicious = ["verify","urgent","login","password","account","bank","payment",
                  "free","winner","prize","claim","update","confirm","gift","crypto","wallet"]
    shorteners = ["bit.ly","tinyurl.com","t.co","goo.gl","ow.ly","is.gd"]
    return pd.DataFrame([{
        "length": len(text),
        "digits": sum(c.isdigit() for c in text),
        "special_chars": sum(not c.isalnum() and not c.isspace() for c in text),
        "has_url": int(p.scheme in ["http","https"] and bool(p.netloc)),
        "https": int(p.scheme == "https"),
        "num_dots": text.count("."),
        "num_slashes": text.count("/"),
        "num_at": text.count("@"),
        "num_hyphen": text.count("-"),
        "num_equals": text.count("="),
        "num_question": text.count("?"),
        "suspicious_words": sum(w in lower for w in suspicious),
        "shortener": int(any(x in lower for x in shorteners)),
        "ip_like": int(bool(p.netloc) and p.netloc.replace(".","").isdigit())
    }])

def cnn_prediction(image, cnn):
    if cnn is None:
        return None
    img = image.convert("L").resize((64, 64))
    arr = np.array(img, dtype="float32") / 255.0
    arr = arr.reshape(1, 64, 64, 1)
    return float(cnn.predict(arr, verbose=0)[0][0])

model, cnn = load_models()
uploaded = st.file_uploader("Upload QR Code Image", type=["png","jpg","jpeg"])

if uploaded:
    image = Image.open(uploaded)
    st.image(image, caption="Uploaded QR Code", use_container_width=True)

    # DL image analysis
    if cnn is not None:
        dl_score = cnn_prediction(image, cnn)
        if dl_score >= 0.5:
            st.warning(f"DL (CNN) image score: {dl_score:.2%} suspicious")
        else:
            st.info(f"DL (CNN) image score: {dl_score:.2%} suspicious")
    else:
        st.warning("CNN model not found. Run: python train_dl_model.py")

    decoded = decode_qr(image)

    if not decoded:
        st.error("QR code could not be decoded. Try a clear QR image.")
    else:
        st.success("QR code decoded successfully.")
        st.subheader("Decoded Content")
        st.code(decoded)

        features = extract_features(decoded)

        if model:
            pred = int(model.predict(features)[0])
            ml_score = float(model.predict_proba(features)[0][1])
            if pred == 1:
                st.error("ML Result: Suspicious QR Code")
            else:
                st.success("ML Result: Likely Safe QR Code")
            st.write(f"ML suspicious probability: **{ml_score:.2%}**")
        else:
            ml_score = 0.5
            st.warning("ML model not found. Run: python train_model.py")

        if cnn is not None:
            final_score = (0.4 * dl_score) + (0.6 * ml_score)
            st.subheader("Final AI Result")
            st.write(f"Combined score: **{final_score:.2%}**")
            if final_score >= 0.5:
                st.error("FINAL RESULT: Suspicious QR Code")
            else:
                st.success("FINAL RESULT: Likely Safe QR Code")

        st.subheader("Extracted ML Features")
        st.dataframe(features.T.rename(columns={0:"Value"}))
else:
    st.info("Upload a QR-code image to start.")
