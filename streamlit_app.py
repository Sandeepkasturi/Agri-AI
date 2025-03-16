import streamlit as st
import google.generativeai as genai
from PIL import Image
import io
import cv2
import numpy as np

# Set API Key
genai.configure(api_key="AIzaSyCzEjZj0VlVSO3L_bti6DhUrvq0dDDFYX8")

# Streamlit UI
st.set_page_config(layout="wide")  # Enable wide layout
st.title("Agriculture Bot: AI-Powered Plant Disease Analyzer")
st.write("Upload an image of a plant to diagnose diseases and get treatment recommendations.")

# Layout: Split into two columns
col1, col2 = st.columns(2)


# Webcam Capture Function
def capture_webcam():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    if ret:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cap.release()
        return Image.fromarray(frame)
    cap.release()
    return None


with col1:
    if st.button("Capture Image from Webcam"):
        captured_image = capture_webcam()
    else:
        captured_image = None

    # Upload Image Option
    uploaded_file = st.file_uploader("Or upload a plant image (leaf, stem, etc.)", type=["jpg", "jpeg", "png"])

    # Prompt Input
    prompt = st.text_area("Enter your query:", "Analyze this plant image for diseases and suggest treatments.")

    # Select Model
    gemini_models = ["gemini-1.5-flash-latest", "gemini-2.0-flash-lite", "gemini-2.0-flash", "gemini-2.0-pro-exp-02-05"]
    model_name = st.selectbox("Select AI Model:", gemini_models)

    # Process Image (either uploaded or captured)
    image = captured_image if captured_image else (Image.open(uploaded_file) if uploaded_file else None)
    analyze_button = st.button("Analyze Plant Image")

with col2:
    if image and analyze_button:
        # Resize image
        image.thumbnail([1024, 1024], Image.Resampling.LANCZOS)
        st.image(image, caption="Processed Plant Image", use_column_width=True)

        # Convert image to bytes
        img_byte_array = io.BytesIO()
        image.save(img_byte_array, format='PNG')
        img_byte_array = img_byte_array.getvalue()

        # Generate plant disease analysis using Gemini AI
        model = genai.GenerativeModel(model_name)
        response = model.generate_content([prompt, {"mime_type": "image/png", "data": img_byte_array}])

        # Display plant disease analysis dynamically
        st.subheader("AI Plant Disease Diagnosis:")
        st.write(response.text)
