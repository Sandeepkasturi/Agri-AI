import streamlit as st
import google.generativeai as genai
from PIL import Image
import io

# Set API Key
genai.configure(api_key=st.secrets["api_key"])

# Streamlit UI
st.set_page_config(layout="wide")  # Enable wide layout
st.title("Agriculture Bot: AI-Powered Plant Disease Analyzer")
st.write("Upload an image of a plant to diagnose diseases and get treatment recommendations.")

# Layout: Split into two columns
col1, col2 = st.columns(2)

# Replace Webcam Capture Function (Now using the webcam through streamlit)
def capture_webcam():
    # Streamlit allows direct webcam capture
    image = st.camera_input("Capture Image from Webcam")
    if image:
        return Image.open(image)
    return None

with col1:
    # Capture image button using the webcam
    captured_image = capture_webcam() if st.button("Capture Image from Webcam") else None

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
