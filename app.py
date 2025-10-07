import streamlit as st
from ultralytics import YOLO
from PIL import Image

st.title("🚗 Car Detection Web App")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    model = YOLO("yolo11n.pt")
    results = model.predict(image)

    # Save detected image
    results[0].save(filename="detected.jpg")

    # Display detection result
    st.image("detected.jpg", caption="Detected Cars", use_container_width=True)

    st.success("✅ Car detection completed successfully!")
