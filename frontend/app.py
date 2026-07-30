import streamlit as st
from PIL import Image
import requests

st.set_page_config(
    page_title="Brain Disease Detection",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Brain Disease Detection")
st.write("Upload an MRI image to detect brain diseases using YOLOv11.")

uploaded_file = st.file_uploader(
    "Choose an MRI Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original MRI")
        st.image(image, use_container_width=True)

    if st.button("🔍 Detect Disease"):

        files = {
            "file": (
                uploaded_file.name,
                uploaded_file.getvalue(),
                uploaded_file.type
            )
        }

        with st.spinner("Detecting..."):

            response = requests.post(
                "http://127.0.0.1:8000/predict",
                files=files
            )

        if response.status_code == 200:

            result = response.json()

            disease = result["detections"][0]["class"]
            confidence = result["detections"][0]["confidence"]

            with col2:

                st.subheader("Prediction")

                prediction_url = (
                    "http://127.0.0.1:8000"
                    + result["prediction_image"]
                )

                st.image(
                    prediction_url,
                    use_container_width=True
                )

            st.success("Prediction Completed")

            st.metric(
                "Detected Disease",
                disease
            )

            st.progress(confidence)

            st.metric(
                "Confidence",
                f"{confidence*100:.2f}%"
            )

        else:
            st.error("Prediction Failed")