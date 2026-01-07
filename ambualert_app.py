import streamlit as st
import cv2
import os
import tempfile
from ultralytics import YOLO
import numpy as np


st.set_page_config(page_title="AmbuAlert 🚨", layout="wide")

MODEL_PATH = "best.pt"

# Load YOLO model
@st.cache_resource
def load_model():
    model = YOLO(MODEL_PATH)
    return model

model = load_model()

# ==========================
# Helper Functions
# ==========================
def draw_alert(frame):
    """Draw a red alert banner on the frame."""
    height, width, _ = frame.shape
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (width, 80), (0, 0, 255), -1)
    frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)
    cv2.putText(frame, " GIVE WAY TO AMBULANCE ",
                (40, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.3,
                (255, 255, 255), 3, cv2.LINE_AA)
    return frame


def process_image(model, image_path):
    img = cv2.imread(image_path)
    results = model(img)
    annotated = results[0].plot()
    annotated = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)

    detected_classes = [model.names[int(cls)] for cls in results[0].boxes.cls]
    if any("ambulance" in cls.lower() for cls in detected_classes):
        annotated = draw_alert(annotated)

    output_path = image_path.replace(".jpg", "_annotated.jpg")
    cv2.imwrite(output_path, annotated)
    return output_path


def process_video(model, video_path):
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30

    temp_output = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    out = cv2.VideoWriter(temp_output.name, cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame_width, frame_height))

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        results = model(frame)
        annotated = results[0].plot()
        annotated = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)

        detected_classes = [model.names[int(cls)] for cls in results[0].boxes.cls]
        if any("ambulance" in cls.lower() for cls in detected_classes):
            annotated = draw_alert(annotated)

        out.write(annotated)

    cap.release()
    out.release()
    return temp_output.name

st.title("🚑 AmbuAlert — Smart Ambulance Detection System")
st.markdown("""
Detect ambulances in images or videos using AI and alert drivers to **give way**.  
Upload any image or video below 👇
""")

uploaded_file = st.file_uploader("📁 Upload an image or video", type=["jpg", "jpeg", "png", "mp4", "mov"])

if uploaded_file:
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded_file.read())
        input_path = tmp.name

    # Detect file type
    file_ext = os.path.splitext(uploaded_file.name)[1].lower()

    if file_ext in [".jpg", ".jpeg", ".png"]:
        st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
        with st.spinner("🖼️ Processing image..."):
            output_path = process_image(model, input_path)
        st.success("✅ Image processing complete!")

        # Display processed image
        st.image(output_path, caption="Processed Image with Detections", use_container_width=True)

        # Download button
        with open(output_path, "rb") as f:
            st.download_button("⬇️ Download Processed Image", f, file_name="annotated_image.jpg")

    elif file_ext in [".mp4", ".mov"]:
        st.info("🎥 Uploaded video successfully. Click below to process.")
        process_button = st.button("🚀 Start Processing Video")

        if process_button:
            with st.spinner("🎬 Video is being processed... please wait..."):
                output_path = process_video(model, input_path)

            st.success("✅ Video processing complete!")
            st.markdown("### Download your processed video:")
            with open(output_path, "rb") as f:
                st.download_button("⬇️ Download Processed Video", f, file_name="annotated_video.mp4")

    else:
        st.error("❌ Unsupported file type!")

# Sidebar
st.sidebar.header("About AmbuAlert 🚨")
st.sidebar.markdown("""
**AmbuAlert** is an AI-based system that detects ambulances  
in traffic footage and displays alerts urging drivers to **give way**.

🔹 Built using:
- YOLOv8 Object Detection  
- OpenCV for visual overlays  
- Streamlit for web interface
""")

st.sidebar.markdown("---")
st.sidebar.info("💡 Tip: Works best on clear daytime traffic footage.")
