import os
import cv2
import tempfile
import streamlit as st
from ultralytics import YOLO


# --------------------------------------------------
# Streamlit page configuration
# --------------------------------------------------
st.set_page_config(
    page_title="AmbuAlert",
    layout="wide"
)

# Path to trained YOLOv8 model
MODEL_PATH = "ambualert_yolov8n.pt"


@st.cache_resource
def load_model():
    """
    Loads the YOLOv8 model once and caches it for reuse.
    This avoids reloading the model on every Streamlit interaction.
    """
    return YOLO(MODEL_PATH)


# Load model at app startup
model = load_model()


def draw_alert(frame):
    """
    Draws a red alert banner on the video frame
    when an ambulance is detected.
    """
    height, width, _ = frame.shape
    overlay = frame.copy()

    cv2.rectangle(overlay, (0, 0), (width, 80), (0, 0, 255), -1)
    frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)

    cv2.putText(
        frame,
        "GIVE WAY TO AMBULANCE",
        (40, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.3,
        (255, 255, 255),
        3,
        cv2.LINE_AA
    )

    return frame


def process_image(model, image_path):
    """
    Runs YOLO inference on a single image and
    returns the path to the annotated output image.
    """
    image = cv2.imread(image_path)
    if image is None:
        return None

    results = model(image)
    annotated_image = results[0].plot()
    annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)

    detected_classes = [model.names[int(cls)] for cls in results[0].boxes.cls]
    if any("ambulance" in cls.lower() for cls in detected_classes):
        annotated_image = draw_alert(annotated_image)

    output_path = image_path + "_annotated.jpg"
    cv2.imwrite(output_path, annotated_image)

    return output_path


def process_video(model, video_path):
    """
    Processes a video frame-by-frame and overlays
    an alert banner whenever an ambulance is detected.
    """
    cap = cv2.VideoCapture(video_path)

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    temp_output = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    out = cv2.VideoWriter(
        temp_output.name,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (frame_width, frame_height)
    )

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        results = model(frame)
        annotated_frame = results[0].plot()
        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)

        detected_classes = [model.names[int(cls)] for cls in results[0].boxes.cls]
        if any("ambulance" in cls.lower() for cls in detected_classes):
            annotated_frame = draw_alert(annotated_frame)

        out.write(annotated_frame)

    cap.release()
    out.release()

    return temp_output.name


# --------------------------------------------------
# Streamlit UI
# --------------------------------------------------
st.title("🚑 AmbuAlert — Smart Ambulance Detection System")

st.markdown(
    """
    Detect ambulances in images or videos using AI and
    display a visual alert urging drivers to give way.
    """
)

uploaded_file = st.file_uploader(
    "Upload an image or video",
    type=["jpg", "jpeg", "png", "mp4", "mov"]
)

if uploaded_file:
    # Save uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded_file.read())
        input_path = tmp.name

    file_ext = os.path.splitext(uploaded_file.name)[1].lower()

    if file_ext in [".jpg", ".jpeg", ".png"]:
        st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

        with st.spinner("Processing image..."):
            output_path = process_image(model, input_path)

        if output_path:
            st.success("Image processing complete.")
            st.image(
                output_path,
                caption="Processed Image with Detections",
                use_container_width=True
            )

            with open(output_path, "rb") as f:
                st.download_button(
                    "Download Processed Image",
                    f,
                    file_name="annotated_image.jpg"
                )

    elif file_ext in [".mp4", ".mov"]:
        st.info("Video uploaded successfully.")

        if st.button("Start Video Processing"):
            with st.spinner("Processing video... this may take a moment."):
                output_path = process_video(model, input_path)

            st.success("Video processing complete.")

            with open(output_path, "rb") as f:
                st.download_button(
                    "Download Processed Video",
                    f,
                    file_name="annotated_video.mp4"
                )

    else:
        st.error("Unsupported file type.")


# --------------------------------------------------
# Sidebar information
# --------------------------------------------------
st.sidebar.header("About AmbuAlert")

st.sidebar.markdown(
    """
    **AmbuAlert** is an AI-based system that detects ambulances
    in traffic footage and displays a visual alert urging drivers
    to give way.

    **Technologies used:**
    - YOLOv8 for object detection
    - OpenCV for visual overlays
    - Streamlit for web-based interface
    """
)

st.sidebar.markdown("---")
st.sidebar.info("Best performance is achieved on clear daytime traffic footage.")
