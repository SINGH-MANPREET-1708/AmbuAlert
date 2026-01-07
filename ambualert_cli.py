import os
import cv2
from ultralytics import YOLO


# Path to the trained YOLOv8 model weights
MODEL_PATH = "ambualert_yolov8n.pt"

# Input image or video file for inference
SOURCE_INPUT = "demo_video.mp4"

# Supported file extensions
IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png"]
VIDEO_EXTENSIONS = [".mp4", ".mov"]


def process_image(model, image_path):
    """
    Runs YOLO inference on a single image and saves the annotated output.
    """
    print(f"Processing image: {image_path}")

    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Unable to read image file '{image_path}'")
        return

    # Perform object detection
    results = model(img)
    annotated_image = results[0].plot()

    # Save output image with '_annotated' suffix
    file_name, file_ext = os.path.splitext(image_path)
    output_path = f"{file_name}_annotated{file_ext}"
    cv2.imwrite(output_path, annotated_image)

    print(f"Image processing complete. Output saved as '{output_path}'")


def process_video(model, video_path):
    """
    Processes a video frame-by-frame, detects ambulances,
    and overlays an alert banner when detected.
    """
    print(f"Processing video: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Unable to open video file '{video_path}'")
        return

    # Extract video properties to preserve original quality
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30

    file_name, _ = os.path.splitext(video_path)
    output_path = f"{file_name}_annotated.mp4"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    print("Video inference in progress...")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Run YOLO inference on the current frame
        results = model(frame)
        annotated_frame = results[0].plot()
        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)

        # Extract detected class names
        detected_classes = [model.names[int(cls)] for cls in results[0].boxes.cls]

        # Display alert banner only if an ambulance is detected
        if any("ambulance" in cls.lower() for cls in detected_classes):
            overlay = annotated_frame.copy()
            cv2.rectangle(overlay, (0, 0), (frame_width, 80), (0, 0, 255), -1)
            annotated_frame = cv2.addWeighted(overlay, 0.5, annotated_frame, 0.5, 0)
            cv2.putText(
                annotated_frame,
                "GIVE WAY TO AMBULANCE",
                (40, 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.3,
                (255, 255, 255),
                3,
                cv2.LINE_AA,
            )

        out.write(annotated_frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"Video processing complete. Output saved as '{output_path}'")


def main():
    """
    Entry point for CLI-based inference.
    Automatically determines whether the input is an image or a video.
    """
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at '{MODEL_PATH}'")
        return

    if not os.path.exists(SOURCE_INPUT):
        print(f"Error: Input file not found at '{SOURCE_INPUT}'")
        return

    # Load YOLOv8 model
    try:
        model = YOLO(MODEL_PATH)
        print("YOLO model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    _, file_ext = os.path.splitext(SOURCE_INPUT)

    if file_ext.lower() in IMAGE_EXTENSIONS:
        process_image(model, SOURCE_INPUT)
    elif file_ext.lower() in VIDEO_EXTENSIONS:
        process_video(model, SOURCE_INPUT)
    else:
        print(f"Unsupported file type '{file_ext}'.")
        print(f"Supported images: {IMAGE_EXTENSIONS}")
        print(f"Supported videos: {VIDEO_EXTENSIONS}")


if __name__ == "__main__":
    main()
