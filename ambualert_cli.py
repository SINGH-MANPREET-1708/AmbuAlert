import cv2
from ultralytics import YOLO
import os

MODEL_PATH = 'best.pt'

SOURCE_INPUT = 'test.mp4'


IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png']
VIDEO_EXTENSIONS = ['.mp4','.mov']


def process_image(model, image_path):
    
    print(f"Processing image: {image_path}")
    try:
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Could not read image file '{image_path}'")
            return
        results = model(img)
        annotated_image = results[0].plot()

        # Create the output filename
        file_name, file_ext = os.path.splitext(image_path)
        output_path = f"{file_name}_annotated{file_ext}"

        # Save the annotated image
        cv2.imwrite(output_path, annotated_image)
        print(f"✅ Image processing complete! Saved as '{output_path}'")

    except Exception as e:
        print(f"An error occurred during image processing: {e}")

def process_video(model, video_path):
    print(f"Processing video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file '{video_path}'")
        return

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps == 0:
        fps = 30  # fallback

    # Define the output filename and VideoWriter object
    file_name, _ = os.path.splitext(video_path)
    output_path = f"{file_name}_annotated.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    print("This may take a moment...")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Run YOLO model
        results = model(frame)

        # Get annotated frame
        annotated_frame = results[0].plot()
        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)

        # ✅ Case-insensitive ambulance detection
        detected_classes = [model.names[int(cls)] for cls in results[0].boxes.cls]
        if any("ambulance" in cls.lower() for cls in detected_classes):
            # Draw alert
            overlay = annotated_frame.copy()
            cv2.rectangle(overlay, (0, 0), (frame_width, 80), (0, 0, 255), -1)
            annotated_frame = cv2.addWeighted(overlay, 0.5, annotated_frame, 0.5, 0)
            cv2.putText(annotated_frame, "--- GIVE WAY TO AMBULANCE ---",
                        (40, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.3,
                        (255, 255, 255), 3, cv2.LINE_AA)

        out.write(annotated_frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"✅ Video processing complete! Saved as '{output_path}'")



def main():
   
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at '{MODEL_PATH}'")
        return

    if not os.path.exists(SOURCE_INPUT):
        print(f"Error: Source file not found at '{SOURCE_INPUT}'")
        return

    try:
        model = YOLO(MODEL_PATH)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Determine file type and process accordingly
    _, file_ext = os.path.splitext(SOURCE_INPUT)

    if file_ext.lower() in IMAGE_EXTENSIONS:
        process_image(model, SOURCE_INPUT)
    elif file_ext.lower() in VIDEO_EXTENSIONS:
        process_video(model, SOURCE_INPUT)
    else:
        print(f"Error: Unsupported file type '{file_ext}'.")
        print(f"Please use one of the supported formats:")
        print(f"Images: {IMAGE_EXTENSIONS}")
        print(f"Videos: {VIDEO_EXTENSIONS}")


if __name__ == "__main__":
    main()

