from ultralytics import YOLO
import os

# --------------------------------------------------
# Configuration
# --------------------------------------------------

# Path to dataset configuration file (YOLO format)
# Update this path according to your dataset location
DATA_PATH = "path/to/data.yaml"

# Pretrained YOLOv8 model
MODEL_NAME = "yolov8n.pt"

# Training parameters
EPOCHS = 100
IMAGE_SIZE = 640
BATCH_SIZE = 16
PATIENCE = 10

# Output settings
PROJECT_NAME = "AmbuAlert_Training"
OUTPUT_DIR = "runs"


# --------------------------------------------------
# Training
# --------------------------------------------------

def train_model():
    """
    Trains a YOLOv8n model for ambulance detection
    using transfer learning.
    """
    print("Initializing YOLOv8 model...")
    model = YOLO(MODEL_NAME)

    print("Starting training...")
    results = model.train(
        data=DATA_PATH,
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        patience=PATIENCE,
        project=OUTPUT_DIR,
        name=PROJECT_NAME,
        save=True,
        verbose=True
    )

    print("Training completed.")
    print(f"Results saved in: {results.save_dir}")

    best_model_path = os.path.join(
        results.save_dir, "weights", "best.pt"
    )

    print(f"Best model saved at: {best_model_path}")
    return best_model_path


# --------------------------------------------------
# Evaluation
# --------------------------------------------------

def evaluate_model(best_model_path):
    """
    Evaluates the trained model on the test split.
    """
    print("Loading best model for evaluation...")
    model = YOLO(best_model_path)

    print("Running evaluation on test set...")
    model.val(
        data=DATA_PATH,
        imgsz=IMAGE_SIZE,
        split="test",
        plots=True
    )

    print("Evaluation complete.")


# --------------------------------------------------
# Entry Point
# --------------------------------------------------

if __name__ == "__main__":
    best_model = train_model()
    evaluate_model(best_model)
