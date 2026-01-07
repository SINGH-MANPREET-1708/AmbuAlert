#  AmbuAlert — Smart Ambulance Detection System

AmbuAlert is an AI-based computer vision system that detects ambulances in traffic images and videos and displays a visual alert urging drivers to **give way**.  
The system is built using the YOLOv8 object detection model and is demonstrated through both a **command-line interface (CLI)** and a **Streamlit web application**.

This project is intended for academic learning, student projects, and prototyping intelligent traffic monitoring solutions.


##  Key Features

- Real-time ambulance detection in images and videos  
- Visual alert overlay: **“GIVE WAY TO AMBULANCE”**  
- Streamlit-based web interface for easy interaction  
- Command-line inference for offline testing  
- YOLOv8 transfer learning for efficient training on limited data  



##  Model Overview

- **Model:** YOLOv8n (Nano version)  
- **Framework:** Ultralytics YOLO  
- **Approach:** Transfer learning from pretrained YOLOv8 weights  
- **Task:** Single-class object detection (Ambulance)

YOLOv8 was chosen for its balance between speed and accuracy, making it suitable for real-time traffic monitoring scenarios.



##  Dataset

- **Total images:** 2036  
  - Training: 1782 images  
  - Validation: 168 images  
  - Test: 86 images  

- **Annotation format:** YOLO  
- **Image size:** 640 × 640  

🔗 **Dataset link (Kaggle):**  
https://www.kaggle.com/datasets/manpreetsingh04/ambulance

> The dataset is derived from a publicly available source and organized into standard YOLO directory structure (`train`, `valid`, `test`).

## Project Structure
AmbuAlert/

├── ambualert_app.py    

├── ambualert_cli.py        

├── train_yolov8.py         

├── ambualert_yolov8n.pt    

├── demo_video.mp4         

├── requirements.txt

└── README.md


##  Clone the Repository

To clone this repository to your local machine, run:


git clone https://github.com/SINGH-MANPREET-1708/AmbuAlert.git4

cd AmbuAlert

## Install Dependencies

Ensure Python 3.8 or higher is installed, then run:

pip install -r requirements.txt

## Run the Streamlit Web Application 

streamlit run ambualert_app.py

-Upload an image or video

-Ambulances are detected automatically

-Download the annotated output


## Contact

Manpreet Singh

B.Tech CSE (AI & ML)

DAV Institute of Engineering & Technology, Jalandhar

Email: mrsingh31524@gmail.com

Phone: +91 62806-20692