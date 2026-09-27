from pathlib import Path
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

# Download the YOLOv9e VisDrone weights
# weights = hf_hub_download(
#     repo_id="dronefreak/visdrone-yolov9e",
#     filename="best.pt"
# )

weights = "weights/visdrone-yolov9c.pt"

# Load the model
model = YOLO(weights) 

current_dir = Path(__file__).resolve().parent
video_path = current_dir / "videos" / "drone_video.mp4"
if not video_path.exists():
    video_path = current_dir / "drone_video.mp4"

# Run inference on your specific video
results = model.predict(
    source=str(video_path),
    conf=0.25,     # Only show detections with confidence > 25%
    imgsz=1280,    # Use 1280 resolution as default in VisDrone inference script
    save=True,     # Save the output video with bounding boxes drawn
    show=True,     # Display the video with bounding boxes live
    device="cuda"  # Use GPU for faster inference
)

print("Inference finished. Check the 'runs/detect/predict' folder for your output video!")
