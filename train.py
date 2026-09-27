from ultralytics import YOLO

def main():
    # Load the YOLOv11x pre-trained model for best accuracy
    model = YOLO("yolo11x.pt")

    # Train the model
    # data="VisDrone.yaml" will automatically download the VisDrone dataset
    # epochs=50 runs training for 50 epochs
    # imgsz=640 sets the image size
    # classes=[0, 1, 3, 4, 5, 8, 9] filters to detect specific classes (humans and vehicles)
    print("Starting training YOLOv11x on VisDrone dataset...")
    results = model.train(
        data="VisDrone.yaml",
        epochs=50,
        imgsz=1280,
        classes=[0, 1, 3, 4, 5, 8, 9]
    )
    print("Training completed successfully.")

if __name__ == "__main__":
    main()
