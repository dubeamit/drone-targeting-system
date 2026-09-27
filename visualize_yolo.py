import cv2
import argparse
import random
from pathlib import Path
import matplotlib.pyplot as plt

def visualize_yolo(image_path, label_path, classes):
    if not image_path.exists():
        print(f"Image not found: {image_path}")
        return
        
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Failed to load image: {image_path}")
        return
        
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, _ = img.shape
    
    if label_path.exists():
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    x_center, y_center, width, height = map(float, parts[1:5])
                    
                    # Convert YOLO format to pixel coordinates
                    x_min = int((x_center - width / 2) * w)
                    y_min = int((y_center - height / 2) * h)
                    x_max = int((x_center + width / 2) * w)
                    y_max = int((y_center + height / 2) * h)
                    
                    class_name = classes.get(class_id, str(class_id))
                    color = (255, 0, 0) if class_id == 0 else (0, 255, 0)
                    
                    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, 2)
                    cv2.putText(img, class_name, (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    else:
        print(f"No label file found for {image_path.name}")

    plt.figure(figsize=(10, 8))
    plt.imshow(img)
    plt.axis('off')
    plt.title(f"Image: {image_path.name}")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize YOLO bounding boxes")
    parser.add_argument("--image", type=str, help="Path to specific image")
    parser.add_argument("--dir", type=str, default="drone_vs_bird/train/images", help="Directory of images to pick randomly from")
    args = parser.parse_args()
    
    classes = {0: "Drone", 1: "Bird"}
    
    if args.image:
        img_path = Path(args.image)
        label_path = img_path.parent.parent / "labels" / (img_path.stem + ".txt")
        visualize_yolo(img_path, label_path, classes)
    else:
        img_dir = Path(args.dir)
        images = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
        if not images:
            print(f"No images found in {img_dir}")
        else:
            random_img = random.choice(images)
            label_path = random_img.parent.parent / "labels" / (random_img.stem + ".txt")
            visualize_yolo(random_img, label_path, classes)
