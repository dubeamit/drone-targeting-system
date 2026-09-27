import os
import yaml
from pathlib import Path

def main():
    dataset_yaml = Path(__file__).resolve().parent / 'data' / 'merged_drone_dataset' / 'dataset.yaml'
    if not dataset_yaml.exists():
        print(f"Error: {dataset_yaml} not found.")
        return

    with open(dataset_yaml, 'r') as f:
        data = yaml.safe_load(f)

    classes = data.get('names', {})
    if isinstance(classes, list):
        classes = {i: name for i, name in enumerate(classes)}
        
    dataset_base = dataset_yaml.parent
    output_dir = dataset_base / 'class_reviews'
    
    # Create class directories
    for class_id, class_name in classes.items():
        (output_dir / class_name).mkdir(parents=True, exist_ok=True)
        
    splits = ['train', 'val', 'test']
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    
    links_created = 0
    missing_images = 0
    
    for split in splits:
        labels_dir = dataset_base / 'labels' / split
        images_dir = dataset_base / 'images' / split
        
        if not labels_dir.exists() or not images_dir.exists():
            continue
            
        for label_file in labels_dir.glob('*.txt'):
            # Find corresponding image
            image_path = None
            for ext in image_extensions:
                candidate = images_dir / (label_file.stem + ext)
                if candidate.exists():
                    image_path = candidate
                    break
                    
            if not image_path:
                missing_images += 1
                continue
                
            # Read label to find which classes are in this image
            present_classes = set()
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    class_id = int(parts[0])
                    present_classes.add(class_id)
                    
            # Create symlink for each class present
            for class_id in present_classes:
                class_name = classes.get(class_id)
                if not class_name:
                    continue
                    
                symlink_target = output_dir / class_name / image_path.name
                
                # Check if symlink already exists to avoid FileExistsError
                if not symlink_target.exists():
                    try:
                        symlink_target.symlink_to(image_path)
                        links_created += 1
                    except Exception as e:
                        print(f"Error creating symlink {symlink_target}: {e}")

    print(f"Successfully created {links_created} symlinks in {output_dir}")
    if missing_images > 0:
        print(f"Warning: {missing_images} label files had no corresponding image.")

if __name__ == '__main__':
    main()
