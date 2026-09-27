import os
import shutil
from pathlib import Path
import yaml

def main():
    base_dir = Path(__file__).resolve().parent
    main_dataset_yaml_path = base_dir / 'data' / 'merged_drone_dataset' / 'dataset.yaml'
    incremental_dir = base_dir / 'incremental_dataset'
    
    # 1. Read existing main dataset to get the current classes
    with open(main_dataset_yaml_path, 'r') as f:
        main_data = yaml.safe_load(f)
        
    main_classes = main_data.get('names', {})
    if isinstance(main_classes, list):
        main_classes = {i: name for i, name in enumerate(main_classes)}
        
    # We will add drone as 22 and bird as 23
    new_classes = main_classes.copy()
    drone_id = 22
    bird_id = 23
    new_classes[drone_id] = 'drone'
    new_classes[bird_id] = 'bird'
    
    # 2. Setup incremental dataset directory structure
    splits = ['train', 'val', 'valid', 'test']
    for split in ['train', 'val', 'test']:
        (incremental_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
        (incremental_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)
        
    # Mapping logic for each dataset
    datasets_to_process = [
        {
            'name': 'drone_vs_bird',
            'path': base_dir / 'drone_vs_bird',
            'id_map': {0: drone_id, 1: bird_id} # 0 is Drone, 1 is Bird in this dataset
        },
        {
            'name': 'roboflow_drone',
            'path': base_dir / 'roboflow_drone',
            'id_map': {0: drone_id} # 0 is drone in this dataset
        }
    ]
    
    total_images_copied = 0
    total_labels_copied = 0
    
    for dataset in datasets_to_process:
        d_path = dataset['path']
        d_name = dataset['name']
        id_map = dataset['id_map']
        
        for split in splits:
            images_dir = d_path / split / 'images'
            labels_dir = d_path / split / 'labels'
            
            # Map 'valid' from roboflow/yolo to 'val' standard
            target_split = 'val' if split == 'valid' else split
            
            if not images_dir.exists():
                images_dir = d_path / 'images' / split # Alternative structure
                labels_dir = d_path / 'labels' / split
                if not images_dir.exists():
                    continue
            
            for image_file in images_dir.glob('*'):
                if not image_file.is_file():
                    continue
                    
                # Define new unique filename
                new_stem = f"{d_name}_{image_file.stem}"
                new_image_name = new_stem + image_file.suffix
                new_image_path = incremental_dir / 'images' / target_split / new_image_name
                
                # Copy image
                shutil.copy2(image_file, new_image_path)
                total_images_copied += 1
                
                # Process label
                label_file = labels_dir / (image_file.stem + '.txt')
                new_label_path = incremental_dir / 'labels' / target_split / (new_stem + '.txt')
                
                if label_file.exists():
                    with open(label_file, 'r') as f_in, open(new_label_path, 'w') as f_out:
                        for line in f_in:
                            parts = line.strip().split()
                            if not parts:
                                continue
                            original_id = int(parts[0])
                            
                            if original_id in id_map:
                                new_id = id_map[original_id]
                                new_line = f"{new_id} {' '.join(parts[1:])}\n"
                                f_out.write(new_line)
                    total_labels_copied += 1
                else:
                    # Create empty label file for negative samples
                    new_label_path.touch()
                    
    print(f"Copied {total_images_copied} images and {total_labels_copied} label files to {incremental_dir}")
    
    # 3. Write updated dataset.yaml for the incremental dataset
    incremental_yaml = {
        'path': './',
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'nc': len(new_classes),
        'names': new_classes
    }
    
    with open(incremental_dir / 'dataset.yaml', 'w') as f:
        yaml.dump(incremental_yaml, f, sort_keys=False)
        
    print(f"Created incremental dataset.yaml at {incremental_dir / 'dataset.yaml'}")

if __name__ == '__main__':
    main()
