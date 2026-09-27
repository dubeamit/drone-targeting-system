import os
from pathlib import Path

def polygon_to_bbox(coords):
    # coords is a list of [x1, y1, x2, y2, ...]
    if len(coords) < 4:
        return None
        
    x_coords = coords[0::2]
    y_coords = coords[1::2]
    
    x_min = min(x_coords)
    x_max = max(x_coords)
    y_min = min(y_coords)
    y_max = max(y_coords)
    
    # Constrain to 0-1
    x_min, x_max = max(0.0, x_min), min(1.0, x_max)
    y_min, y_max = max(0.0, y_min), min(1.0, y_max)
    
    width = x_max - x_min
    height = y_max - y_min
    x_center = x_min + width / 2
    y_center = y_min + height / 2
    
    return x_center, y_center, width, height

def main():
    dataset_dir = Path(__file__).resolve().parent / 'drone_vs_bird'
    splits = ['train', 'valid', 'test']
    
    fixed_count = 0
    
    for split in splits:
        labels_dir = dataset_dir / split / 'labels'
        if not labels_dir.exists():
            continue
            
        for label_file in labels_dir.glob('*.txt'):
            # Determine class based on filename
            filename = label_file.name
            if filename.startswith('B'):
                class_id = 1  # Bird
            elif filename.startswith('D'):
                class_id = 0  # Drone
            else:
                continue # Skip unknown patterns
                
            # Read polygon and convert
            new_lines = []
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) > 5:
                        # It's a polygon!
                        try:
                            coords = list(map(float, parts[1:]))
                            bbox = polygon_to_bbox(coords)
                            if bbox:
                                xc, yc, w, h = bbox
                                new_lines.append(f"{class_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")
                        except ValueError:
                            pass
                    elif len(parts) == 5:
                        # Already a bounding box, just fix the class ID
                        new_lines.append(f"{class_id} " + " ".join(parts[1:]) + "\n")
            
            # Write back fixed bounding boxes
            with open(label_file, 'w') as f:
                f.writelines(new_lines)
                
            fixed_count += 1
            
    print(f"Successfully fixed class IDs and converted polygons to bounding boxes for {fixed_count} files in drone_vs_bird.")

if __name__ == '__main__':
    main()
