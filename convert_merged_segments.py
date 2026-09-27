import os
from pathlib import Path

def polygon_to_bbox(coords):
    if len(coords) < 4:
        return None
        
    x_coords = coords[0::2]
    y_coords = coords[1::2]
    
    x_min = min(x_coords)
    x_max = max(x_coords)
    y_min = min(y_coords)
    y_max = max(y_coords)
    
    # Constrain to 0-1 just in case
    x_min, x_max = max(0.0, x_min), min(1.0, x_max)
    y_min, y_max = max(0.0, y_min), min(1.0, y_max)
    
    width = x_max - x_min
    height = y_max - y_min
    x_center = x_min + width / 2
    y_center = y_min + height / 2
    
    return x_center, y_center, width, height

def main():
    merged_labels_dir = Path(__file__).resolve().parent / 'data' / 'merged_drone_dataset' / 'labels'
    splits = ['train', 'val', 'test']
    
    files_fixed = 0
    total_segments_converted = 0
    
    for split in splits:
        split_dir = merged_labels_dir / split
        if not split_dir.exists():
            continue
            
        for label_file in split_dir.glob('*.txt'):
            needs_update = False
            new_lines = []
            
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts:
                        continue
                        
                    if len(parts) > 5:
                        # Convert segment polygon to bounding box
                        needs_update = True
                        class_id = int(parts[0])
                        try:
                            coords = list(map(float, parts[1:]))
                            bbox = polygon_to_bbox(coords)
                            if bbox:
                                xc, yc, w, h = bbox
                                new_lines.append(f"{class_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")
                                total_segments_converted += 1
                        except ValueError:
                            pass
                    else:
                        # Standard bounding box, keep as is
                        new_lines.append(line)
            
            if needs_update:
                with open(label_file, 'w') as f:
                    f.writelines(new_lines)
                files_fixed += 1
                
    print(f"Successfully converted {total_segments_converted} segment polygons across {files_fixed} files to standard bounding boxes!")

if __name__ == '__main__':
    main()
