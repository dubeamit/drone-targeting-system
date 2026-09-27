import os
import yaml
from collections import defaultdict
from pathlib import Path

def main():
    base_dir = Path(__file__).resolve().parent
    dataset_yaml = base_dir / 'data' / 'merged_drone_dataset' / 'dataset.yaml'
    if not dataset_yaml.exists():
        dataset_yaml = base_dir / 'drone_vs_bird' / 'data.yaml'
    if not dataset_yaml.exists():
        print(f"Error: dataset YAML not found.")
        return

    with open(dataset_yaml, 'r') as f:
        data = yaml.safe_load(f)

    classes = data.get('names', {})
    if isinstance(classes, list):
        classes = {i: name for i, name in enumerate(classes)}
        
    dataset_base = dataset_yaml.parent
    
    # Check train, val, test directories
    splits = ['train', 'val', 'test']
    
    class_counts = defaultdict(int)
    total_instances = 0
    
    for split in splits:
        labels_dir = dataset_base / 'labels' / split
        if not labels_dir.exists():
            continue
            
        for label_file in labels_dir.glob('*.txt'):
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    class_id = int(parts[0])
                    class_counts[class_id] += 1
                    total_instances += 1
                    
    print(f"{'Class ID':<10} | {'Class Name':<30} | {'Count':<10} | {'Percentage':<10}")
    print("-" * 68)
    
    # Print sorted by count (descending)
    sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
    
    for class_id, count in sorted_classes:
        class_name = classes.get(class_id, f"Unknown ({class_id})")
        percentage = (count / total_instances * 100) if total_instances > 0 else 0
        print(f"{class_id:<10} | {class_name:<30} | {count:<10} | {percentage:.2f}%")
        
    print("-" * 68)
    print(f"{'Total':<10} | {'':<30} | {total_instances:<10} | 100.00%")
    
    # Also list classes with 0 instances
    zero_count_classes = []
    for class_id, class_name in classes.items():
        if class_id not in class_counts:
            zero_count_classes.append((class_id, class_name))
            
    if zero_count_classes:
        print("\nClasses with 0 instances:")
        for class_id, class_name in zero_count_classes:
            print(f"{class_id:<10} | {class_name:<30} | 0          | 0.00%")

if __name__ == '__main__':
    main()
