import os
import shutil
from pathlib import Path

# Define paths
base_dir = Path(__file__).resolve().parent
out_dir = base_dir / "data" / "merged_drone_dataset"

# Global classes
global_classes = [
    "person", "soldier", "bicycle", "motorcycle", "car", "van", "truck", "bus", "tricycle",
    "tank", "military_vehicle", "infantry_fighting_vehicle", "armored_personnel_carrier",
    "engineering_vehicle", "assault_helicopter", "transport_helicopter", "assault_airplane",
    "transport_airplane", "anti_aircraft_vehicle", "towed_artillery", "self_propelled_artillery",
    "others"
]

# Create output directories
for split in ["train", "val", "test"]:
    (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
    (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

# Helper function to process and copy files
def process_dataset(img_dir, lbl_dir, split, class_map, prefix=""):
    if not img_dir.exists() or not lbl_dir.exists():
        print(f"Skipping {img_dir} - Not found")
        return

    images = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")) + list(img_dir.glob("*.jpeg"))
    for img_path in images:
        lbl_path = lbl_dir / (img_path.stem + ".txt")
        new_img_name = f"{prefix}_{img_path.name}"
        new_lbl_name = f"{prefix}_{img_path.stem}.txt"
        
        out_img_path = out_dir / "images" / split / new_img_name
        out_lbl_path = out_dir / "labels" / split / new_lbl_name
        
        # Copy image
        shutil.copy(img_path, out_img_path)
        
        # Process label if exists
        if lbl_path.exists():
            with open(lbl_path, "r") as f_in, open(out_lbl_path, "w") as f_out:
                for line in f_in:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    old_class = int(float(parts[0]))
                    if old_class in class_map:
                        new_class = class_map[old_class]
                        f_out.write(f"{new_class} {' '.join(parts[1:])}\n")

print("Processing VisDrone...")
visdrone_map = {0: 0, 1: 0, 2: 2, 3: 4, 4: 5, 5: 6, 6: 8, 7: 8, 8: 7, 9: 3, 10: 21}
vis_base = base_dir / "new_data" / "visdrone_yolo"
for split in ["train", "val", "test"]:
    process_dataset(vis_base / "images" / split, vis_base / "labels" / split, split, visdrone_map, "visdrone")

print("Processing AMAD-5...")
amad_map = {0: 9, 1: 10, 2: 0, 3: 1, 4: 4}
amad_base = base_dir / "new_data" / "AMAD-5"
for split in ["train", "val", "test"]:
    process_dataset(amad_base / split / "images", amad_base / split / "labels", split, amad_map, "amad")

print("Processing Military Equipment...")
mileq_map = {0: 9, 1: 11, 2: 12, 3: 13, 4: 14, 5: 15, 6: 16, 7: 17, 8: 18, 9: 19, 10: 20}
mileq_base = base_dir / "new_data" / "dataset_military_equipment"
# train
process_dataset(mileq_base / "images_train", mileq_base / "labels_train", "train", mileq_map, "mileq")
# val (labels_test maps to images_val? Let's check if there are images_test)
if (mileq_base / "labels_val").exists():
    process_dataset(mileq_base / "images_val", mileq_base / "labels_val", "val", mileq_map, "mileq")
elif (mileq_base / "labels_test").exists():
    process_dataset(mileq_base / "images_val", mileq_base / "labels_test", "val", mileq_map, "mileq")

print("Processing UAV-Tank-Dataset...")
uav_map = {0: 9}
uav_base = base_dir / "new_data" / "UAV-Tank-Dataset" / "Drone & Aerial View: Battle Tank Detection Dataset"
# Doesn't have explicit split, putting all in train
process_dataset(uav_base / "images", uav_base / "labels", "train", uav_map, "uavtank")

# Create yaml
yaml_content = f"""path: {out_dir.resolve()}
train: images/train
val: images/val
test: images/test

nc: {len(global_classes)}
names:
"""
for idx, name in enumerate(global_classes):
    yaml_content += f"  {idx}: {name}\n"

with open(out_dir / "dataset.yaml", "w") as f:
    f.write(yaml_content)

print("Dataset merging complete! Saved to:", out_dir)
