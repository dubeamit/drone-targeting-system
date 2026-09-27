import os
from pathlib import Path

def main():
    base_dir = Path(__file__).resolve().parent
    incremental_dir = base_dir / 'incremental_dataset'
    merged_dir = base_dir / 'data' / 'merged_drone_dataset'
    
    splits = ['train', 'val', 'test']
    subdirs = ['images', 'labels']
    
    count = 0
    for subdir in subdirs:
        for split in splits:
            src_dir = incremental_dir / subdir / split
            dst_dir = merged_dir / subdir / split
            
            if not src_dir.exists():
                continue
                
            dst_dir.mkdir(parents=True, exist_ok=True)
            
            for file_path in src_dir.glob('*'):
                if not file_path.is_file():
                    continue
                    
                symlink_target = dst_dir / file_path.name
                if not symlink_target.exists():
                    try:
                        symlink_target.symlink_to(file_path.resolve())
                        count += 1
                    except FileExistsError:
                        pass
                        
    print(f"Successfully created {count} symlinks from incremental_dataset to merged_drone_dataset.")

if __name__ == '__main__':
    main()
