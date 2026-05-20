import os
import shutil
import random
from pathlib import Path
from ultralytics import YOLO

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg

def main():
    images_dir = cfg.INPUT_IMAGES_DIR
    sample_dir = cfg.SAMPLE_RUN_DIR
    sample_dir.mkdir(parents=True, exist_ok=True)
    
    # Clear out sample_dir first
    for f in sample_dir.iterdir():
        if f.is_file():
            f.unlink()
    
    # We have 6 known classes now.
    from src.features.known_defects_registry import get_known_defect_names
    known_classes = set(get_known_defect_names())
    
    image_paths = [p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in {'.jpg', '.png'}]
    
    model = YOLO(str(cfg.YOLO_MODEL_PATH))
    
    known_images = []
    unknown_images = []
    
    for p in image_paths:
        res = model.predict(source=str(p), conf=0.25, verbose=False)[0]
        has_unknown = False
        has_known = False
        
        if res.boxes is not None and len(res.boxes) > 0:
            for box in res.boxes:
                cls_name = res.names[int(box.cls[0])]
                if cls_name in known_classes:
                    has_known = True
                else:
                    has_unknown = True
        
        if has_unknown:
            unknown_images.append(p)
        elif has_known:
            known_images.append(p)
            
    print(f"Total: {len(image_paths)}, Known: {len(known_images)}, Unknown: {len(unknown_images)}")
    
    random.shuffle(known_images)
    random.shuffle(unknown_images)
    
    # The user asked for 4 known and 16 unknown of 2 defect classes
    # But we might only have 9 unknowns (if that's all there is in the dataset)
    # We will grab as many unknowns as we can up to 16, and fill the rest with knowns up to 20
    selected_unknown = unknown_images[:16]
    needed_known = 20 - len(selected_unknown)
    selected_known = known_images[:needed_known]
    
    keep = selected_known + selected_unknown
    
    for p in keep:
        shutil.copy(str(p), str(sample_dir / p.name))
            
    print(f"Copied {len(selected_known)} known and {len(selected_unknown)} unknown images to {sample_dir.name}.")

if __name__ == '__main__':
    main()
