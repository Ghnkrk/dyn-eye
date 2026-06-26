import sys
from pathlib import Path
from ultralytics import YOLO
import glob

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg

def main():
    baseline_path = cfg.MODELS_DIR / "best_initial.pt"
    m_base = YOLO(str(baseline_path))
    
    image_paths = sorted(glob.glob(str(cfg.INPUT_IMAGES_DIR / "*.jpg")))
    print(f"Scanning {len(image_paths)} images with baseline at conf=0.25...")
    
    found = 0
    for p in image_paths:
        res = m_base.predict(p, conf=0.25, verbose=False)[0]
        if len(res.boxes) > 0:
            found += 1
            dets = [f"{res.names[int(b.cls[0])]}({float(b.conf[0]):.2f})" for b in res.boxes]
            print(f"{Path(p).name}: {dets}")
            
    print(f"Total images with baseline detections >= 0.25: {found}")

if __name__ == "__main__":
    main()
