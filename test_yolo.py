import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg

def test_graph():
    from src.pipeline.graph import run_discovery_pipeline
    
    print("Calling run_discovery_pipeline...")
    res = run_discovery_pipeline(
        input_images_dir=str(cfg.SAMPLE_RUN_DIR),
        confidence_threshold=cfg.YOLO_CONFIDENCE_THRESHOLD,
    )
    print("Graph returned:", list(res.keys()))

if __name__ == '__main__':
    test_graph()
