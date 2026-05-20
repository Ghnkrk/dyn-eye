"""
DYN-EYE — Entry Point

Quick-start commands:
    # Run discovery pipeline CLI
    python -m src.pipeline.graph --images-dir data/input_images

    # Run dashboard
    python -m dashboard.app

    # Run Label Studio ML backend
    python -m src.label_studio._wsgi

    # Setup FAISS index
    python main.py setup-faiss

    # Full pipeline
    python main.py discover
"""
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def main():
    parser = argparse.ArgumentParser(
        description="DYN-EYE — Unknown Defect Discovery Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  dashboard     Launch the monitoring dashboard (port 8501)
  discover      Run the full discovery pipeline
  retrain       Run the retraining pipeline
  setup-faiss   Build FAISS index from known defect crops
  ml-backend    Start the Label Studio ML backend (port 9090)
        """,
    )
    parser.add_argument(
        "command",
        choices=["dashboard", "discover", "retrain", "setup-faiss", "ml-backend"],
        help="Command to run",
    )
    parser.add_argument("--images-dir", default=None, help="Input images directory")
    parser.add_argument("--confidence", type=float, default=None)
    parser.add_argument("--model", default=None, help="YOLO model path")
    parser.add_argument("--project-id", type=int, default=None, help="Label Studio project ID")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--from-vlm-cache", action="store_true", help="Bypass YOLO/VLM steps and run from cached VLM output")

    args = parser.parse_args()

    if args.command == "dashboard":
        import uvicorn
        import config as cfg
        port = args.port or cfg.DASHBOARD_PORT
        print(f"[Dashboard] Starting DYN-EYE Dashboard on port {port}")
        uvicorn.run(
            "dashboard.app:app",
            host=cfg.DASHBOARD_HOST,
            port=port,
            reload=True,
        )

    elif args.command == "discover":
        from src.pipeline.graph import run_discovery_pipeline
        from src.features.known_defects_registry import get_known_defect_names
        known = get_known_defect_names()
        print(f"[Registry] Known defect classes: {known or '(none yet — all treated as unknown)'}")
        result = run_discovery_pipeline(
            input_images_dir=args.images_dir,
            confidence_threshold=args.confidence,
            yolo_model_path=args.model,
            from_vlm_cache=args.from_vlm_cache,
        )
        print(f"\n[OK] Pipeline complete. Run ID: {result.get('run_id')}")
        print(f"   Unknown images: {len(result.get('unknown_image_paths', []))}")
        print(f"   Crops: {len(result.get('crop_paths', []))}")
        print(f"   Clusters: {result.get('num_clusters', 0)}")

    elif args.command == "retrain":
        if not args.project_id:
            print("[ERROR] --project-id is required for retraining")
            sys.exit(1)
        from src.retraining.agent import run_retraining_pipeline
        result = run_retraining_pipeline(
            project_id=args.project_id,
            epochs=args.epochs,
        )
        training = result.get("training_result", {})
        deploy = result.get("deploy_result", {})
        sync = result.get("sync_result", {})
        print(f"\n[OK] Retraining complete.")
        print(f"   Training: {'SUCCESS' if training.get('success') else 'FAILED'}")
        print(f"   Deployment: {'SUCCESS' if deploy.get('success') else 'FAILED'}")
        if sync.get("success"):
            print(f"   Registry: +{len(sync.get('new_classes_added', []))} classes, {sync.get('faiss_vectors', 0)} FAISS vectors")
        elif sync.get("skipped"):
            print(f"   Registry: SKIPPED ({sync.get('reason', '')})")
        else:
            print(f"   Registry: FAILED ({sync.get('error', 'unknown')})")

    elif args.command == "setup-faiss":
        from src.features.faiss_index import FAISSIndexManager
        from src.features.known_defects_registry import register_defects
        import config as cfg
        crops_dir = args.images_dir or str(cfg.KNOWN_DEFECTS_DIR)
        manager = FAISSIndexManager()
        count = manager.setup(known_crops_dir=crops_dir)
        # Auto-register class names from subdirectory names
        from pathlib import Path
        class_names = [
            d.name for d in sorted(Path(crops_dir).iterdir()) if d.is_dir()
        ]
        if class_names:
            added = register_defects(class_names, source="faiss_setup")
            print(f"[Registry] Registered {len(added)} new defect classes: {added}")
        print(f"[OK] FAISS index built with {count} vectors")

    elif args.command == "ml-backend":
        from label_studio_ml.api import init_app
        from src.label_studio.model import VLMBackend
        import config as cfg
        port = args.port or cfg.LABEL_STUDIO_ML_BACKEND_PORT
        print(f"[ML Backend] Starting ML Backend on port {port}")
        # Dedicated model directory to avoid scanning workspace root
        ls_model_dir = cfg.MODELS_DIR / "label_studio"
        ls_model_dir.mkdir(parents=True, exist_ok=True)
        app = init_app(model_class=VLMBackend, model_dir=str(ls_model_dir))
        app.run(host="0.0.0.0", port=port, debug=True)


if __name__ == "__main__":
    main()
