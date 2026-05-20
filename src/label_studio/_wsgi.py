"""
Label Studio ML Backend — WSGI Entry Point

Starts the ML backend server that Label Studio connects to
for VLM-based pre-annotations.
"""
import os
import sys
import argparse
import logging
import logging.config

# Configure logging
logging.config.dictConfig({
    "version": 1,
    "formatters": {
        "standard": {
            "format": "[%(asctime)s] [%(levelname)s] [%(name)s::%(funcName)s::%(lineno)d] %(message)s"
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "DEBUG",
            "stream": "ext://sys.stdout",
            "formatter": "standard"
        }
    },
    "root": {
        "level": "INFO",
        "handlers": ["console"],
        "propagate": True
    }
})

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from label_studio_ml.api import init_app
from src.label_studio.model import VLMBackend


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Label Studio VLM ML Backend")
    parser.add_argument("-p", "--port", dest="port", type=int, default=9090, help="Server port")
    parser.add_argument("--host", dest="host", type=str, default="0.0.0.0", help="Server host")
    parser.add_argument("-d", "--debug", dest="debug", action="store_true", help="Debug mode")
    args = parser.parse_args()

    app = init_app(
        model_class=VLMBackend,
        model_dir=os.environ.get("MODEL_DIR", os.path.dirname(__file__)),
        redis_queue=os.environ.get("RQ_QUEUE_NAME", "default"),
        redis_host=os.environ.get("REDIS_HOST", "localhost"),
        redis_port=os.environ.get("REDIS_PORT", 6379),
    )

    app.run(host=args.host, port=args.port, debug=args.debug)
else:
    # WSGI mode (for gunicorn)
    app = init_app(
        model_class=VLMBackend,
        model_dir=os.environ.get("MODEL_DIR", os.path.dirname(__file__)),
        redis_queue=os.environ.get("RQ_QUEUE_NAME", "default"),
        redis_host=os.environ.get("REDIS_HOST", "localhost"),
        redis_port=os.environ.get("REDIS_PORT", 6379),
    )
