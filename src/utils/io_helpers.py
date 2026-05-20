"""
Shared I/O helper utilities.
"""
import json
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def load_json(path: str | Path) -> dict | list:
    """Load and parse a JSON file."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(data: dict | list, path: str | Path) -> None:
    """Save data as pretty-printed JSON."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


def list_images(directory: str | Path) -> list[Path]:
    """List all image files in a directory (non-recursive)."""
    d = Path(directory)
    if not d.is_dir():
        return []
    return sorted(
        f for f in d.iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    )
