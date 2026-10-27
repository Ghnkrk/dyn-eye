"""
Read/audit database for the DYN-EYE dashboard.

The pipeline and retraining code (src/pipeline, src/retraining,
src/features) remain the source of truth, reading and writing the
existing JSON files under data/ exactly as before. This package mirrors
that JSON state into SQLite so the API/dashboard can query, filter, and
paginate it instead of re-parsing files on every request.

See src/db/sync.py for the JSON -> DB mirroring functions.
"""
from src.db.session import get_session, init_db

__all__ = ["get_session", "init_db"]
