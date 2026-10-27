"""
SQLite engine/session for the dashboard read/audit database.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from sqlmodel import Session, SQLModel, create_engine

import config as cfg

# check_same_thread=False: the discovery/retraining pipelines run in
# background threads (see dashboard _run_discovery_bg / _run_retraining_bg),
# each needs its own Session but shares this one engine/connection pool.
engine = create_engine(
    cfg.DATABASE_URL,
    connect_args={"check_same_thread": False} if cfg.DATABASE_URL.startswith("sqlite") else {},
)


def init_db() -> None:
    """Create tables that don't exist yet. Safe to call on every startup."""
    SQLModel.metadata.create_all(engine)


@contextmanager
def get_session() -> Iterator[Session]:
    """Context-managed session: `with get_session() as session: ...`."""
    session = Session(engine)
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_session_dep() -> Iterator[Session]:
    """FastAPI dependency form: `session: Session = Depends(get_session_dep)`."""
    with get_session() as session:
        yield session
