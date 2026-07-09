"""SQLAlchemy engine + session factory.

SQLite is used by default (zero external setup, durable across restarts,
good enough for an assignment-scale service). Swapping to Postgres in
production is a one-line change to DATABASE_URL since we only use the
ORM layer and no SQLite-specific SQL.
"""
from __future__ import annotations

import os
from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker


class Base(DeclarativeBase):
    pass


def make_engine(database_url: str):
    connect_args = {"check_same_thread": False} if database_url.startswith("sqlite") else {}
    if database_url.startswith("sqlite:///"):
        db_path = database_url.replace("sqlite:///", "")
        if db_path not in (":memory:",) and os.path.dirname(db_path):
            os.makedirs(os.path.dirname(db_path), exist_ok=True)
    return create_engine(database_url, connect_args=connect_args, future=True)


_engine = None
_SessionLocal: sessionmaker | None = None


def init_db(database_url: str) -> None:
    global _engine, _SessionLocal
    _engine = make_engine(database_url)
    _SessionLocal = sessionmaker(bind=_engine, autoflush=False, autocommit=False, future=True)
    Base.metadata.create_all(bind=_engine)


def get_session() -> Generator[Session, None, None]:
    if _SessionLocal is None:
        raise RuntimeError("Database not initialized. Call init_db() first.")
    session = _SessionLocal()
    try:
        yield session
    finally:
        session.close()
