import logging
import time
from typing import Generator
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from app.config import get_settings
from app.db.models import Base

logger = logging.getLogger(__name__)

_engine = None
_SessionLocal = None


def get_engine():
    global _engine
    if _engine is None:
        settings = get_settings()
        db_url = settings.DATABASE_URL

        # SQLite vs PostgreSQL connection options
        connect_args = {}
        if db_url.startswith("sqlite"):
            connect_args["check_same_thread"] = False
            _engine = create_engine(db_url, connect_args=connect_args)
        else:
            _engine = create_engine(
                db_url,
                pool_pre_ping=True,
                pool_size=10,
                max_overflow=20,
            )
    return _engine


def get_session_local():
    global _SessionLocal
    if _SessionLocal is None:
        engine = get_engine()
        _SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return _SessionLocal


def init_db(max_retries: int = 10, retry_delay: float = 2.0):
    """
    Initializes database tables with retry logic to ensure connection readiness in Docker.
    """
    engine = get_engine()
    attempt = 0
    while attempt < max_retries:
        try:
            logger.info(f"Connecting to database (attempt {attempt + 1}/{max_retries})...")
            Base.metadata.create_all(bind=engine)
            logger.info("Database tables initialized successfully.")
            return
        except Exception as e:
            attempt += 1
            logger.warning(
                f"Database connection attempt {attempt} failed: {e}. "
                f"Retrying in {retry_delay} seconds..."
            )
            if attempt >= max_retries:
                logger.error(f"Failed to connect to database after {max_retries} attempts.")
                raise e
            time.sleep(retry_delay)


def get_db() -> Generator[Session, None, None]:
    """Dependency that yields a database session and safely closes it."""
    session_factory = get_session_local()
    db = session_factory()
    try:
        yield db
    finally:
        db.close()
