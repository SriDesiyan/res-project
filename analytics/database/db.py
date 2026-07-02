from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
import os
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.resolve()
DB_PATH = os.path.join(project_root, "restaurant_analytics.db")
DATABASE_URL = f"sqlite:///{DB_PATH}"

# check_same_thread is False for SQLite so we can use the DB across different threads if needed
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
