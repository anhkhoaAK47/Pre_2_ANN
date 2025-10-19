# app/db.py
# SQLAlchemy session/engine bootstrap cho MySQL (hoặc bất kỳ SQLAlchemy URL nào trong .env)

import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "").strip()
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL is not set. Example: mysql+pymysql://user:pass@localhost:3306/triage_db")

# Configure engine based on database type
if DATABASE_URL.startswith("sqlite"):
    # SQLite specific configuration
    engine = create_engine(DATABASE_URL, future=True)
else:
    # MySQL/PostgreSQL configuration with pool_pre_ping
    engine = create_engine(DATABASE_URL, pool_pre_ping=True, future=True)

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
Base = declarative_base()

# Dependency dùng với FastAPI
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
