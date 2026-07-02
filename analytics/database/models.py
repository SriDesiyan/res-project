from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime
from datetime import datetime
from .db import Base

class CustomerSession(Base):
    __tablename__ = "customer_sessions"

    id = Column(Integer, primary_key=True, index=True)
    session_uuid = Column(String, unique=True, index=True)
    customer_track_id = Column(Integer, index=True)
    table_id = Column(String, index=True)

    entry_time = Column(Float)
    seated_time = Column(Float)
    exit_time = Column(Float, nullable=True)

    duration_seconds = Column(Float, nullable=True)

    waiter_engaged = Column(Boolean, default=False)
    waiter_response_time = Column(Float, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)


class OccupancyLog(Base):
    __tablename__ = "occupancy_logs"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(Float, index=True)
    table_id = Column(String, index=True)

    occupancy_count = Column(Integer)
    waiter_count = Column(Integer)

    is_occupied = Column(Boolean)
    created_at = Column(DateTime, default=datetime.utcnow)


class TableStateHistory(Base):
    __tablename__ = "table_state_history"

    id = Column(Integer, primary_key=True, index=True)
    table_id = Column(String, index=True)

    state = Column(String)  # vacant, occupied, dirty, cleaning, clean
    
    # --- Extensions for deterministic FSM auditing ---
    previous_state = Column(String, nullable=True)
    trigger = Column(String, nullable=True)
    event_id = Column(String, nullable=True)
    confidence = Column(Float, nullable=True)
    session_uuid = Column(String, nullable=True)

    start_time = Column(Float)
    end_time = Column(Float, nullable=True)

    duration_seconds = Column(Float, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)


class WaiterServiceMetric(Base):
    __tablename__ = "waiter_service_metrics"

    id = Column(Integer, primary_key=True, index=True)
    table_id = Column(String, index=True)

    customer_arrival_time = Column(Float)
    waiter_first_seen_time = Column(Float)

    response_time_seconds = Column(Float)

    food_arrival_time = Column(Float, nullable=True)
    service_duration_seconds = Column(Float, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)


class DailySummary(Base):
    __tablename__ = "daily_summary"

    id = Column(Integer, primary_key=True, index=True)
    date = Column(String, index=True)  # YYYY-MM-DD

    total_customers = Column(Integer, default=0)
    peak_occupancy = Column(Integer, default=0)

    avg_customer_duration = Column(Float, default=0.0)
    avg_waiter_response_time = Column(Float, default=0.0)

    table_turnover_count = Column(Integer, default=0)

    created_at = Column(DateTime, default=datetime.utcnow)
