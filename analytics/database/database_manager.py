from sqlalchemy import func
import uuid

from .db import engine, SessionLocal, Base
from .models import (
    CustomerSession,
    OccupancyLog,
    TableStateHistory,
    WaiterServiceMetric
)

class DatabaseManager:
    def __init__(self):
        self.engine = engine
        self.SessionLocal = SessionLocal

    def initialize_db(self):
        Base.metadata.create_all(bind=self.engine)

    def log_occupancy(self, timestamp: float, table_id: str, occupancy_count: int, waiter_count: int, is_occupied: bool):
        db = self.SessionLocal()
        try:
            log = OccupancyLog(
                timestamp=timestamp,
                table_id=table_id,
                occupancy_count=occupancy_count,
                waiter_count=waiter_count,
                is_occupied=is_occupied
            )
            db.add(log)
            db.commit()
        finally:
            db.close()

    def create_customer_session(self, customer_track_id: int, table_id: str, entry_time: float) -> str:
        db = self.SessionLocal()
        try:
            session_uuid = str(uuid.uuid4())
            session = CustomerSession(
                session_uuid=session_uuid,
                customer_track_id=customer_track_id,
                table_id=table_id,
                entry_time=entry_time,
                seated_time=entry_time
            )
            db.add(session)
            db.commit()
            return session_uuid
        finally:
            db.close()

    def close_customer_session(self, session_uuid: str, exit_time: float):
        db = self.SessionLocal()
        try:
            session = db.query(CustomerSession).filter(CustomerSession.session_uuid == session_uuid).first()
            if session:
                session.exit_time = exit_time
                session.duration_seconds = exit_time - session.entry_time
                db.commit()
        finally:
            db.close()

    def adjust_customer_session_entry_time(self, session_uuid: str, absent_duration: float):
        db = self.SessionLocal()
        try:
            session = db.query(CustomerSession).filter(CustomerSession.session_uuid == session_uuid).first()
            if session:
                session.entry_time += absent_duration
                session.seated_time += absent_duration
                db.commit()
        finally:
            db.close()

    def log_table_state(self, table_id: str, new_state: str, start_time: float, previous_state: str = None, trigger: str = None, event_id: str = None, confidence: float = None, session_uuid: str = None):
        db = self.SessionLocal()
        try:
            prev_state = db.query(TableStateHistory).filter(
                TableStateHistory.table_id == table_id,
                TableStateHistory.end_time == None
            ).order_by(TableStateHistory.id.desc()).first()

            if prev_state:
                if prev_state.state == new_state:
                    return 
                prev_state.end_time = start_time
                prev_state.duration_seconds = start_time - prev_state.start_time
                
            new_log = TableStateHistory(
                table_id=table_id,
                state=new_state,
                start_time=start_time,
                previous_state=previous_state,
                trigger=trigger,
                event_id=event_id,
                confidence=confidence,
                session_uuid=session_uuid
            )
            db.add(new_log)
            db.commit()
        finally:
            db.close()

    def log_waiter_metric(self, table_id: str, customer_arrival_time: float, waiter_seen_time: float):
        db = self.SessionLocal()
        try:
            metric = WaiterServiceMetric(
                table_id=table_id,
                customer_arrival_time=customer_arrival_time,
                waiter_first_seen_time=waiter_seen_time,
                response_time_seconds=waiter_seen_time - customer_arrival_time
            )
            db.add(metric)
            
            session = db.query(CustomerSession).filter(
                CustomerSession.table_id == table_id,
                CustomerSession.exit_time == None
            ).order_by(CustomerSession.id.desc()).first()
            
            if session and not session.waiter_engaged:
                session.waiter_engaged = True
                session.waiter_response_time = waiter_seen_time - session.entry_time

            db.commit()
        finally:
            db.close()

    def update_waiter_food_served(self, table_id: str, food_arrival_time: float):
        db = self.SessionLocal()
        try:
            metric = db.query(WaiterServiceMetric).filter(
                WaiterServiceMetric.table_id == table_id,
                WaiterServiceMetric.food_arrival_time == None
            ).order_by(WaiterServiceMetric.id.desc()).first()
            if metric:
                metric.food_arrival_time = food_arrival_time
                metric.service_duration_seconds = food_arrival_time - metric.customer_arrival_time
                db.commit()
        finally:
            db.close()

    def get_table_analytics(self) -> dict:
        db = self.SessionLocal()
        try:
            stats = {}
            tables = db.query(CustomerSession.table_id).distinct().all()
            for (tid,) in tables:
                visits = db.query(func.count(CustomerSession.id)).filter(CustomerSession.table_id == tid).scalar()
                avg_duration = db.query(func.avg(CustomerSession.duration_seconds)).filter(CustomerSession.table_id == tid).scalar()
                stats[tid] = {
                    "total_visits": visits,
                    "avg_duration": round(avg_duration or 0, 1) if avg_duration else 0.0
                }
            return stats
        finally:
            db.close()

    def get_daily_summary(self) -> dict:
        db = self.SessionLocal()
        try:
            total_customers = db.query(func.count(CustomerSession.id)).scalar()
            avg_duration = db.query(func.avg(CustomerSession.duration_seconds)).scalar()
            avg_response = db.query(func.avg(WaiterServiceMetric.response_time_seconds)).scalar()
            
            return {
                "total_customers": total_customers or 0,
                "avg_customer_duration": round(avg_duration or 0, 1) if avg_duration else 0.0,
                "avg_waiter_response_time": round(avg_response or 0, 1) if avg_response else 0.0
            }
        finally:
            db.close()
