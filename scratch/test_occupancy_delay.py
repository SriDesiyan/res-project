import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "analytics"))

from occupancy.occupancy_engine import OccupancyEngine

class MockPerson:
    def __init__(self, track_id, bbox, role="customer", confirmed=True):
        self.track_id = track_id
        self.bbox = bbox
        self.role = role
        self.confirmed = confirmed

def main():
    tables = {
        "table_1": {
            "polygon": [[0, 0], [100, 0], [100, 100], [0, 100]],
            "center": [50, 50]
        }
    }
    
    engine = OccupancyEngine(["table_1"], tables)
    
    # We will simulate a customer sitting at table_1 (bottom-center anchor is in the box)
    # The bottom center anchor is (50, 90).
    person = MockPerson(track_id=1, bbox=[10, 10, 90, 90])
    
    # DWELL_THRESHOLD = 75
    # TEMPORAL_VOTE_FRAMES = 10
    # OCCUPANCY_DELAY_SECONDS = 5.0
    # Let's run at 25 fps
    fps = 25.0
    
    # Run for 200 frames (8 seconds)
    print("Simulating customer occupancy:")
    for frame in range(1, 250):
        frame_time = frame / fps
        engine.update([person], frame_time)
        status = engine.get_table_status("table_1")
        all_status = engine.get_all_status(frame_time)[0]
        
        # Log transition points
        if frame in (1, 10, 75, 80, 85, 90, 100, 200, 210, 220, 230, 240):
            print(f"Frame {frame:3d} ({frame_time:.2f}s) | "
                  f"verified: {len(engine.tables['table_1'].current_customers)} | "
                  f"disp_cnt: {status['customer_count']} | "
                  f"is_occupied: {all_status['is_occupied']} | "
                  f"occ_start: {engine.tables['table_1'].occupied_start_time} | "
                  f"occ_sec: {all_status['occupied_seconds']}")

    print("\nSimulating customer leaving:")
    # Run for 20 frames without any person
    for frame in range(250, 270):
        frame_time = frame / fps
        engine.update([], frame_time)
        status = engine.get_table_status("table_1")
        all_status = engine.get_all_status(frame_time)[0]
        if frame in (250, 255, 260, 265):
            print(f"Frame {frame:3d} ({frame_time:.2f}s) | "
                  f"verified: {len(engine.tables['table_1'].current_customers)} | "
                  f"disp_cnt: {status['customer_count']} | "
                  f"is_occupied: {all_status['is_occupied']}")

if __name__ == "__main__":
    main()
