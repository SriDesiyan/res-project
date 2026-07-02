"""
Session Manager — Global customer identity stitching.

Matches broken YOLO track IDs using OSNet visual Re-ID embeddings
to create persistent Session IDs.
"""
import torch
import math
import cv2
import numpy as np


class SessionManager:
    def __init__(self, similarity_threshold=0.85, timeout_sec=900):
        self.similarity_threshold = similarity_threshold
        self.timeout_sec = timeout_sec
        
        # session_id -> dict(start_time, embedding, gallery, last_table, last_seen_time, role, dormant_since)
        self.active_sessions = {}
        # track_id -> session_id
        self.track_to_session = {}
        self._next_session_id = 1

    def update(self, persons: list, frame_time: float, frame_shape: tuple = None, tables: dict = None) -> dict:
        
        current_active_tids = {p.track_id for p in persons}
        current_active_sids = {self.track_to_session[tid] for tid in current_active_tids if tid in self.track_to_session}

        resumed_sessions = {}

        for person in persons:
            # If person already has a session mapping
            if person.track_id in self.track_to_session:
                sid = self.track_to_session[person.track_id]
                
                # Check if it was dormant
                if self.active_sessions[sid].get("dormant_since") is not None:
                    dormant_since = self.active_sessions[sid]["dormant_since"]
                    absent_duration = frame_time - dormant_since
                    # FIX: leave start_time alone upon re-entry to track total visit duration
                    resumed_sessions[sid] = absent_duration
                    self.active_sessions[sid]["dormant_since"] = None
                    print(f"[SessionManager] Resumed Session {sid}: absent {absent_duration:.1f}s (start_time unchanged)")
                
                # Update session data
                self.active_sessions[sid]["last_seen_time"] = frame_time
                if person.assigned_table:
                    self.active_sessions[sid]["last_table"] = person.assigned_table
                
                # Update the embedding gallery
                if person.visual_embedding is not None:
                    crop_area = (person.bbox[2] - person.bbox[0]) * (person.bbox[3] - person.bbox[1])
                    gallery = self.active_sessions[sid].setdefault("gallery", [])
                    gallery.append((person.visual_embedding, crop_area))
                    gallery.sort(key=lambda x: x[1], reverse=True)
                    self.active_sessions[sid]["gallery"] = gallery[:5]
                    self.active_sessions[sid]["embedding"] = gallery[0][0]
                
                # Sync role: session role takes precedence, or upgrade session to waiter
                if self.active_sessions[sid]["role"] == "waiter":
                    person.role = "waiter"
                elif person.role == "waiter":
                    # Customers are only established once they have been assigned/seated at a table for >5.0s
                    is_established_customer = (
                        self.active_sessions[sid]["role"] == "customer"
                        and self.active_sessions[sid].get("last_table") is not None
                        and (frame_time - self.active_sessions[sid]["start_time"] > 5.0)
                    )
                    if not is_established_customer:
                        self.active_sessions[sid]["role"] = "waiter"
                    else:
                        person.role = "customer"
                
                person.session_id = sid
                # Override the person's tracker timer with the true session start time
                person.first_seen = self.active_sessions[sid]["start_time"]
                continue

            # This is a NEW track_id. Try to stitch it.
            matched_sid = None
            if person.visual_embedding is not None:
                matched_sid = self._find_best_match(person, current_active_sids, frame_time, frame_shape, tables)

            if matched_sid is not None:
                # Stitch successful!
                self.track_to_session[person.track_id] = matched_sid
                
                # Check if it was dormant
                if self.active_sessions[matched_sid].get("dormant_since") is not None:
                    dormant_since = self.active_sessions[matched_sid]["dormant_since"]
                    absent_duration = frame_time - dormant_since
                    # FIX: leave start_time alone upon re-entry to track total visit duration
                    resumed_sessions[matched_sid] = absent_duration
                    self.active_sessions[matched_sid]["dormant_since"] = None
                    print(f"[SessionManager] Resumed Session {matched_sid} via Re-ID: absent {absent_duration:.1f}s")

                self.active_sessions[matched_sid]["last_seen_time"] = frame_time
                if person.assigned_table:
                    self.active_sessions[matched_sid]["last_table"] = person.assigned_table
                
                # Update the embedding gallery upon Re-ID match
                if person.visual_embedding is not None:
                    crop_area = (person.bbox[2] - person.bbox[0]) * (person.bbox[3] - person.bbox[1])
                    gallery = self.active_sessions[matched_sid].setdefault("gallery", [])
                    gallery.append((person.visual_embedding, crop_area))
                    gallery.sort(key=lambda x: x[1], reverse=True)
                    self.active_sessions[matched_sid]["gallery"] = gallery[:5]
                    self.active_sessions[matched_sid]["embedding"] = gallery[0][0]

                # Sync role
                if self.active_sessions[matched_sid]["role"] == "waiter":
                    person.role = "waiter"
                elif person.role == "waiter":
                    # Customers are only established once they have been assigned/seated at a table for >5.0s
                    is_established_customer = (
                        self.active_sessions[matched_sid]["role"] == "customer"
                        and self.active_sessions[matched_sid].get("last_table") is not None
                        and (frame_time - self.active_sessions[matched_sid]["start_time"] > 5.0)
                    )
                    if not is_established_customer:
                        self.active_sessions[matched_sid]["role"] = "waiter"
                    else:
                        person.role = "customer"
                
                person.session_id = matched_sid
                # Sync timer
                person.first_seen = self.active_sessions[matched_sid]["start_time"]
                print(f"[SessionManager] Stitched Track ID {person.track_id} to existing Session {matched_sid} (Role: {person.role})")
                current_active_sids.add(matched_sid)
            else:
                # Create a brand new session
                sid = f"S{self._next_session_id}"
                self._next_session_id += 1
                
                crop_area = 0
                gallery = []
                if person.visual_embedding is not None:
                    crop_area = (person.bbox[2] - person.bbox[0]) * (person.bbox[3] - person.bbox[1])
                    gallery = [(person.visual_embedding, crop_area)]

                self.active_sessions[sid] = {
                    "start_time": person.first_seen,
                    "embedding": person.visual_embedding,
                    "gallery": gallery,
                    "last_table": person.assigned_table,
                    "last_seen_time": frame_time,
                    "role": person.role,
                    "dormant_since": None
                }
                self.track_to_session[person.track_id] = sid
                person.session_id = sid
                current_active_sids.add(sid)

        # Mark sessions that are not on screen in this frame as dormant
        for sid, data in self.active_sessions.items():
            if sid not in current_active_sids and data.get("dormant_since") is None:
                data["dormant_since"] = frame_time
                print(f"[SessionManager] Session {sid} marked dormant")

        # Cleanup old disconnected sessions
        self._cleanup(frame_time)
        return resumed_sessions

    def _find_best_match(self, person, current_active_sids: set, frame_time: float, frame_shape: tuple = None, tables: dict = None) -> str | None:
        """Find the best matching disconnected session for this person."""
        best_sid = None
        best_sim = 0.0
        
        emb = person.visual_embedding
        
        for sid, data in self.active_sessions.items():
            if sid in current_active_sids:
                continue  # This session is already represented by someone on screen
                
            # Calculate similarity against all embeddings in the session's gallery
            max_gallery_sim = -1.0
            gallery = data.get("gallery", [])
            if gallery:
                for g_emb, _ in gallery:
                    sim_val = torch.mm(emb, g_emb.t()).item()
                    if sim_val > max_gallery_sim:
                        max_gallery_sim = sim_val
            elif data["embedding"] is not None:
                max_gallery_sim = torch.mm(emb, data["embedding"].t()).item()
                
            if max_gallery_sim == -1.0:
                continue
                
            sim = max_gallery_sim
            
            # Reject role mismatch to prevent cross-matching
            if data["role"] != person.role:
                continue
            
            # Decoupled Spatio-Temporal Gates
            # Time decay penalty (linear over the 15-minute timeout window)
            dormant_duration = frame_time - data["last_seen_time"]
            time_penalty = 0.10 * (dormant_duration / self.timeout_sec)
            sim -= time_penalty
            
            # Spatial Gate: Edge proximity boost
            near_edge = False
            if frame_shape is not None:
                fh, fw = frame_shape[:2]
                margin_x = fw * 0.08
                margin_y = fh * 0.08
                cx, cy = person.centroid
                if cx < margin_x or cx > fw - margin_x or cy < margin_y or cy > fh - margin_y:
                    near_edge = True
            
            # Spatial Gate: Last table proximity boost
            near_last_table = False
            if data["last_table"] and tables and data["last_table"] in tables:
                poly = np.array(tables[data["last_table"]]["polygon"], dtype=np.int32)
                dist = cv2.pointPolygonTest(poly, person.bottom_center, measureDist=True)
                if dist >= -150.0:  # within 150 pixels of the table polygon boundary
                    near_last_table = True

            # Enforce spatial gating for customer sessions to prevent teleportation/cross-matching
            if data["role"] == "customer":
                if data["last_table"]:
                    if not near_last_table:
                        continue
                else:
                    if not near_edge:
                        continue
            
            spatial_boost = 0.0
            if near_edge:
                spatial_boost += 0.05
            if near_last_table:
                spatial_boost += 0.08
                
            sim += spatial_boost
                
            if sim > best_sim and sim > self.similarity_threshold:
                best_sim = sim
                best_sid = sid
                
        return best_sid

    def _cleanup(self, frame_time: float):
        # remove sessions that have been disconnected for more than 15 mins
        to_delete = []
        for sid, data in self.active_sessions.items():
            if frame_time - data["last_seen_time"] > self.timeout_sec:
                to_delete.append(sid)
                
        for sid in to_delete:
            del self.active_sessions[sid]
            # Remove associated track mappings
            for tid, mapped_sid in list(self.track_to_session.items()):
                if mapped_sid == sid:
                    del self.track_to_session[tid]

