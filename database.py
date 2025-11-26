import sqlite3
import numpy as np
import cv2
import time
from datetime import datetime
import pandas as pd
import os

class FaceDatabase:
    def __init__(self, db_name="assets/faces.db"):
        # Ensure directory exists
        os.makedirs(os.path.dirname(db_name), exist_ok=True)
        self.conn = sqlite3.connect(db_name)
        self.cursor = self.conn.cursor()
        self.create_tables()
    
    def create_tables(self):
        # Table 1: Registered Users
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                image_blob BLOB NOT NULL
            )
        """)
        
        # Table 2: Attendance Logs
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS attendance (
                log_id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                name TEXT NOT NULL,
                time_unix INTEGER NOT NULL,
                time_str TEXT NOT NULL,
                capture_image_blob BLOB NOT NULL
            )
        """)
        self.conn.commit()
    
    def add_user(self, name: str, image_array: np.ndarray):
        success, encoded_img = cv2.imencode('.jpg', image_array)
        if success:
            img_blob = encoded_img.tobytes()
            self.cursor.execute("INSERT INTO users (name, image_blob) VALUES (?, ?)", (name, img_blob))
            self.conn.commit()
            return True
        return False

    def log_attendance(self, user_id: int, name: str, full_image_array: np.ndarray | cv2.typing.MatLike) -> tuple[bool, str | None]:
        now = datetime.now()
        unix_time = int(time.time())
        human_time = now.strftime("%Y-%m-%d %H:%M:%S")

        success, encoded_img = cv2.imencode('.jpg', full_image_array)
        
        if success:
            img_blob = encoded_img.tobytes()
            self.cursor.execute("""
                INSERT INTO attendance (user_id, name, time_unix, time_str, capture_image_blob) 
                VALUES (?, ?, ?, ?, ?)
            """, (user_id, name, unix_time, human_time, img_blob))
            self.conn.commit()
            print(f"Attendance Logged: {name} at {human_time}")
            return True, human_time
        return False, None

    def get_all_users(self):
        self.cursor.execute("SELECT id, name, image_blob FROM users")
        rows = self.cursor.fetchall()
        
        users = []
        for row in rows:
            uid, name, blob = row
            nparr = np.frombuffer(blob, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            users.append((uid, name, img))
        return users

    def close(self):
        self.conn.close()