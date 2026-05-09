from abc import ABC, abstractmethod
import sqlite3
import numpy as np
import cv2
import time
from datetime import datetime
import os

class Database(ABC):
    @property
    @abstractmethod
    def conn(self):
        pass

    @property
    @abstractmethod
    def db_name(self) -> str:
        pass

    @db_name.setter
    @abstractmethod
    def db_name(self, value: str):
        pass

    @abstractmethod
    def create_table(self, query: str):
        pass

    @abstractmethod
    def read(self, query: str, params: tuple = ()):
        pass

    @abstractmethod
    def write(self, query: str, params: tuple = ()) -> bool:
        pass


class FaceDatabase(Database):
    def __init__(self, db_name="assets/faces.db"):
        self._db_name = db_name
        self._conn: sqlite3.Connection | None = None
        self._cursor: sqlite3.Cursor | None = None
        self._connect()
        self.create_tables()

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            raise RuntimeError("Database connection is closed.")
        return self._conn

    @property
    def cursor(self) -> sqlite3.Cursor:
        if self._cursor is None:
            raise RuntimeError("Database cursor is unavailable.")
        return self._cursor

    @property
    def db_name(self) -> str:
        return self._db_name

    @db_name.setter
    def db_name(self, value: str):
        self.close()
        self._db_name = value
        self._connect()
        self.create_tables()

    def _connect(self):
        directory = os.path.dirname(self._db_name)
        if directory:
            os.makedirs(directory, exist_ok=True)
        self._conn = sqlite3.connect(self._db_name)
        self._cursor = self._conn.cursor()

    def create_table(self, query: str):
        self.cursor.execute(query)
        self.conn.commit()

    def read(self, query: str, params: tuple = ()):
        self.cursor.execute(query, params)
        return self.cursor.fetchall()

    def write(self, query: str, params: tuple = ()) -> bool:
        self.cursor.execute(query, params)
        self.conn.commit()
        return True
    
    def create_tables(self):
        # Table 1: Registered Users
        self.create_table("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                image_blob BLOB NOT NULL
            )
        """)
        
        # Table 2: Attendance Logs
        self.create_table("""
            CREATE TABLE IF NOT EXISTS attendance (
                log_id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                name TEXT NOT NULL,
                time_unix INTEGER NOT NULL,
                time_str TEXT NOT NULL,
                capture_image_blob BLOB NOT NULL
            )
        """)
    
    def add_user(self, name: str, image_array: np.ndarray):
        success, encoded_img = cv2.imencode('.jpg', image_array)
        if success:
            img_blob = encoded_img.tobytes()
            return self.write(
                "INSERT INTO users (name, image_blob) VALUES (?, ?)",
                (name, img_blob),
            )
        return False

    def log_attendance(self, user_id: int, name: str, full_image_array: np.ndarray | cv2.typing.MatLike) -> tuple[bool, str | None]:
        now = datetime.now()
        unix_time = int(time.time())
        human_time = now.strftime("%Y-%m-%d %H:%M:%S")

        success, encoded_img = cv2.imencode('.jpg', full_image_array)
        
        if success:
            img_blob = encoded_img.tobytes()
            write_success = self.write("""
                INSERT INTO attendance (user_id, name, time_unix, time_str, capture_image_blob) 
                VALUES (?, ?, ?, ?, ?)
            """, (user_id, name, unix_time, human_time, img_blob))
            if write_success:
                print(f"Attendance Logged: {name} at {human_time}")
                return True, human_time
        return False, None

    def get_all_users(self):
        rows = self.read("SELECT id, name, image_blob FROM users")
        
        users = []
        for row in rows:
            uid, name, blob = row
            nparr = np.frombuffer(blob, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            users.append((uid, name, img))
        return users

    def close(self):
        if self._conn is not None:
            self._conn.close()
        self._conn = None
        self._cursor = None
