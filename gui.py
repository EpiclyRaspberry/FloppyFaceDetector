import tkinter as tk
import TKinterModernThemes as TKMT
from tkinter import simpledialog, messagebox
from PIL import Image, ImageTk
from io import BytesIO
import cv2
import os
import shutil
import subprocess
import pandas as pd
import numpy as np
import mediapipe as mp
from openpyxl.drawing.image import Image as ExcelImage
from openpyxl.utils import get_column_letter
import mediapipe.python.solutions.face_mesh as mp_face_mesh
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.components.containers import NormalizedLandmark

# Custom Imports
from database import FaceDatabase
from face_processor import FaceLandmarkerWrapper
from recognizer import FaceRecognizer

class FaceMeshApp(TKMT.ThemedTKinterFrame):
    WINDOW_WIDTH = 620
    WINDOW_HEIGHT = 560
    WIDTH = 640
    HEIGHT = 480

    def __init__(self):
        super().__init__("Face Attendance", "park", "dark")
        self.root.geometry(f"{self.WINDOW_WIDTH + 40}x{self.WINDOW_HEIGHT + 150}")


        # Initialize Modules
        self.db = FaceDatabase()
        self.processor = FaceLandmarkerWrapper()
        self.recognizer = FaceRecognizer()
        
        # State
        self.known_faces_cache = self.db.get_all_users()
        self.detection_running = False
        self.tk_image_ref = None 
        self.results: vision.FaceLandmarkerResult = None # type: ignore
        self.display_image = None 
        self.current_frame_bgr = None 
        # Camera
        self.cap = cv2.VideoCapture(0)
        
        # Setup UI
        self.setup_widgets()
        if not self.cap.isOpened():
            print("Error: Could not open video stream.")
            self.root.destroy() 
            raise IOError("Cannot open webcam.")
        
        # Start Loop
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.after(10, self.update_frame)

    # ... [Paste setup_widgets, toggle_process_state, etc.] ...
    def setup_widgets(self):
        self.video_label = self.Label("", padx=10, pady=10)
        self.status_label = self.Label(text="Idle")
        
        self.buttonRows = self.addFrame("buttonrows", padx=10, pady=5, sticky=tk.W + tk.E)
        self.buttonRows.snap_button = self.buttonRows.Button( # type: ignore
            "Log Attendance", 
            command=self.log_attendance_process, 
            pady=5, col=0, sticky=tk.W + tk.E
        )
        
        self.buttonRows.start_process_button = self.buttonRows.Button( # type: ignore
            "Start/Stop Camera", 
            command=self.toggle_process_state,
            pady=5, col=1, sticky=tk.W + tk.E
        )
        
        self.add_new_button = self.Button(
            "Add New Face to DB", 
            command=self.add_new_face_to_db, 
            pady=5
        )
        self.export_button = self.Button(
            "Export DB to Excel", 
            command=self.export_data,
            pady=5
        )
    
    def toggle_process_state(self):
        self.detection_running = not self.detection_running
        self.status_label.config(text=f"Processing: On" if self.detection_running else 'Idle')
        # ws.MessageBeep(ws.MB_ICONEXCLAMATION)

    def add_new_face_to_db(self):
        cropped_img = self.get_current_cropped_face()
        
        if self.detection_running is False:
            messagebox.showwarning("Warning", "Face detection is not active. Please start the camera processing.")
            return

        if cropped_img is None or cropped_img.size == 0:
            messagebox.showerror("Error", "No face detected! Turn on processing and look at the camera.")
            return

        name = simpledialog.askstring("Input", "Enter Name of Person:")
        
        if name:
            success = self.db.add_user(name, cropped_img)
            if success:
                self.known_faces_cache = self.db.get_all_users()
                messagebox.showinfo("Success", f"Saved {name} to database.")
            else:
                messagebox.showerror("Error", "Failed to save to database.")

    def export_data(self):
        filename = "attendance_report.xlsx"
        if self.export_to_excel(filename):
            messagebox.showinfo("Export Success", f"Data exported to {filename}")
            self._open_export_file(filename)
        else:
            messagebox.showerror("Export Failed", "Could not export data.")

    def export_to_excel(self, filename="attendance_report.xlsx"):
        try:
            query_users = "SELECT id, name, image_blob FROM users"
            df_users = pd.read_sql_query(query_users, self.db.conn)
            query_attendance = "SELECT log_id, user_id, name, time_str, capture_image_blob FROM attendance"
            df_attendance = pd.read_sql_query(query_attendance, self.db.conn)

            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                self._export_sheet_with_images(
                    writer=writer,
                    dataframe=df_users,
                    sheet_name='Registered Users',
                    image_column='image_blob',
                    image_header='Face Image',
                )
                self._export_sheet_with_images(
                    writer=writer,
                    dataframe=df_attendance,
                    sheet_name='Attendance Logs',
                    image_column='capture_image_blob',
                    image_header='Capture Image',
                )
                
            print(f"Database exported successfully to {filename}")
            return True
        except Exception as e:
            print(f"Export failed: {e}")
            return False

    def _export_sheet_with_images(self, writer, dataframe: pd.DataFrame, sheet_name: str, image_column: str, image_header: str):
        export_df = dataframe.drop(columns=[image_column]).copy()
        export_df[image_header] = ""
        export_df.to_excel(writer, sheet_name=sheet_name, index=False)

        worksheet = writer.sheets[sheet_name]
        image_column_index = len(export_df.columns)
        image_column_letter = get_column_letter(image_column_index)
        worksheet.column_dimensions[image_column_letter].width = 18

        for row_number, image_blob in enumerate(dataframe[image_column], start=2):
            worksheet.row_dimensions[row_number].height = 85

            if not image_blob:
                continue

            pil_image = Image.open(BytesIO(image_blob))
            pil_image.thumbnail((96, 96))
            image_buffer = BytesIO()
            pil_image.save(image_buffer, format="PNG")
            image_buffer.seek(0)

            excel_image = ExcelImage(image_buffer)
            excel_image.anchor = f"{image_column_letter}{row_number}"
            worksheet.add_image(excel_image)

    def _open_export_file(self, filename: str):
        try:
            if os.name == "nt":
                os.startfile(filename)  # type: ignore[attr-defined]
                return

            opener = shutil.which("xdg-open")
            if opener is not None:
                subprocess.Popen(
                    [opener, filename],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
        except Exception as e:
            print(f"Could not automatically open export file: {e}")
        
    def log_attendance_process(self):
        if not self.detection_running:
            messagebox.showwarning("Warning", "Face detection is not active. Please start the camera processing.")
            return

        target_img = self.get_current_cropped_face()
        if target_img is None or target_img.size == 0:
            messagebox.showerror("Error", "No face detected! Turn on processing and look at the camera.")
            return

        users = self.known_faces_cache
        if not users:
            messagebox.showerror("Error", "Database is empty. Add a face first.")
            return

        self.status_label.config(text="Identifying...")
        self.root.update()

        recognition_result = self.recognizer.find_match(target_img, users)
        if recognition_result:
            best_match_uid, best_match_name, best_distance = recognition_result

            is_confirmed = messagebox.askyesno(
                "Confirm Identity", 
                f"System identified you as:\n\n{best_match_name}\n\nIs this correct?\nConfidence: {best_distance:.4f}"
            )
            
            if is_confirmed:
                if self.current_frame_bgr is not None:
                    success, timestamp_str = self.db.log_attendance(
                        best_match_uid, 
                        best_match_name, 
                        self.current_frame_bgr
                    )
                    
                    if success:
                        msg = f"Attendance Logged: {best_match_name}\nTime: {timestamp_str}"
                        # ws.MessageBeep(ws.MB_OK)
                        messagebox.showinfo("Attendance Success", msg)
                        self.status_label.config(text=f"Logged: {best_match_name}, at {timestamp_str}")
                    else:
                        msg = "Match found, but Database Write Failed."
                        # ws.MessageBeep(ws.MB_ICONHAND)
                        messagebox.showerror("Database Error", msg)
                        self.status_label.config(text="Write Error")
            else:
                msg = "Identity rejected by user. Please adjust lighting or angle and try again."
                self.status_label.config(text="Identity Rejected")
                messagebox.showinfo("Retry", msg)

        else:
            msg = "No Match Found in Database."
            # ws.MessageBeep(ws.MB_ICONHAND)
            self.status_label.config(text="Unknown Face")
            messagebox.showwarning("Failed", msg)

    def get_current_cropped_face(self):
        if not self.results or not self.results.face_landmarks or self.current_frame_bgr is None:
            return None

        face_landmarks = self.results.face_landmarks[0]
        cropped_img, _ = self.crop_face(self.current_frame_bgr, face_landmarks)
        return cropped_img

    def crop_face(self, image: np.ndarray, landmarks: list[NormalizedLandmark], padding_factor: float = 0.20): # type: ignore
        h, w, _ = image.shape
        x_coords = [l.x for l in landmarks]
        y_coords = [l.y for l in landmarks]
        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)
        
        face_w, face_h = max_x - min_x, max_y - min_y
        p_w, p_h = face_w * padding_factor, face_h * padding_factor
        
        start_x = max(0, int((min_x - p_w) * w))
        start_y = max(0, int((min_y - p_h) * h))
        end_x = min(w, int((max_x + p_w) * w))
        end_y = min(h, int((max_y + p_h) * h))
        
        return image[start_y:end_y, start_x:end_x], (start_x, start_y, end_x-start_x, end_y-start_y)

    def draw_landmarks_on_image(self, rgb_image, detection_result):
        face_landmarks_list = detection_result.face_landmarks
        annotated_image = np.copy(rgb_image)
        height, width, _ = annotated_image.shape

        for face_landmarks in face_landmarks_list:
            for connection in mp_face_mesh.FACEMESH_CONTOURS:
                start_idx = connection[0]
                end_idx = connection[1]
                start_point = face_landmarks[start_idx]
                end_point = face_landmarks[end_idx]

                x1 = int(start_point.x * width)
                y1 = int(start_point.y * height)
                x2 = int(end_point.x * width)
                y2 = int(end_point.y * height)
            
                cv2.line(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 1)

        return annotated_image

    def update_frame(self):
        success, image = self.cap.read()
        
        if success:
            image = cv2.flip(image, 1) 
            self.current_frame_bgr = image.copy() # Save RAW BGR
            self.display_image = image.copy() 
            
            if self.detection_running:
                self.display_image = cv2.cvtColor(self.display_image, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=self.display_image)
                self.results = self.processor.landmarker.detect(mp_image)
                
                status_text = 'No face detected'
                status_color = (0, 0, 255) 

                if self.results.face_landmarks:
                    num_faces = len(self.results.face_landmarks)
                    if num_faces >= 1:
                        status_text = 'Face Detected'
                        status_color = (0, 255, 0)
                        self.display_image = self.draw_landmarks_on_image(self.display_image, self.results)

                text_color = (0, 255, 0) if status_color == (0, 255, 0) else (255, 0, 0)
                cv2.putText(self.display_image, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2, cv2.LINE_AA)
            else:
                self.display_image = cv2.cvtColor(self.display_image, cv2.COLOR_BGR2RGB)

            pil_image = Image.fromarray(self.display_image)
            pil_image = pil_image.resize((self.WIDTH, self.HEIGHT), Image.Resampling.LANCZOS)
            tk_image = ImageTk.PhotoImage(pil_image)
            self.video_label.config(image=tk_image)
            self.tk_image_ref = tk_image 
        
        self.root.after(10, self.update_frame)

    def on_closing(self):
        self.cap.release()
        self.db.close()
        self.root.destroy()
