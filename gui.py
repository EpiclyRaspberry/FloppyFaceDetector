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
    WINDOW_WIDTH = 980
    WINDOW_HEIGHT = 760
    WIDTH = 640
    HEIGHT = 480

    def __init__(self):
        super().__init__("Face Attendance", "park", "dark")
        self.root.geometry(f"{self.WINDOW_WIDTH}x{self.WINDOW_HEIGHT}")
        self.root.minsize(self.WINDOW_WIDTH, self.WINDOW_HEIGHT)
        self.root.configure(bg="#101820")


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
        self.status_text = "Idle"
        self.status_tone = "idle"
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

    def setup_widgets(self):
        self.root.grid_columnconfigure(0, weight=3)
        self.root.grid_columnconfigure(1, weight=2)
        self.root.grid_rowconfigure(1, weight=1)

        self.header_frame = tk.Frame(self.root, bg="#101820", padx=20, pady=18)
        self.header_frame.grid(row=0, column=0, columnspan=2, sticky="ew")
        self.header_frame.grid_columnconfigure(0, weight=1)

        self.title_label = tk.Label(
            self.header_frame,
            text="Face Attendance Console",
            font=("Segoe UI", 20, "bold"),
            fg="#F4F1DE",
            bg="#101820",
        )
        self.title_label.grid(row=0, column=0, sticky="w")

        self.subtitle_label = tk.Label(
            self.header_frame,
            text="Live detection, face enrollment, and attendance export in one workspace.",
            font=("Segoe UI", 10),
            fg="#9DB4C0",
            bg="#101820",
        )
        self.subtitle_label.grid(row=1, column=0, sticky="w", pady=(4, 0))

        self.preview_card = tk.Frame(
            self.root,
            bg="#17232F",
            highlightbackground="#2A3A4A",
            highlightthickness=1,
            padx=18,
            pady=18,
        )
        self.preview_card.grid(row=1, column=0, sticky="nsew", padx=(20, 10), pady=(0, 20))
        self.preview_card.grid_columnconfigure(0, weight=1)

        self.preview_title = tk.Label(
            self.preview_card,
            text="Camera Preview",
            font=("Segoe UI", 14, "bold"),
            fg="#F4F1DE",
            bg="#17232F",
        )
        self.preview_title.grid(row=0, column=0, sticky="w")

        self.preview_hint = tk.Label(
            self.preview_card,
            text="Keep one face centered for reliable matching and enrollment.",
            font=("Segoe UI", 9),
            fg="#91A3B0",
            bg="#17232F",
        )
        self.preview_hint.grid(row=1, column=0, sticky="w", pady=(2, 12))

        self.video_frame = tk.Frame(
            self.preview_card,
            bg="#0B1117",
            highlightbackground="#2F4858",
            highlightthickness=1,
            padx=10,
            pady=10,
        )
        self.video_frame.grid(row=2, column=0, sticky="nsew")

        self.video_label = tk.Label(
            self.video_frame,
            bg="#0B1117",
            fg="#F4F1DE",
            text="Waiting for camera feed...",
            font=("Segoe UI", 11),
        )
        self.video_label.pack(fill="both", expand=True)

        self.side_panel = tk.Frame(self.root, bg="#101820")
        self.side_panel.grid(row=1, column=1, sticky="nsew", padx=(10, 20), pady=(0, 20))
        self.side_panel.grid_columnconfigure(0, weight=1)

        self.status_card = tk.Frame(
            self.side_panel,
            bg="#17232F",
            highlightbackground="#2A3A4A",
            highlightthickness=1,
            padx=18,
            pady=18,
        )
        self.status_card.grid(row=0, column=0, sticky="ew")
        self.status_card.grid_columnconfigure(0, weight=1)

        self.status_heading = tk.Label(
            self.status_card,
            text="System Status",
            font=("Segoe UI", 13, "bold"),
            fg="#F4F1DE",
            bg="#17232F",
        )
        self.status_heading.grid(row=0, column=0, sticky="w")

        self.status_label = tk.Label(
            self.status_card,
            text="Idle",
            font=("Segoe UI", 16, "bold"),
            fg="#F4F1DE",
            bg="#263845",
            padx=14,
            pady=10,
        )
        self.status_label.grid(row=1, column=0, sticky="ew", pady=(12, 10))

        self.status_detail_label = tk.Label(
            self.status_card,
            text="Camera is ready. Start processing to begin detection.",
            font=("Segoe UI", 10),
            fg="#9DB4C0",
            bg="#17232F",
            justify="left",
            wraplength=260,
        )
        self.status_detail_label.grid(row=2, column=0, sticky="w")

        self.metrics_card = tk.Frame(
            self.side_panel,
            bg="#17232F",
            highlightbackground="#2A3A4A",
            highlightthickness=1,
            padx=18,
            pady=18,
        )
        self.metrics_card.grid(row=1, column=0, sticky="ew", pady=(14, 0))
        self.metrics_card.grid_columnconfigure(0, weight=1)
        self.metrics_card.grid_columnconfigure(1, weight=1)

        self.db_count_value = tk.Label(
            self.metrics_card,
            text="0",
            font=("Segoe UI", 22, "bold"),
            fg="#F2CC8F",
            bg="#17232F",
        )
        self.db_count_value.grid(row=0, column=0, sticky="w")
        self.db_count_label = tk.Label(
            self.metrics_card,
            text="Registered faces",
            font=("Segoe UI", 10),
            fg="#9DB4C0",
            bg="#17232F",
        )
        self.db_count_label.grid(row=1, column=0, sticky="w")

        self.frame_faces_value = tk.Label(
            self.metrics_card,
            text="0",
            font=("Segoe UI", 22, "bold"),
            fg="#81B29A",
            bg="#17232F",
        )
        self.frame_faces_value.grid(row=0, column=1, sticky="w")
        self.frame_faces_label = tk.Label(
            self.metrics_card,
            text="Faces in frame",
            font=("Segoe UI", 10),
            fg="#9DB4C0",
            bg="#17232F",
        )
        self.frame_faces_label.grid(row=1, column=1, sticky="w")

        self.actions_card = tk.Frame(
            self.side_panel,
            bg="#17232F",
            highlightbackground="#2A3A4A",
            highlightthickness=1,
            padx=18,
            pady=18,
        )
        self.actions_card.grid(row=2, column=0, sticky="ew", pady=(14, 0))
        self.actions_card.grid_columnconfigure(0, weight=1)

        self.actions_heading = tk.Label(
            self.actions_card,
            text="Actions",
            font=("Segoe UI", 13, "bold"),
            fg="#F4F1DE",
            bg="#17232F",
        )
        self.actions_heading.grid(row=0, column=0, sticky="w")

        self.start_process_button = tk.Button(
            self.actions_card,
            text="Start Camera Processing",
            command=self.toggle_process_state,
            font=("Segoe UI", 11, "bold"),
            bg="#3D5A80",
            fg="#F4F1DE",
            activebackground="#4B6C96",
            activeforeground="#F4F1DE",
            relief="flat",
            bd=0,
            padx=12,
            pady=12,
            cursor="hand2",
        )
        self.start_process_button.grid(row=1, column=0, sticky="ew", pady=(14, 10))

        self.snap_button = tk.Button(
            self.actions_card,
            text="Log Attendance",
            command=self.log_attendance_process,
            font=("Segoe UI", 11, "bold"),
            bg="#81B29A",
            fg="#0B1117",
            activebackground="#8FC4AA",
            activeforeground="#0B1117",
            relief="flat",
            bd=0,
            padx=12,
            pady=12,
            cursor="hand2",
        )
        self.snap_button.grid(row=2, column=0, sticky="ew", pady=(0, 10))

        self.add_new_button = tk.Button(
            self.actions_card,
            text="Add Face To Database",
            command=self.add_new_face_to_db,
            font=("Segoe UI", 11, "bold"),
            bg="#F2CC8F",
            fg="#0B1117",
            activebackground="#F6D8A7",
            activeforeground="#0B1117",
            relief="flat",
            bd=0,
            padx=12,
            pady=12,
            cursor="hand2",
        )
        self.add_new_button.grid(row=3, column=0, sticky="ew", pady=(0, 10))

        self.export_button = tk.Button(
            self.actions_card,
            text="Export Database To Excel",
            command=self.export_data,
            font=("Segoe UI", 11, "bold"),
            bg="#E07A5F",
            fg="#F4F1DE",
            activebackground="#E89079",
            activeforeground="#F4F1DE",
            relief="flat",
            bd=0,
            padx=12,
            pady=12,
            cursor="hand2",
        )
        self.export_button.grid(row=4, column=0, sticky="ew")

        self._refresh_counts()
        self._set_status("Idle", "idle", "Camera is ready. Start processing to begin detection.")
    
    def toggle_process_state(self):
        self.detection_running = not self.detection_running
        if self.detection_running:
            self.start_process_button.config(text="Stop Camera Processing", bg="#8D99AE", activebackground="#A6B0C1")
            self._set_status("Processing Enabled", "idle", "Live face detection is active.")
        else:
            self.start_process_button.config(text="Start Camera Processing", bg="#3D5A80", activebackground="#4B6C96")
            self._set_status("Idle", "idle", "Detection paused. Camera preview is still available.")
        self._notify_user()

    def add_new_face_to_db(self):
        cropped_img = self.get_current_cropped_face()
        
        if self.detection_running is False:
            self._notify_user()
            messagebox.showwarning("Warning", "Face detection is not active. Please start the camera processing.")
            return

        if self._has_multiple_faces():
            self._set_status("Multiple Faces Detected", "error", "Only one face can be enrolled at a time.")
            self._notify_user()
            messagebox.showerror("Error", "Multiple faces detected. Please make sure only one face is visible.")
            return

        if cropped_img is None or cropped_img.size == 0:
            self._set_status("No Face Detected", "error", "No usable face was found for enrollment.")
            self._notify_user()
            messagebox.showerror("Error", "No face detected! Turn on processing and look at the camera.")
            return

        name = simpledialog.askstring("Input", "Enter Name of Person:")
        
        if name:
            success = self.db.add_user(name, cropped_img)
            if success:
                self.known_faces_cache = self.db.get_all_users()
                self._refresh_counts()
                self._set_status("Face Saved", "success", f"Saved {name} to the local face database.")
                self._notify_user()
                messagebox.showinfo("Success", f"Saved {name} to database.")
            else:
                self._set_status("Database Write Error", "error", "The face image could not be written to the database.")
                self._notify_user()
                messagebox.showerror("Error", "Failed to save to database.")

    def export_data(self):
        filename = "attendance_report.xlsx"
        if self.export_to_excel(filename):
            self._set_status("Export Complete", "success", f"Attendance workbook saved as {filename}.")
            messagebox.showinfo("Export Success", f"Data exported to {filename}")
            self._open_export_file(filename)
        else:
            self._set_status("Export Failed", "error", "Excel export did not complete successfully.")
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

    def _set_status(self, text: str, tone: str, detail: str):
        palette = {
            "idle": {"bg": "#263845", "fg": "#F4F1DE"},
            "success": {"bg": "#24463A", "fg": "#DFF3E3"},
            "error": {"bg": "#5B2A2A", "fg": "#FFE2E2"},
            "working": {"bg": "#5A4316", "fg": "#FFF1CC"},
        }
        style = palette.get(tone, palette["idle"])
        self.status_text = text
        self.status_tone = tone
        self.status_label.config(text=text, bg=style["bg"], fg=style["fg"])
        self.status_detail_label.config(text=detail)

    def _refresh_counts(self):
        self.db_count_value.config(text=str(len(self.known_faces_cache)))

        faces_in_frame = 0
        if self.results and self.results.face_landmarks:
            faces_in_frame = len(self.results.face_landmarks)
        self.frame_faces_value.config(text=str(faces_in_frame))

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
            self._notify_user()
            messagebox.showwarning("Warning", "Face detection is not active. Please start the camera processing.")
            return

        if self._has_multiple_faces():
            self._set_status("Multiple Faces Detected", "error", "Attendance logging requires exactly one visible face.")
            self._notify_user()
            messagebox.showerror("Error", "Multiple faces detected. Attendance logging requires exactly one face.")
            return

        target_img = self.get_current_cropped_face()
        if target_img is None or target_img.size == 0:
            self._set_status("No Face Detected", "error", "No usable face was found for attendance logging.")
            self._notify_user()
            messagebox.showerror("Error", "No face detected! Turn on processing and look at the camera.")
            return

        users = self.known_faces_cache
        if not users:
            self._set_status("Database Empty", "error", "Add at least one face before trying to match attendance.")
            self._notify_user()
            messagebox.showerror("Error", "Database is empty. Add a face first.")
            return

        self._set_status("Identifying...", "working", "Comparing the current face against registered users.")
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
                        self._notify_user()
                        messagebox.showinfo("Attendance Success", msg)
                        self._set_status("Attendance Logged", "success", f"Logged {best_match_name} at {timestamp_str}.")
                    else:
                        msg = "Match found, but Database Write Failed."
                        self._notify_user()
                        messagebox.showerror("Database Error", msg)
                        self._set_status("Write Error", "error", "A match was found, but attendance could not be saved.")
            else:
                msg = "Identity rejected by user. Please adjust lighting or angle and try again."
                self._set_status("Identity Rejected", "error", "User rejected the suggested identity match.")
                self._notify_user()
                messagebox.showinfo("Retry", msg)

        else:
            msg = "No Match Found in Database."
            self._set_status("Unknown Face", "error", "No registered user matched the detected face.")
            self._notify_user()
            messagebox.showwarning("Failed", msg)

    def _notify_user(self):
        self.root.bell()

    def _has_multiple_faces(self) -> bool:
        return bool(self.results and self.results.face_landmarks and len(self.results.face_landmarks) > 1)

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
                    if num_faces == 1:
                        status_text = 'Face Detected'
                        status_color = (0, 255, 0)
                        self._set_status("Face Detected", "success", "One face detected and ready for enrollment or attendance.")
                        self.display_image = self.draw_landmarks_on_image(self.display_image, self.results)
                    elif num_faces > 1:
                        status_text = 'Multiple Faces Detected'
                        status_color = (0, 0, 255)
                        self._set_status("Multiple Faces Detected", "error", "Reduce the frame to one person before continuing.")
                        self.display_image = self.draw_landmarks_on_image(self.display_image, self.results)
                else:
                    self._set_status("No Face Detected", "error", "Move into frame and face the camera.")

                text_color = (0, 255, 0) if status_color == (0, 255, 0) else (255, 0, 0)
                cv2.putText(self.display_image, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2, cv2.LINE_AA)
            else:
                self.display_image = cv2.cvtColor(self.display_image, cv2.COLOR_BGR2RGB)

            pil_image = Image.fromarray(self.display_image)
            pil_image = pil_image.resize((self.WIDTH, self.HEIGHT), Image.Resampling.LANCZOS)
            tk_image = ImageTk.PhotoImage(pil_image)
            self.video_label.config(image=tk_image)
            self.tk_image_ref = tk_image 

            self._refresh_counts()
        
        self.root.after(10, self.update_frame)

    def on_closing(self):
        self.cap.release()
        self.db.close()
        self.root.destroy()
