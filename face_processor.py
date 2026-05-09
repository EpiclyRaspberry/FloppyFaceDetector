import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.python.solutions import face_mesh as mp_face_mesh
from mediapipe.tasks.python.components.containers import NormalizedLandmark
from importlib.metadata import version
import cv2
import numpy as np
import os

class FaceLandmarkerWrapper:
    def __init__(self, model_path="assets/face_landmarker.task"):
        self._validate_runtime_dependencies()

        resolved_model_path = os.path.abspath(model_path)
        if not os.path.exists(resolved_model_path):
            raise FileNotFoundError(f"Model not found at {resolved_model_path}")

        base_options = python.BaseOptions(model_asset_path=resolved_model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=2,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(options)

    @staticmethod
    def _validate_runtime_dependencies():
        protobuf_version = version("protobuf")
        protobuf_major = int(protobuf_version.split(".", 1)[0])

        if protobuf_major >= 5:
            raise RuntimeError(
                "Incompatible runtime detected: mediapipe 0.10.x requires protobuf < 5, "
                f"but protobuf {protobuf_version} is installed. Install a compatible set such as "
                "'protobuf==4.25.8 tensorflow==2.15.1 keras==2.15.0'."
            )

    def detect(self, image_bgr):
        """Accepts BGR image, returns MediaPipe result object."""
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        return self.landmarker.detect(mp_image)

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
