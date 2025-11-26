from deepface import DeepFace

class FaceRecognizer:
    def __init__(self, model_name="ArcFace"):
        self.model_name = model_name

    def find_match(self, target_img, user_db_list) -> tuple[int, str, float] | None:
        best_name: str = ""
        best_uid: int = -1
        best_dist = 1.0 
        match_found = False

        for uid, name, db_image in user_db_list:
            try:
                result = DeepFace.verify(
                    img1_path=target_img,
                    img2_path=db_image,
                    model_name=self.model_name,
                    detector_backend="skip",
                    enforce_detection=False,
                    distance_metric="cosine"
                )
                
                if result['verified'] and result['distance'] < best_dist:
                    best_dist = result['distance']
                    best_name = name
                    best_uid = uid
                    match_found = True
            except Exception as e:
                print(f"Error checking {name}: {e}")
                continue

        if match_found:
            return best_uid, best_name, best_dist
        return None