import face_recognition
import cv2
import os
import glob
import numpy as np

class SimpleFacerec:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []

        # Resize frame for a faster speed
        self.frame_resizing = 0.25

    def load_encoding_images(self, images_path):
        """
        Load encoding images from path
        :param images_path: Path where images are stored
        """
        # Load Images
        images_path = glob.glob(os.path.join(images_path, "*.*"))

        print(f"{len(images_path)} encoding images found.")

        # Store image encoding and names
        for img_path in images_path:
            img = cv2.imread(img_path)
            if img is None:
                print(f"Warning: Unable to load image {img_path}")
                continue

            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Get the filename only from the initial file path.
            basename = os.path.basename(img_path)
            (filename, ext) = os.path.splitext(basename)
            # Get encoding
            face_encodings = face_recognition.face_encodings(rgb_img)
            if face_encodings:
                img_encoding = face_encodings[0]
                # Store file name and file encoding
                self.known_face_encodings.append(img_encoding)
                self.known_face_names.append(filename)
            else:
                print(f"No face found in image {img_path}")

        print("Encoding images loaded")

    def detect_known_faces(self, frame):
        # Resize the frame to speed up face detection
        small_frame = cv2.resize(frame, None, fx=self.frame_resizing, fy=self.frame_resizing)
        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # Compare the face encoding with known face encodings
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
            name = "Unknown"

            # Find the best match
            if matches:
                face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = self.known_face_names[best_match_index]

            face_names.append(name)

        # Adjust face locations back to the original size
        face_locations = np.array(face_locations)
        face_locations = face_locations / self.frame_resizing
        return face_locations.astype(int), face_names
