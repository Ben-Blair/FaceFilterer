import os
import cv2
import face_recognition
import pickle
import numpy as np
import shutil

# ============ STEP 1: LOAD ENCODINGS ============
with open('EncodeFile.p', 'rb') as file:
    encodeListKnown, studentIds = pickle.load(file)

# ============ STEP 2: DEFINE PARAMETERS ============
# Folder containing the images you want to scan
source_folder = '/Users/user/Desktop/input'

# Destination folder to hold images that contain the target person's face
destination_folder = '/Users/user/Desktop/output'
if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)

# The specific person's ID/name you want to detect
target_person_id = 'me'  # Replace with the actual ID or name in studentIds

# ============ STEP 3: SCAN THROUGH ALL IMAGES IN THE FOLDER ============
valid_extensions = ('.png', '.jpg', '.jpeg')
for filename in os.listdir(source_folder):
    # Only process valid image files
    if filename.lower().endswith(valid_extensions):
        filepath = os.path.join(source_folder, filename)

        # Read the image
        image = cv2.imread(filepath)
        if image is None:
            continue  # Skip if the image is not valid

        # Convert image to RGB (face_recognition uses RGB)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Find face locations and encodings in the image
        face_locations = face_recognition.face_locations(rgb_image)
        face_encodings = face_recognition.face_encodings(rgb_image, face_locations)

        # ============ STEP 4: COMPARE EACH FACE ENCODING ============
        found_target = False
        for face_encoding in face_encodings:
            # Compare this face with your known encodings
            matches = face_recognition.compare_faces(encodeListKnown, face_encoding, tolerance=0.6)
            face_distances = face_recognition.face_distance(encodeListKnown, face_encoding)
            match_index = np.argmin(face_distances)

            # If a match is found, check if it's the target person
            if matches[match_index]:
                # Check if the matched ID is the target person's ID
                if studentIds[match_index] == target_person_id:
                    found_target = True
                    break  # No need to check more faces if we already found the target

        # If found_target is True, copy or move the file to the destination folder
        if found_target:
            src_path = os.path.join(source_folder, filename)
            dst_path = os.path.join(destination_folder, filename)

            # Copy the file (use shutil.move() if you want to move instead)
            shutil.copy2(src_path, dst_path) # use move instead of copy2 if you want to move the file instead of copying it
            print(f"[INFO] Copied '{filename}' because it contains '{target_person_id}'.")
