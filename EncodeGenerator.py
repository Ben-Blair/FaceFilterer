import cv2
import face_recognition
import pickle
import os

# Importing the student images
folderPath = 'Images'
pathList = os.listdir(folderPath)
print(pathList)
imgList = []
studentIds = []

# Filter out non-image files (DS Store, etc.)
valid_extensions = ('.png', '.jpg', '.jpeg')  # Add other extensions if needed
for path in pathList:
    if path.endswith(valid_extensions):  # Check if the file is an image
        imgList.append(cv2.imread(os.path.join(folderPath, path)))
        studentIds.append(os.path.splitext(path)[0])
print(studentIds)


def findEncodings(imagesList):
    encodeList = []
    for img in imagesList:
        if img is None:
            continue  # Skip invalid images
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

print("Encoding Started...")
encodeListKnown = findEncodings(imgList)
encodeListKnownWithIds = [encodeListKnown, studentIds]
print("Encoding Complete")

file = open("EncodeFile.p", "wb")
pickle.dump(encodeListKnownWithIds, file)
file.close()
print("File Saved")




