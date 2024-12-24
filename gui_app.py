import os
import cv2
import face_recognition
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from zipfile import ZipFile

def create_encoding_for_face(face_image_path):
    """
    Creates a face encoding for the single face image provided.
    """
    if not os.path.exists(face_image_path):
        raise ValueError(f"Face image not found: {face_image_path}")

    img = cv2.imread(face_image_path)
    if img is None:
        raise ValueError(f"Failed to read image file: {face_image_path}")

    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_img)

    if len(face_locations) == 0:
        raise ValueError(f"No faces found in {face_image_path}")

    face_encodings = face_recognition.face_encodings(rgb_img, face_locations)
    if len(face_encodings) == 0:
        raise ValueError(f"Could not generate face encoding for {face_image_path}")

    return face_encodings[0]  # return the first encoding

def find_matched_images(source_folder, face_encoding, progress_callback=None):
    """
    Goes through each photo in source_folder, compares to face_encoding.
    Returns a list of full file paths that matched.
    """
    matched_filepaths = []
    valid_extensions = ('.png', '.jpg', '.jpeg')

    # Gather valid image files
    all_files = [f for f in os.listdir(source_folder) if f.lower().endswith(valid_extensions)]
    total_files = len(all_files)

    for i, filename in enumerate(all_files, start=1):
        if progress_callback:
            progress_callback(i, total_files)  # update progress bar

        filepath = os.path.join(source_folder, filename)
        image = cv2.imread(filepath)
        if image is None:
            continue  # skip unreadable images

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_image)
        face_encodings = face_recognition.face_encodings(rgb_image, face_locations)

        # Compare each found face against the single face_encoding
        for enc in face_encodings:
            match = face_recognition.compare_faces([face_encoding], enc, tolerance=0.6)[0]
            if match:
                matched_filepaths.append(filepath)  # store full path
                break  # no need to check more faces in this image

    return matched_filepaths

class FaceFilterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Filter App")

        # User selections
        self.source_folder = ""
        self.face_image_path = ""

        # Widgets
        self.create_widgets()

    def create_widgets(self):
        frame = tk.Frame(self.root, padx=10, pady=10)
        frame.pack()

        # 1. Select input folder
        btn_select_folder = tk.Button(frame, text="Select Input Folder", command=self.select_input_folder)
        btn_select_folder.grid(row=0, column=0, padx=5, pady=5, sticky="ew")

        self.label_folder = tk.Label(frame, text="No folder selected", fg="gray")
        self.label_folder.grid(row=0, column=1, padx=5, pady=5, sticky="w")

        # 2. Select face image
        btn_select_face = tk.Button(frame, text="Select Person's Face Photo", command=self.select_face_image)
        btn_select_face.grid(row=1, column=0, padx=5, pady=5, sticky="ew")

        self.label_face = tk.Label(frame, text="No face image selected", fg="gray")
        self.label_face.grid(row=1, column=1, padx=5, pady=5, sticky="w")

        # 3. Process button
        btn_process = tk.Button(frame, text="Process/Filter", command=self.process_images)
        btn_process.grid(row=2, column=0, columnspan=2, padx=5, pady=10, sticky="ew")

        # Progress bar
        self.progress_bar = ttk.Progressbar(frame, orient="horizontal", length=200, mode="determinate")
        self.progress_bar.grid(row=3, column=0, columnspan=2, pady=5, sticky="ew")
        self.progress_bar.grid_remove()  # hidden by default

    def select_input_folder(self):
        folder_selected = filedialog.askdirectory(title="Select the folder with input photos")
        if folder_selected:
            self.source_folder = folder_selected
            self.label_folder.config(text=os.path.basename(folder_selected), fg="black")

    def select_face_image(self):
        file_selected = filedialog.askopenfilename(
            title="Select the face photo",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png")]
        )
        if file_selected:
            self.face_image_path = file_selected
            self.label_face.config(text=os.path.basename(file_selected), fg="black")

    def process_images(self):
        """
        1. Check user selections.
        2. Create an encoding for the chosen face photo.
        3. Scan the input folder for matches (with progress bar).
        4. Prompt user for ZIP file location and create a ZIP with matched images.
        """
        if not self.source_folder:
            messagebox.showwarning("Warning", "Please select an input folder first.")
            return
        if not self.face_image_path:
            messagebox.showwarning("Warning", "Please select a face photo first.")
            return

        # Show (and reset) the progress bar
        self.progress_bar['value'] = 0
        self.progress_bar.grid()
        self.progress_bar.update()

        # Create the face encoding
        try:
            face_encoding = create_encoding_for_face(self.face_image_path)
        except Exception as e:
            messagebox.showerror("Error", str(e))
            self.progress_bar.grid_remove()
            return

        # Find matched images
        matched_files = find_matched_images(
            self.source_folder,
            face_encoding,
            progress_callback=self.update_progress
        )

        # Hide progress bar
        self.progress_bar.grid_remove()

        # Let user know how many matches were found
        messagebox.showinfo("Results", f"Found {len(matched_files)} images matching the selected face.")

        if len(matched_files) == 0:
            return  # nothing to zip

        # Ask user where to save the ZIP
        save_zip_path = filedialog.asksaveasfilename(
            title="Save matched images as ZIP",
            defaultextension=".zip",
            filetypes=[("Zip Files", "*.zip")],
            initialfile="output.zip"
        )
        if not save_zip_path:
            return  # user canceled

        # Create the ZIP file with matched images
        try:
            with ZipFile(save_zip_path, 'w') as zipf:
                for filepath in matched_files:
                    # Arcname: how the file will appear inside the zip
                    filename = os.path.basename(filepath)
                    zipf.write(filepath, arcname=filename)
            
            messagebox.showinfo("Done", f"Matched images saved as:\n{save_zip_path}")
        except Exception as ex:
            messagebox.showerror("Error", f"Could not create ZIP file:\n{ex}")

    def update_progress(self, current, total):
        """Update the progress bar with the current index vs total."""
        if total <= 0:
            self.progress_bar['value'] = 0
        else:
            percent = (current / total) * 100
            self.progress_bar['value'] = percent
        self.progress_bar.update()

def main():
    root = tk.Tk()
    app = FaceFilterApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
