import os
import cv2
import face_recognition
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from zipfile import ZipFile
from PIL import Image, ImageTk

# 1) Import from tkinterdnd2
try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
except ImportError:
    raise ImportError(
        "Please install tkinterdnd2:\n\n"
        "   pip install tkinterdnd2\n\n"
        "and ensure it's properly installed."
    )


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


def find_matched_images(
    source_folder,
    face_encoding,
    progress_callback=None,
    matched_callback=None
):
    """
    Goes through each photo in source_folder, compares to face_encoding.
    Returns a list of full file paths that matched.

    progress_callback(i, total): for updating a progress bar
    matched_callback(filepath):   for "real-time" match handling
    """
    matched_filepaths = []
    valid_extensions = ('.png', '.jpg', '.jpeg')

    all_files = [f for f in os.listdir(source_folder) if f.lower().endswith(valid_extensions)]
    total_files = len(all_files)

    for i, filename in enumerate(all_files, start=1):
        if progress_callback:
            progress_callback(i, total_files)

        filepath = os.path.join(source_folder, filename)
        image = cv2.imread(filepath)
        if image is None:
            continue  # skip unreadable images

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_image)
        face_encodings = face_recognition.face_encodings(rgb_image, face_locations)

        for enc in face_encodings:
            match = face_recognition.compare_faces([face_encoding], enc, tolerance=0.6)[0]
            if match:
                matched_filepaths.append(filepath)

                # Real-time callback
                if matched_callback:
                    matched_callback(filepath)

                break  # Stop after first match in this image

    return matched_filepaths


class FaceFilterApp:
    def __init__(self, root):
        # root has already been created as TkinterDnD.Tk()
        self.root = root
        self.root.title("Face Filterer")
        self.root.resizable(True, True)

        # Internal variables
        self.source_folder = ""
        self.face_image_path = ""
        self.face_img_tk = None
        self.matched_files = []

        self.create_widgets()

    def create_widgets(self):
        """
        Create the main layout:
          [Left Frame]: Buttons, labels, progress bar
          [Right Frame]: Face preview (selected or matched)
        """
        main_frame = tk.Frame(self.root, padx=10, pady=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # LEFT frame
        left_frame = tk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=False)

        # RIGHT frame - we’ll let this be our drop zone (white box)
        right_frame = tk.Frame(main_frame, bg="white")
        right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # ---- Make right_frame a drop target for files/folders ----
        right_frame.drop_target_register(DND_FILES)
        right_frame.dnd_bind('<<Drop>>', self.drop_folder)

        # --------------- LEFT FRAME CONTENT ---------------
        # Create buttons with a hand cursor on hover
        btn_select_folder = tk.Button(
            left_frame, 
            text="Select Input Folder", 
            command=self.select_input_folder,
            cursor="hand2"  # Change to hand cursor on hover
        )
        btn_select_folder.pack(pady=5, fill=tk.X)

        self.label_folder = tk.Label(left_frame, text="No folder selected", fg="gray")
        self.label_folder.pack(pady=5)

        btn_select_face = tk.Button(
            left_frame, 
            text="Select Person's Face Photo", 
            command=self.select_face_image,
            cursor="hand2"  # Change to hand cursor on hover
        )
        btn_select_face.pack(pady=5, fill=tk.X)

        self.label_face = tk.Label(left_frame, text="No face image selected", fg="gray")
        self.label_face.pack(pady=5)

        btn_process = tk.Button(
            left_frame, 
            text="Process/Filter", 
            command=self.process_images,
            cursor="hand2"  # Change to hand cursor on hover
        )
        btn_process.pack(pady=10, fill=tk.X)

        self.progress_bar = ttk.Progressbar(
            left_frame, 
            orient="horizontal", 
            length=200, 
            mode="determinate"
        )
        self.progress_bar.pack(pady=5)
        self.progress_bar.pack_forget()  # hidden by default

        # --------------- RIGHT FRAME CONTENT ---------------
        # Fix the size of right_frame so large images won't auto-resize the window.
        right_frame.config(width=400, height=400)
        right_frame.pack_propagate(False)  # don't let children auto-resize it

        # This label will show either the reference face or real-time matched face
        self.face_preview_label = tk.Label(
            right_frame, 
            text="Drag & drop a folder here\nOR\nUse 'Select Input Folder'",
            bg="white"
        )
        self.face_preview_label.pack(expand=True)  # center it if bigger space

    def drop_folder(self, event):
        """
        Handle the folder that gets dropped onto the white box.
        """
        # event.data may look like: '{C:/some path/with spaces}' or multiple items
        dropped_data = event.data
        
        # If multiple items are dropped, you might do:
        # paths = dropped_data.split('} {')
        # Then pick e.g. the first path
        if dropped_data.startswith('{') and dropped_data.endswith('}'):
            dropped_data = dropped_data[1:-1]  # remove leading/trailing braces

        folder_path = dropped_data

        if os.path.isdir(folder_path):
            self.source_folder = folder_path
            # Show only the folder name in the label
            self.label_folder.config(text=os.path.basename(folder_path), fg="black")
            # Optionally show some feedback on the right frame label with only the folder name:
            self.face_preview_label.config(
                text=f"Dropped folder:\n{os.path.basename(folder_path)}"
            )
        else:
            messagebox.showwarning("Warning", "The dropped item is not a folder.")

    def select_input_folder(self):
        """
        Let the user pick a folder, defaulting to the Downloads directory.
        """
        downloads_dir = os.path.join(os.path.expanduser("~"), "Downloads")
        folder_selected = filedialog.askdirectory(
            title="Select the folder with input photos",
            initialdir=downloads_dir
        )
        if folder_selected:
            self.source_folder = folder_selected
            # Show only the folder name in the label
            self.label_folder.config(text=os.path.basename(folder_selected), fg="black")

    def select_face_image(self):
        """
        Let the user pick an image file, defaulting to the Downloads directory.
        """
        downloads_dir = os.path.join(os.path.expanduser("~"), "Downloads")
        file_selected = filedialog.askopenfilename(
            title="Select the face photo",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png")],
            initialdir=downloads_dir
        )
        if file_selected:
            self.face_image_path = file_selected
            # Show only the file name in the label
            self.label_face.config(text=os.path.basename(file_selected), fg="black")

            # Display the reference face on the right
            try:
                ref_img = Image.open(file_selected)
                # Scale it so it fits within 400x400 (to avoid resizing the window).
                ref_img.thumbnail((400, 400), Image.Resampling.LANCZOS)
                self.face_img_tk = ImageTk.PhotoImage(ref_img)
                self.face_preview_label.config(image=self.face_img_tk, text="")
            except Exception as e:
                messagebox.showerror("Error", f"Could not display image:\n{e}")

    def process_images(self):
        if not self.source_folder:
            messagebox.showwarning("Warning", "Please select an input folder (or drag & drop) first.")
            return
        if not self.face_image_path:
            messagebox.showwarning("Warning", "Please select a face photo first.")
            return

        # Show & reset progress bar
        self.progress_bar['value'] = 0
        self.progress_bar.pack()
        self.progress_bar.update()

        # Create the face encoding
        try:
            face_encoding = create_encoding_for_face(self.face_image_path)
        except Exception as e:
            messagebox.showerror("Error", str(e))
            self.progress_bar.pack_forget()
            return

        # Clear the preview label
        self.face_preview_label.config(text="Scanning for matches...", image='')

        # Prepare to store matched files
        self.matched_files = []

        def matched_callback(filepath):
            self.matched_files.append(filepath)
            self.show_matched_image(filepath)

        # Run the matching
        find_matched_images(
            self.source_folder,
            face_encoding,
            progress_callback=self.update_progress,
            matched_callback=matched_callback
        )

        self.progress_bar.pack_forget()

        # Show results
        total_matches = len(self.matched_files)
        if total_matches == 0:
            self.face_preview_label.config(text="No matches found")
            messagebox.showinfo("Results", "No matching faces found.")
            return

        # Let the user know how many we found
        messagebox.showinfo("Results", f"Found {total_matches} images matching the selected face.")

        # Ask the user for a location to save the ZIP.
        # Default directory: Downloads
        downloads_dir = os.path.join(os.path.expanduser("~"), "Downloads")
        zip_path = filedialog.asksaveasfilename(
            initialdir=downloads_dir,
            title="Save matched images as ZIP",
            defaultextension=".zip",
            filetypes=[("Zip Files", "*.zip")],
            initialfile="matched_images.zip"
        )
        if not zip_path:
            return  # user canceled

        # Create the ZIP
        try:
            with ZipFile(zip_path, 'w') as zipf:
                for filepath in self.matched_files:
                    filename = os.path.basename(filepath)
                    zipf.write(filepath, arcname=filename)

            messagebox.showinfo("Done", f"Matched images saved to:\n{zip_path}")
        except Exception as ex:
            messagebox.showerror("Error", f"Could not create ZIP file:\n{ex}")

    def update_progress(self, current, total):
        if total <= 0:
            self.progress_bar['value'] = 0
        else:
            percent = (current / total) * 100
            self.progress_bar['value'] = percent
        self.progress_bar.update()

    def show_matched_image(self, filepath):
        """
        Load the matched image as a thumbnail (max size 400x400)
        and display it in the right frame's label.
        Each new match replaces the old image in real time.
        """
        try:
            img = Image.open(filepath)
            img.thumbnail((400, 400), Image.Resampling.LANCZOS)
            matched_img_tk = ImageTk.PhotoImage(img)

            self.face_preview_label.config(image=matched_img_tk, text="")
            # Keep a reference so it's not garbage collected
            self.face_preview_label.image = matched_img_tk
        except Exception as e:
            print(f"Could not display matched image {filepath}: {e}")


def main():
    # 2) Create a TkinterDnD.Tk() instead of tk.Tk()
    root = TkinterDnD.Tk()
    root.title("Face Filterer")

    # ---- Center the GUI on the screen ----
    # You can choose whatever default width/height you prefer
    window_width = 800
    window_height = 400

    # Force Tkinter to compute window properties before geometry
    root.update_idletasks()

    # Calculate x and y coordinates for the window
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    x = (screen_width // 2) - (window_width // 2)
    y = (screen_height // 2) - (window_height // 2)

    # Position the window at (x, y) with the determined width/height
    root.geometry(f"{window_width}x{window_height}+{x}+{y}")

    app = FaceFilterApp(root)
    root.mainloop()



if __name__ == "__main__":
    main()
