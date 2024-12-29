# FaceFilter



## Overview
This is a Python application that filters and organizes photos based on facial recognition. It provides a user-friendly GUI for selecting photos containing specific faces.

## Features
- Drag-and-drop folder selection
- Face recognition-based photo filtering
- Progress tracking during processing
- Automatic ZIP file creation of matched photos
- Real-time preview of matched images

## Requirements
- Python 3.x
- OpenCV
- face_recognition
- numpy
- tkinterdnd2
- Pillow

## Installation
Create a virtual environment and install dependencies:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install opencv-python face_recognition numpy tkinterdnd2 pillow
```

## Usage
1. Run the GUI application:
```bash
python gui_app.py
```
2. Select or drag a folder containing photos
3. Choose a reference face photo
4. Click "Process/Filter" to find matches
5. Save matched photos as ZIP

## Scripts
- `gui_app.py`: Main GUI application
- `encodegenerator.py`: Generate face encodings
- `separate_images.py`: Batch process images