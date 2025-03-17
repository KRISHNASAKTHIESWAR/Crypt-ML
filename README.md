# Face Recognition System

A real-time face recognition system using ONNX Runtime and MobileFaceNet.

## Features
- Webcam face capture
- Image upload support
- Real-time face recognition
- Face similarity comparison

## Setup
1. Clone the repository
2. Ensure you have the following files in place:
   - `models/MobileFaceNet.onnx`
   - `assets/reference_faces.json`
3. Open `index.html` in a web server

## Usage
- Use webcam capture button to take a photo
- Or upload an image using the file input
- System will compare against stored face embeddings