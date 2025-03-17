import onnxruntime as ort
import numpy as np
import cv2
import json
import os

MODEL_PATH = "models/MobileFaceNet.onnx"
IMAGE_DIR = "ref_images/"
OUTPUT_FILE = "assets/reference_faces.json"

# Load ONNX Model
session = ort.InferenceSession(MODEL_PATH)
print([inp.name for inp in session.get_inputs()])

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (112, 112))
    img = img.astype(np.float32) / 255.0
    img = img.transpose(2, 0, 1)  # HWC to CHW
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Generate Embeddings for Reference Faces
embeddings = {}

for filename in os.listdir(IMAGE_DIR):
    if filename.endswith((".jpg", ".png", ".jpeg")):
        image_path = os.path.join(IMAGE_DIR, filename)
        img_tensor = preprocess_image(image_path)
        input_tensor = {session.get_inputs()[0].name: img_tensor}
        output = session.run(None, input_tensor)[0].flatten()
        embeddings[filename.split(".")[0]] = output.tolist()  # Store as list for JSON

# Save embeddings to file
with open(OUTPUT_FILE, "w") as f:
    json.dump(embeddings, f, indent=4)

print("Embeddings saved to:", OUTPUT_FILE)
