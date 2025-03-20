import os
import binascii
import io
import json
import onnx
import numpy as np
import onnxruntime as ort
from flask import Flask, request, jsonify, render_template
from Crypto.Cipher import AES
from dotenv import load_dotenv
from PIL import Image
import base64
import cv2
from io import BytesIO
from secure_prediction import secure_predict

app = Flask(__name__)

# ✅ Step 1: Load Encryption Key & IV from .env
load_dotenv()
ENCRYPTION_KEY = os.getenv("ENCRYPTION_KEY")
ENCRYPTION_IV = os.getenv("ENCRYPTION_IV")

if not ENCRYPTION_KEY or not ENCRYPTION_IV:
    raise ValueError("❌ ERROR: Missing encryption key or IV in .env file")

encryption_key = binascii.unhexlify(ENCRYPTION_KEY)
iv = binascii.unhexlify(ENCRYPTION_IV)

# ✅ Step 2: Read and Decrypt Model
with open("models/MobileFaceNet_encrypted.onnx", "rb") as enc_file:
    encrypted_data = enc_file.read()

stored_iv = encrypted_data[:16]  # First 16 bytes are IV
cipher = AES.new(encryption_key, AES.MODE_CBC, stored_iv)

decrypted_data = cipher.decrypt(encrypted_data[16:])  # Decrypt after IV
decrypted_data = decrypted_data.rstrip(b"\x00")  # Remove null padding

# ✅ Step 3: Load Model into ONNX Runtime
model_stream = io.BytesIO(decrypted_data)
model = onnx.load_model(model_stream)
session = ort.InferenceSession(model.SerializeToString())

print("✅ Model successfully decrypted and loaded into memory.")

# ✅ Step 4: Load Stored Face Embeddings
REFERENCE_PATH = "reference_faces.json"

if os.path.exists(REFERENCE_PATH):
    with open(REFERENCE_PATH, "r") as f:
        reference_faces = json.load(f)
else:
    reference_faces = {}  # Empty dictionary if no faces stored

def cosine_similarity(vec1, vec2):
    """Compute cosine similarity between two vectors."""
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def preprocess_image(image_data):
    """Preprocess image for model input"""
    # Convert base64 to image
    img = Image.open(BytesIO(base64.b64decode(image_data)))
    img = img.convert('RGB')
    
    # Convert to numpy and preprocess
    img_np = np.array(img)
    img_np = cv2.resize(img_np, (112, 112))
    img_np = img_np.astype(np.float32) / 255.0
    img_np = np.transpose(img_np, (2, 0, 1))
    img_np = np.expand_dims(img_np, axis=0)
    return img_np

@app.route('/')
def home():
    return render_template('index.html')

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get image from request
        data = request.json
        img_data = data.get('image')
        if not img_data:
            return jsonify({'error': 'No image provided'}), 400

        # Preprocess image
        input_data = preprocess_image(img_data)
        
        # Get model path
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, "models", "MobileFaceNet.onnx")
        
        # Run secure prediction
        embedding = secure_predict(model_path, input_data)
        
        if embedding is None:
            return jsonify({'error': 'Failed to generate embedding'}), 500

        # Compare with stored embeddings
        best_match = None
        best_score = -1

        for name, ref_embedding in reference_faces.items():
            score = cosine_similarity(embedding.flatten(), np.array(ref_embedding))
            if score > best_score:
                best_score = score
                best_match = name

        # Return result
        if best_score > 0.6:  # Recognition threshold
            return jsonify({
                'identity': best_match,
                'similarity': float(best_score),
                'embedding_shape': embedding.shape
            }), 200
        else:
            return jsonify({
                'identity': 'Unknown',
                'similarity': float(best_score),
                'embedding_shape': embedding.shape
            }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ✅ Step 7: Register a New Face
@app.route("/register", methods=["POST"])
def register():
    try:
        data = request.json
        name = data.get('name')
        img_data = data.get('image')
        
        if not name or not img_data:
            return jsonify({'error': 'Name and image required'}), 400

        # Preprocess image
        input_data = preprocess_image(img_data)
        
        # Get model path
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, "models", "MobileFaceNet.onnx")
        
        # Generate embedding
        embedding = secure_predict(model_path, input_data)
        
        if embedding is None:
            return jsonify({'error': 'Failed to generate embedding'}), 500

        # Store embedding
        reference_faces[name] = embedding.flatten().tolist()

        # Save to file
        with open(REFERENCE_PATH, 'w') as f:
            json.dump(reference_faces, f)

        return jsonify({'message': f'Successfully registered {name}'}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ✅ Step 8: Run Flask Server
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
