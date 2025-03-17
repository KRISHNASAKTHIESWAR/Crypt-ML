import os
import binascii
import io
import json
import onnx
import numpy as np
import onnxruntime as ort
from flask import Flask, request, jsonify
from Crypto.Cipher import AES
from dotenv import load_dotenv
from PIL import Image
import base64

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

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get image from request
        data = request.json
        img_data = base64.b64decode(data["image"])

        # Convert image to tensor
        image = Image.open(io.BytesIO(img_data)).resize((112, 112)).convert("RGB")
        image_np = np.asarray(image).astype(np.float32) / 255.0  # Normalize
        image_np = np.transpose(image_np, (2, 0, 1))  # Channels first (C, H, W)
        image_np = np.expand_dims(image_np, axis=0)  # Add batch dimension

        # Run inference
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        embedding = session.run([output_name], {input_name: image_np})[0].flatten()  # Convert to 1D array

        # ✅ Step 5: Compare with Stored Embeddings
        best_match = None
        best_score = -1  # Cosine similarity range is -1 to 1

        for name, ref_embedding in reference_faces.items():
            score = cosine_similarity(embedding, np.array(ref_embedding))
            if score > best_score:  # Find the highest similarity
                best_score = score
                best_match = name

        # ✅ Step 6: Return Result
        if best_score > 0.6:  # Threshold for recognition
            return jsonify({"identity": best_match, "similarity": best_score}), 200
        else:
            return jsonify({"identity": "Unknown", "similarity": best_score}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ✅ Step 7: Register a New Face
@app.route("/register", methods=["POST"])
def register():
    try:
        data = request.json
        name = data["name"]
        img_data = base64.b64decode(data["image"])

        # Convert image to tensor
        image = Image.open(io.BytesIO(img_data)).resize((112, 112)).convert("RGB")
        image_np = np.asarray(image).astype(np.float32) / 255.0
        image_np = np.transpose(image_np, (2, 0, 1))
        image_np = np.expand_dims(image_np, axis=0)

        # Run inference to get embedding
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        embedding = session.run([output_name], {input_name: image_np})[0].flatten()

        # Store embedding
        reference_faces[name] = embedding.tolist()

        # Save to JSON file
        with open(REFERENCE_PATH, "w") as f:
            json.dump(reference_faces, f)

        return jsonify({"message": f"Face registered for {name}"}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ✅ Step 8: Run Flask Server
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
