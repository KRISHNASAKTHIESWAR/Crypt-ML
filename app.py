from flask import Flask, request, render_template, jsonify
from secure_prediction import secure_predict, generate_fernet_key
import cv2
import numpy as np
import os
import base64
import traceback
from dotenv import load_dotenv
import json

# Initialize Flask app and load environment variables
app = Flask(__name__)
load_dotenv()

# Example: Update initialize_encryption() to write .env file
def initialize_encryption():
    """Initialize or load encryption key"""
    if 'MODEL_KEY' not in os.environ or not os.environ['MODEL_KEY']:
        key = generate_fernet_key()
        os.environ['MODEL_KEY'] = key.decode()
        # Write the key into a .env file
        with open('.env', 'w') as f:
            f.write(f"MODEL_KEY={key.decode()}")
        print("Generated new encryption key and .env file created")
    else:
        print("Using existing encryption key")
    return os.environ['MODEL_KEY']

def preprocess_image(image_file):
    try:
        # Read image from uploaded file (use read() then convert using cv2.imdecode)
        file_bytes = image_file.read()
        nparr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise ValueError("Failed to decode image")
        
        # Preprocess: convert BGR->RGB, resize, normalize and add batch dimension.
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (112, 112))
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        return img
    except Exception as e:
        print(f"Image preprocessing error: {str(e)}")
        traceback.print_exc()
        raise

def load_reference_embeddings():
    """Load stored face embeddings from file (embeddings.json)"""
    REFERENCE_PATH = "reference_faces.json"
    if os.path.exists(REFERENCE_PATH):
        with open(REFERENCE_PATH, 'r') as f:
            return json.load(f)
    else:
        return {}

def cosine_similarity(vec1, vec2):
    """Compute cosine similarity between two vectors"""
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Initialize encryption key; ensures model encryption is used.
        key = initialize_encryption()
        print("Encryption key initialized")
        
        # Verify image file exists in the form-data
        if 'image' not in request.files:
            print("Predict: No image uploaded")
            return jsonify({'error': 'No image uploaded'}), 400
        image_file = request.files['image']
        if not image_file.filename:
            print("Predict: Empty file uploaded")
            return jsonify({'error': 'Empty file'}), 400
            
        print(f"Processing image: {image_file.filename}")
        
        # Preprocess image for model input
        try:
            input_data = preprocess_image(image_file)
            print(f"Preprocessed image shape: {input_data.shape}")
        except Exception as e:
            print("Error in image preprocessing")
            return jsonify({'error': f'Image preprocessing failed: {str(e)}'}), 400
        
        # Set model path (here assuming unencrypted model file exists; secure_predict handles encryption internally)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, "models", "MobileFaceNet.onnx")
        encrypted_path = f"{model_path}.encrypted"
        
        if not os.path.exists(model_path) and not os.path.exists(encrypted_path):
            print("Predict: Model file not found")
            return jsonify({'error': 'Model not found'}), 404
        
        # Run secure prediction (decrypts model internally and performs inference)
        result = secure_predict(model_path, input_data)
        if result is None:
            print("Predict: secure_predict returned None")
            return jsonify({'error': 'Prediction failed'}), 500
        
        # Obtain the current face embedding from the result
        current_embedding = result[0].tolist()
        
        # Load reference embeddings (for face registrations)
        reference_embeddings = load_reference_embeddings()
        best_match = None
        best_score = -1
        
        # Compare against every registered embedding and print the similarity score
        for name, ref_embedding in reference_embeddings.items():
            score = cosine_similarity(current_embedding, ref_embedding)
            print(f"Comparing with {name}: similarity = {score}")
            if score > best_score:
                best_score = score
                best_match = name
        
        # Define a threshold for recognition; if similarity exceeds threshold, grant access.
        threshold = 0.6  # Adjust threshold value if needed
        access_granted = best_score > threshold
        
        print(f"Best match: {best_match} with score {best_score}")
        print(f"Access {'Granted' if access_granted else 'Denied'} (Threshold: {threshold})")
        
        return jsonify({
            'success': True,
            'access': "Granted" if access_granted else "Denied",
            'identity': best_match if access_granted else "Unknown",
            'confidence': float(best_score),
            'embedding_shape': result.shape
        })
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500
@app.route('/register', methods=['POST'])
def register():
    try:
        # Check that face identifier and image are provided
        if 'image' not in request.files or 'face_id' not in request.form:
            return jsonify({'error': 'Face ID and image required'}), 400
        
        face_id = request.form.get('face_id')
        image_file = request.files['image']
        if not image_file.filename:
            return jsonify({'error': 'Empty file uploaded'}), 400

        # Preprocess image
        input_data = preprocess_image(image_file)
        print(f"Preprocessed image shape: {input_data.shape}")

        # Set model path and run secure prediction (this decrypts the model internally)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, "models", "MobileFaceNet.onnx")
        if not os.path.exists(model_path):
            return jsonify({'error': 'Model not found'}), 404

        result = secure_predict(model_path, input_data)
        if result is None:
            return jsonify({'error': 'Prediction failed'}), 500

        # Convert embedding to a plain list
        current_embedding = result[0].tolist()

        # Load existing reference embeddings (or create new dict if not present)
        REFERENCE_PATH = "reference_faces.json"
        if os.path.exists(REFERENCE_PATH):
            with open(REFERENCE_PATH, 'r') as f:
                reference_embeddings = json.load(f)
        else:
            reference_embeddings = {}

        # Save the new embedding under the provided face_id
        reference_embeddings[face_id] = current_embedding

        with open(REFERENCE_PATH, 'w') as f:
            json.dump(reference_embeddings, f, indent=4)

        print(f"Registered face ID {face_id}")
        return jsonify({'message': f'Face {face_id} registered successfully'})
    except Exception as e:
        print(f"Registration error: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Ensure encryption is initialized before starting the server
    initialize_encryption()
    app.run(debug=True)