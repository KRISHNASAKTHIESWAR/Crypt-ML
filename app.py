from flask import Flask, request, render_template, jsonify
from secure_prediction import secure_predict, generate_fernet_key
import cv2
import numpy as np
import os
import base64
import traceback
from dotenv import load_dotenv

# Initialize Flask app and load environment variables
app = Flask(__name__)
load_dotenv()

def initialize_encryption():
    """Initialize or load encryption key"""
    if 'MODEL_KEY' not in os.environ or not os.environ['MODEL_KEY']:
        key = generate_fernet_key()
        os.environ['MODEL_KEY'] = key.decode()
        # Save to .env file
        with open('.env', 'w') as f:
            f.write(f"MODEL_KEY={key.decode()}")
    return os.environ['MODEL_KEY']

def preprocess_image(image_file):
    try:
        # Read image from uploaded file
        nparr = np.frombuffer(image_file.read(), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise ValueError("Failed to decode image")
        
        # Preprocess
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

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Initialize encryption first
        key = initialize_encryption()
        print("Encryption key initialized")

        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400
        
        image_file = request.files['image']
        if not image_file.filename:
            return jsonify({'error': 'Empty file'}), 400
            
        print(f"Processing image: {image_file.filename}")
        
        # Preprocess image
        try:
            input_data = preprocess_image(image_file)
            print(f"Preprocessed image shape: {input_data.shape}")
        except Exception as e:
            return jsonify({'error': f'Image preprocessing failed: {str(e)}'}), 400
        
        # Get model path
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, "models", "MobileFaceNet.onnx")
        encrypted_path = f"{model_path}.encrypted"
        
        if not os.path.exists(model_path) and not os.path.exists(encrypted_path):
            return jsonify({'error': 'Model not found'}), 404
        
        # Run secure prediction
        result = secure_predict(model_path, input_data)
        
        if result is not None:
            embedding = result[0].tolist()
            return jsonify({
                'success': True,
                'embedding_shape': result.shape,
                'embedding': embedding[:5]
            })
        else:
            return jsonify({'error': 'Prediction failed'}), 500
            
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Ensure encryption is initialized before starting the server
    initialize_encryption()
    app.run(debug=True)