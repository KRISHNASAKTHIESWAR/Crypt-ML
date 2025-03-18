from encrypt import ModelSecurity
import os
from dotenv import load_dotenv
import onnxruntime as ort
import onnx
import numpy as np
import base64
import traceback

# Load environment variables
load_dotenv()

def generate_fernet_key():
    """Generate a valid Fernet key"""
    try:
        key = os.urandom(32)
        key = base64.urlsafe_b64encode(key)
        print("Generated new Fernet key successfully")
        return key
    except Exception as e:
        print(f"Error generating Fernet key: {e}")
        raise

def secure_predict(model_path, input_data):
    temp_path = None
    try:
        print("Starting secure prediction...")
        
        # Check if we have an encryption key in environment
        if 'MODEL_KEY' in os.environ and os.environ['MODEL_KEY']:
            key = os.environ['MODEL_KEY'].encode()
            print("Using existing encryption key")
        else:
            # If no key exists, we need to encrypt the model first
            key = generate_fernet_key()
            os.environ['MODEL_KEY'] = key.decode()
            print("Generated new encryption key")
            
        # Initialize security with the key
        security = ModelSecurity(key)
        
        # Generate temporary access token
        token = security.generate_access_token(duration=300)  # 5 minutes
        print("Access token generated")
        
        if security.verify_access(token):
            print("Access verified successfully")
            
            # Check if model needs encryption
            if not model_path.endswith('.encrypted'):
                original_path = model_path
                model_path = f"{original_path}.encrypted"
                if not os.path.exists(model_path):
                    print("Encrypting model for first use...")
                    security.encrypt_model(original_path)
                    print("Model encrypted successfully")
            
            # Create temporary path for decrypted model
            temp_path = f"temp_{os.urandom(8).hex()}.onnx"
            
            try:
                # Load and decrypt model
                print("Loading encrypted model...")
                model = security.decrypt_model(model_path, key)
                print("Model decrypted successfully")
                
                # Save temporarily for inference
                onnx.save(model, temp_path)
                
                # Create ONNX Runtime session
                print("Creating ONNX Runtime session...")
                session = ort.InferenceSession(temp_path, providers=['CPUExecutionProvider'])
                
                # Get input name and run inference
                input_name = session.get_inputs()[0].name
                print(f"Running inference with input shape: {input_data.shape}")
                prediction = session.run(None, {input_name: input_data})[0]
                
                return prediction
                
            finally:
                # Clean up temporary model file
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                    print("Cleaned up temporary model file")
                
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        traceback.print_exc()
        return None
    finally:
        # Additional cleanup
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)