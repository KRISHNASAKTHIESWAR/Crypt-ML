from cryptography.fernet import Fernet
import onnx
import os
import hashlib
import time
from datetime import datetime

class ModelSecurity:
    def __init__(self, key=None):
        if key is None:
            self.key = Fernet.generate_key()
        else:
            self.key = key
        self.cipher = Fernet(self.key)
        self.access_token = None
        self.token_expiry = None
        self.access_log = []
    
    def encrypt_model(self, model_path):
        # Load ONNX model and convert to bytes
        with open(model_path, 'rb') as f:
            model_bytes = f.read()
        
        # Encrypt the model bytes
        encrypted_model = self.cipher.encrypt(model_bytes)
        
        # Save encrypted model
        with open(f"{model_path}.encrypted", 'wb') as f:
            f.write(encrypted_model)
        
        return self.key
    
    def decrypt_model(self, encrypted_model_path, key):
        self.cipher = Fernet(key)
        
        # Read encrypted model
        with open(encrypted_model_path, 'rb') as f:
            encrypted_model = f.read()
            
        # Decrypt model bytes
        model_bytes = self.cipher.decrypt(encrypted_model)
        
        # Create temporary file for ONNX model
        temp_path = f"temp_{os.urandom(8).hex()}.onnx"
        with open(temp_path, 'wb') as f:
            f.write(model_bytes)
        
        # Load ONNX model
        model = onnx.load("models/MobileFaceNet.onnx")
        
        # Clean up temporary file
        os.remove(temp_path)
        
        return model

    def generate_access_token(self, duration=3600):
        """Generate temporary access token with expiry"""
        self.access_token = hashlib.sha256(os.urandom(32)).hexdigest()
        self.token_expiry = time.time() + duration
        self.log_access("Token generated")
        return self.access_token
    
    def verify_access(self, token):
        """Verify if access token is valid"""
        if token != self.access_token:
            self.log_access("Invalid token attempt")
            raise ValueError("Invalid access token")
        if time.time() > self.token_expiry:
            self.log_access("Token expired")
            raise ValueError("Token expired")
        self.log_access("Valid access")
        return True
    
    def log_access(self, event):
        """Log access attempts"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.access_log.append(f"{timestamp}: {event}")