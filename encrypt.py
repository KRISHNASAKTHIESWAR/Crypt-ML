import os
from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes
# Read the new AES key
with open("encryption_key.bin", "rb") as key_file:
    aes_key = key_file.read()

# Read the model file
with open("models/MobileFaceNet.onnx", "rb") as model_file:
    model_data = model_file.read()

# Generate a random IV (Initialization Vector)
iv = get_random_bytes(16)  # type: ignore # AES block size is 16 bytes

# Encrypt the model using AES CBC mode
cipher = AES.new(aes_key, AES.MODE_CBC, iv)
padded_model = model_data + b' ' * (16 - len(model_data) % 16)  # Padding
encrypted_model = iv + cipher.encrypt(padded_model)  # Store IV at the beginning

# Save the encrypted model
with open("MobileFaceNet_encrypted.onnx", "wb") as enc_file:
    enc_file.write(encrypted_model)

print("✅ Model successfully encrypted as 'MobileFaceNet_encrypted.onnx'")
