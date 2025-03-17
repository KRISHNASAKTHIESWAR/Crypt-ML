# from Crypto.Cipher import AES
# from Crypto.Random import get_random_bytes

# # Generate a new 256-bit AES key
# aes_key = get_random_bytes(32)  # 32 bytes = 256 bits

# # Save the key securely
# with open("encryption_key.bin", "wb") as key_file:
#     key_file.write(aes_key)

# print("✅ New encryption key generated and saved as 'encryption_key.bin'")


#enc key
# with open("encryption_key.bin", "rb") as key_file:
#     encryption_key = key_file.read()

# print("Encryption Key:", encryption_key.hex())  # Print in hex format for easy reading

#IV key
with open("models/MobileFaceNet_encrypted.onnx", "rb") as enc_file:
    encrypted_data = enc_file.read()

iv = encrypted_data[:16]  # First 16 bytes contain the IV

print("IV Key:", iv.hex())  # Print in hex format for easy reading
