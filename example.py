from secure_prediction import secure_predict
import numpy as np
import cv2
import os

def preprocess_image(image_path):
    # Verify image path exists
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at: {image_path}")
    
    # Read and preprocess image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")
        
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (112, 112))
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return img

try:
    # Get absolute paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(current_dir, "ref_images", "01.jpg")
    model_path = os.path.join(current_dir, "models", "MobileFaceNet.onnx")  # Remove .encrypted extension
    
    print(f"Testing secure model inference...")
    
    # Load and preprocess image
    input_data = preprocess_image(image_path)
    
    # Run secure prediction
    result = secure_predict(model_path, input_data)
    
    if result is not None:
        print(f"Success! Face embedding shape: {result.shape}")
        print(f"First few values: {result[0][:5]}")
    else:
        print("Prediction failed")

except FileNotFoundError as e:
    print(f"File Error: {str(e)}")
except Exception as e:
    print(f"Error: {str(e)}")