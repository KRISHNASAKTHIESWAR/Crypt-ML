from encrypt import ModelSecurity
from secure_prediction import secure_predict
import numpy as np
import cv2
import os
import time
import matplotlib.pyplot as plt
from tqdm import tqdm

def test_model_efficiency(num_iterations=100):
    print("\n=== Testing Model Efficiency ===")
    
    # Initialize lists to store metrics
    inference_times = []
    encryption_times = []
    decryption_times = []
    total_times = []
    
    # Get paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "models", "MobileFaceNet.onnx")
    image_path = os.path.join(current_dir, "ref_images", "01.jpg")
    
    # Load and preprocess image once
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (112, 112))
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    input_data = np.expand_dims(img, axis=0)
    
    try:
        # Run multiple iterations
        for _ in tqdm(range(num_iterations), desc="Testing efficiency"):
            start_total = time.time()
            
            # Measure encryption time
            start_enc = time.time()
            security = ModelSecurity()
            security.encrypt_model(model_path)
            encryption_times.append(time.time() - start_enc)
            
            # Measure prediction (includes decryption) time
            start_pred = time.time()
            result = secure_predict(model_path, input_data)
            inference_times.append(time.time() - start_pred)
            
            total_times.append(time.time() - start_total)
            
        # Plot results
        plt.figure(figsize=(15, 5))
        
        # Plot 1: Time Distribution
        plt.subplot(131)
        plt.boxplot([encryption_times, inference_times, total_times], 
                   labels=['Encryption', 'Inference', 'Total'])
        plt.title('Time Distribution')
        plt.ylabel('Time (seconds)')
        
        # Plot 2: Time Series
        plt.subplot(132)
        plt.plot(encryption_times, label='Encryption')
        plt.plot(inference_times, label='Inference')
        plt.plot(total_times, label='Total')
        plt.title('Time Series Analysis')
        plt.xlabel('Iteration')
        plt.ylabel('Time (seconds)')
        plt.legend()
        
        # Plot 3: Histogram
        plt.subplot(133)
        plt.hist(total_times, bins=20)
        plt.title('Total Time Distribution')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig('efficiency_results.png')
        
        # Print summary statistics
        print("\n=== Performance Summary ===")
        print(f"Average encryption time: {np.mean(encryption_times):.3f}s ± {np.std(encryption_times):.3f}s")
        print(f"Average inference time: {np.mean(inference_times):.3f}s ± {np.std(inference_times):.3f}s")
        print(f"Average total time: {np.mean(total_times):.3f}s ± {np.std(total_times):.3f}s")
        print(f"\nResults saved to 'efficiency_results.png'")
        
    except Exception as e:
        print(f"Error during testing: {str(e)}")
        raise

if __name__ == "__main__":
    test_model_efficiency()