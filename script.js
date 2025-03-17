const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const statusText = document.getElementById("status");
const imageInput = document.getElementById("imageUpload");
const video = document.getElementById('video');
const captureButton = document.getElementById('capture');

let session = null;
let storedEmbeddings = {};

// Load ONNX Model
async function loadModel() {
    try {
        session = await ort.InferenceSession.create("models/MobileFaceNet.onnx");
        console.log("Model Loaded!");
        statusText.innerText = "Model Loaded!";
    } catch (error) {
        console.error("Error loading model:", error);
        statusText.innerText = "Model loading failed!";
    }
}

// Handle Image Upload
function handleImageUpload(event) {
    const file = event.target.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = function (e) {
        const img = new Image();
        img.onload = function () {
            canvas.width = 112;
            canvas.height = 112;
            ctx.drawImage(img, 0, 0, 112, 112);
            authenticateFace();
        };
        img.src = e.target.result;
    };
    reader.readAsDataURL(file);
}

// Convert Image to Tensor
function preprocessImage() {
    const imageData = ctx.getImageData(0, 0, 112, 112);
    const input = new Float32Array(3 * 112 * 112);
    let idx = 0;

    for (let i = 0; i < imageData.data.length; i += 4) {
        input[idx] = imageData.data[i] / 255.0;     // Red
        input[idx + 112 * 112] = imageData.data[i + 1] / 255.0; // Green
        input[idx + 2 * 112 * 112] = imageData.data[i + 2] / 255.0; // Blue
        idx++;
    }

    return new ort.Tensor("float32", input, [1, 3, 112, 112]);
}

// Perform Face Recognition
async function authenticateFace() {
    console.log("⏳ Starting Inference...");
    const startTime = performance.now();

    const inputTensor = preprocessImage();
    console.log("🟢 Input Tensor Shape:", inputTensor.dims);
    console.log("🟢 First 5 Values:", inputTensor.data.slice(0, 5));

    const feeds = { input0: inputTensor };
    try {
        console.log("🟢 Running ONNX Inference...");
        const results = await session.run(feeds);
        const endTime = performance.now();
        console.log(`✅ Inference Completed in ${(endTime - startTime).toFixed(2)}ms`);

        if (!results || !results.output0) {
            console.error("❌ ONNX Inference Error: Output is Undefined!");
            statusText.innerText = "Face Recognition Failed!";
            return;
        }

        const outputData = results.output0.data;
        console.log("🟢 Face Embedding:", outputData);

        // 🔥 Find Matching Face
        const match = findMatchingFace(outputData);
        if (match) {
            console.log(`✅ Face Recognized: ${match}`);
            statusText.innerText = `Face Recognized: ${match}`;
        } else {
            console.log("❌ Face Not Recognized!");
            statusText.innerText = "Face Not Recognized!";
        }
    } catch (error) {
        console.error("❌ ONNX Inference Error:", error);
        statusText.innerText = "Error in Face Recognition!";
    }
}

// Load Stored Embeddings
async function loadStoredEmbeddings() {
    try {
        const response = await fetch("assets/reference_faces.json");
        storedEmbeddings = await response.json();
        console.log("Stored Embeddings Loaded!");
    } catch (error) {
        console.error("Error loading stored embeddings:", error);
    }
}

// Find Closest Match
function normalizeEmbedding(embedding) {
    const norm = Math.sqrt(embedding.reduce((sum, val) => sum + val * val, 0));
    return embedding.map(val => val / norm);
}

function findMatchingFace(embedding) {
    embedding = normalizeEmbedding(embedding);  

    let bestMatch = null;
    let bestDistance = Infinity;

    for (const name in storedEmbeddings) {
        const knownEmbedding = normalizeEmbedding(storedEmbeddings[name]); //Normalize stored embeddings too
        const distance = cosineSimilarity(embedding, knownEmbedding);

        console.log(`Comparing with ${name}: Distance = ${distance}`);  // Debug log

        if (distance > 0.5 && distance < bestDistance) { //Adjusted threshold
            bestDistance = distance;
            bestMatch = name;
        }
    }

    return bestMatch;
}

// Cosine Similarity
function cosineSimilarity(a, b) {
    let dot = 0.0, normA = 0.0, normB = 0.0;
    for (let i = 0; i < a.length; i++) {
        dot += a[i] * b[i];
        normA += a[i] * a[i];
        normB += b[i] * b[i];
    }
    return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}

// Event Listener for Image Upload
imageInput.addEventListener("change", handleImageUpload);

// Add capture photo functionality
function capturePhoto() {
    // Draw current video frame to canvas
    canvas.width = 112;
    canvas.height = 112;
    ctx.drawImage(video, 0, 0, 112, 112);
    // Use existing authentication pipeline
    authenticateFace();
}

// Add these event listeners with your existing ones
captureButton.addEventListener('click', capturePhoto);

// Add this new function to handle webcam setup
async function setupWebcam() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        video.srcObject = stream;
    } catch (err) {
        console.error("Error accessing webcam:", err);
        statusText.innerText = "Error accessing webcam!";
    }
}

// Initialize Model and Stored Embeddings
(async () => {
    await loadModel();
    await loadStoredEmbeddings();
    await setupWebcam(); // Add this line
})();
