document.addEventListener('DOMContentLoaded', function() {
    // DOM Elements
    const tabButtons = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');
    const fileInput = document.getElementById('fileInput');
    const preview = document.getElementById('preview');
    const uploadDetectBtn = document.getElementById('upload-detect-btn');
    const startWebcamBtn = document.getElementById('start-webcam');
    const webcam = document.getElementById('webcam');
    const webcamCanvas = document.getElementById('webcam-canvas');
    const detectionContainer = document.querySelector('.detection-container');
    const resultsContainer = document.getElementById('results');
    const resultImage = document.getElementById('result-image');
    const detectedLabel = document.getElementById('detected-label');
    const confidenceScore = document.getElementById('confidence-score');
    const confidenceFill = document.getElementById('confidence-fill');
    const backBtn = document.getElementById('back-btn');
    const loading = document.getElementById('loading');
    const dropArea = document.getElementById('drop-area');
    const fpsCounter = document.getElementById('fps-counter');

    // Variables
    let selectedImage = null;
    let stream = null;
    let lastFrameTime = 0;
    let frameCount = 0;
    let lastFpsUpdateTime = 0;
    let realTimeIntervalId = null;
    let processingFrame = false;

    // API URL - Change this to match your API endpoint
    const apiUrl = window.location.origin; // e.g., http://localhost:8000

    // Tab Functionality
    tabButtons.forEach(button => {
        button.addEventListener('click', () => {
            // Remove active class from all buttons and content
            tabButtons.forEach(btn => btn.classList.remove('active'));
            tabContents.forEach(content => content.classList.remove('active'));
            
            // Add active class to clicked button and corresponding content
            button.classList.add('active');
            const tabId = button.getAttribute('data-tab');
            document.getElementById(`${tabId}-tab`).classList.add('active');
            
            // Reset webcam when switching away from webcam tab
            if (tabId !== 'webcam' && stream) {
                stopWebcam();
            }
        });
    });

    // Drag and drop functionality
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        dropArea.addEventListener(eventName, () => {
            dropArea.classList.add('highlight');
        });
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, () => {
            dropArea.classList.remove('highlight');
        });
    });

    dropArea.addEventListener('drop', handleDrop);

    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        
        if (files.length) {
            handleFile(files[0]);
        }
    }

    // File Upload Functionality
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length) {
            handleFile(e.target.files[0]);
        }
    });

    function handleFile(file) {
        if (!file.type.match('image.*')) {
            alert('Please select an image file');
            return;
        }

        selectedImage = file;
        const reader = new FileReader();
        
        reader.onload = (e) => {
            preview.src = e.target.result;
            preview.classList.remove('hidden');
            uploadDetectBtn.disabled = false;
        };
        
        reader.readAsDataURL(file);
    }

    // Upload and Detect
    uploadDetectBtn.addEventListener('click', async () => {
        if (!selectedImage) return;
        
        showLoading();
        
        const formData = new FormData();
        formData.append('file', selectedImage);
        
        try {
            const response = await fetch(`${apiUrl}/predict`, {
                method: 'POST',
                body: formData
            });
            
            if (!response.ok) {
                throw new Error(`Error: ${response.status}`);
            }
            
            const result = await response.json();
            showResults(preview.src, result);
        } catch (error) {
            console.error('Error:', error);
            alert('Detection failed. Please try again.');
        } finally {
            hideLoading();
        }
    });

    // Webcam Functionality
    startWebcamBtn.addEventListener('click', () => {
        if (stream) {
            stopWebcam();
            startWebcamBtn.textContent = 'Start Camera';
            
            // Stop real-time processing
            stopRealTimeDetection();
        } else {
            startWebcam();
            startWebcamBtn.textContent = 'Stop Camera';
        }
    });

    async function startWebcam() {
        try {
            stream = await navigator.mediaDevices.getUserMedia({
                video: true,
                audio: false
            });
            webcam.srcObject = stream;
            
            // Automatically start real-time detection when camera starts
            startRealTimeDetection();
        } catch (error) {
            console.error('Error accessing webcam:', error);
            alert('Could not access webcam. Please ensure you have given permission.');
        }
    }

    function stopWebcam() {
        if (stream) {
            stream.getTracks().forEach(track => track.stop());
            stream = null;
            webcam.srcObject = null;
            
            // Reset any real-time elements
            if (fpsCounter) {
                fpsCounter.textContent = '0 FPS';
            }
        }
    }

    // Start real-time detection
    function startRealTimeDetection() {
        if (!stream) return;
        
        // Reset FPS counters
        frameCount = 0;
        lastFrameTime = performance.now();
        lastFpsUpdateTime = performance.now();
        
        // Create an overlay for real-time results
        createRealTimeOverlay();
        
        // Process frames at regular intervals - adjust for desired performance/accuracy balance
        // Lower interval = more frequent updates but potentially more resource usage
        realTimeIntervalId = setInterval(processRealTimeFrame, 100); // 10 FPS
    }

    // Stop real-time detection
    function stopRealTimeDetection() {
        if (realTimeIntervalId) {
            clearInterval(realTimeIntervalId);
            realTimeIntervalId = null;
        }
        
        // Remove real-time overlay if it exists
        const overlay = document.getElementById('realtime-result-overlay');
        if (overlay) {
            overlay.parentNode.removeChild(overlay);
        }
    }

    // Create overlay for real-time results
    function createRealTimeOverlay() {
        // Remove existing overlay if any
        const existingOverlay = document.getElementById('realtime-result-overlay');
        if (existingOverlay) {
            existingOverlay.parentNode.removeChild(existingOverlay);
        }
        
        // Create new overlay
        const overlay = document.createElement('div');
        overlay.id = 'realtime-result-overlay';
        overlay.className = 'realtime-overlay';
        
        // Create label element
        const label = document.createElement('div');
        label.id = 'realtime-label';
        label.className = 'realtime-label';
        label.textContent = 'Detecting...';
        overlay.appendChild(label);
        
        // Create FPS counter
        const fpsElement = document.createElement('div');
        fpsElement.id = 'realtime-fps';
        fpsElement.className = 'realtime-fps';
        fpsElement.textContent = '0 FPS';
        overlay.appendChild(fpsElement);
        
        // Add to webcam container
        const webcamContainer = document.querySelector('.webcam-container');
        webcamContainer.appendChild(overlay);
    }

    // Process a frame for real-time detection
    async function processRealTimeFrame() {
        if (!stream || processingFrame) return;
        
        processingFrame = true;
        
        try {
            // Capture frame from webcam
            const canvas = document.createElement('canvas');
            canvas.width = webcam.videoWidth;
            canvas.height = webcam.videoHeight;
            const ctx = canvas.getContext('2d');
            ctx.drawImage(webcam, 0, 0, canvas.width, canvas.height);
            
            // Convert to blob
            const blob = await new Promise(resolve => {
                canvas.toBlob(blob => resolve(blob), 'image/jpeg', 0.7); // Lower quality for better performance
            });
            
            // Send to API with real-time flag
            const formData = new FormData();
            formData.append('file', blob);
            
            const response = await fetch(`${apiUrl}/predict`, {
                method: 'POST',
                headers: {
                    'X-Real-Time': 'true'
                },
                body: formData
            });
            
            if (response.ok) {
                const result = await response.json();
                updateRealTimeResults(result);
                
                // Update FPS
                frameCount++;
                const now = performance.now();
                if (now - lastFpsUpdateTime > 1000) { // Update FPS every second
                    const fps = Math.round((frameCount * 1000) / (now - lastFpsUpdateTime));
                    updateFpsDisplay(fps);
                    frameCount = 0;
                    lastFpsUpdateTime = now;
                }
            }
        } catch (error) {
            console.error('Real-time detection error:', error);
        } finally {
            processingFrame = false;
        }
    }
    
    // Update FPS display
    function updateFpsDisplay(fps) {
        // Update overlay FPS counter
        const fpsElement = document.getElementById('realtime-fps');
        if (fpsElement) {
            fpsElement.textContent = `${fps} FPS`;
        }
        
        // Update main FPS counter
        if (fpsCounter) {
            fpsCounter.textContent = `${fps} FPS`;
        }
    }

    // Update real-time results on the overlay
    function updateRealTimeResults(result) {
        const labelElement = document.getElementById('realtime-label');
        if (labelElement) {
            const confidence = Math.round(result.confidence * 100);
            labelElement.textContent = `${result.label} (${confidence}%)`;
            
            // Change color based on confidence
            if (confidence >= 80) {
                labelElement.className = 'realtime-label high-confidence';
            } else if (confidence >= 50) {
                labelElement.className = 'realtime-label medium-confidence';
            } else {
                labelElement.className = 'realtime-label low-confidence';
            }
        }
    }

    // Show Results
    function showResults(imageUrl, result) {
        detectionContainer.classList.add('hidden');
        resultsContainer.classList.remove('hidden');
        
        resultImage.src = imageUrl;
        detectedLabel.textContent = result.label;
        
        const confidence = Math.round(result.confidence * 100);
        confidenceScore.textContent = `${confidence}%`;
        confidenceFill.style.width = `${confidence}%`;
        
        // Set color based on confidence
        if (confidence >= 80) {
            confidenceFill.style.backgroundColor = 'var(--success-color)';
        } else if (confidence >= 50) {
            confidenceFill.style.backgroundColor = 'orange';
        } else {
            confidenceFill.style.backgroundColor = 'var(--danger-color)';
        }
    }

    // Back Button
    backBtn.addEventListener('click', () => {
        resultsContainer.classList.add('hidden');
        detectionContainer.classList.remove('hidden');
    });

    // Loading State
    function showLoading() {
        loading.classList.remove('hidden');
    }

    function hideLoading() {
        loading.classList.add('hidden');
    }
});
