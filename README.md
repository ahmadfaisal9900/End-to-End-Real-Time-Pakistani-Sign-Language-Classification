# Sign Language Classification with MLOps

A comprehensive project for sign language classification with web UI and API.

## Setup and Installation

1. Clone this repository
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure your model is in the correct directory or mounted via Docker

## Running the Application Locally

### Start the API server with UI

```bash
python run_api.py
```

This will start the FastAPI server with both the API endpoints and the web UI.

### Access the Application

- Web Interface: http://localhost:8000/
- API Documentation: http://localhost:8000/docs
- API Health Check: http://localhost:8000/health

## Using the Web Interface

1. Open http://localhost:8000/ in your browser
2. Choose either "Upload Image" or "Use Webcam"
3. For image upload:
   - Drag and drop an image or click to select
   - Click "Detect Sign" to process
4. For webcam:
   - Click "Start Camera" to enable your webcam
   - The system will automatically detect signs in real-time
   - Signs will be displayed directly on the video feed
   - FPS (frames per second) is shown to monitor performance
   - Click "Stop Camera" to end the session

## Performance Considerations

- The real-time detection uses a lower image quality to improve performance
- You can adjust the detection frequency in the code if needed
- Performance will vary based on your hardware and internet connection

## Docker Deployment

### Using Pre-built Base Image

This project uses a custom PyTorch base image to speed up builds:

1. Build the base image first (only needed once):
   ```bash
   cd docker
   ./build-base-image.sh
   ```

2. Update the registry path in your main Dockerfile if needed

3. Build and run the application:
   ```bash
   docker-compose up --build
   ```

### Regular Docker Commands

```bash
# Build the Docker image
docker build -t sign-language-api .

# Run the container
docker run -p 8000:8000 sign-language-api
```

## API Endpoints

- `POST /predict` - Upload an image file for prediction
- `POST /predict/base64` - Send base64 encoded image for prediction
- `GET /health` - Check API health status
- `WS /ws/detect` - WebSocket endpoint for real-time streaming

## AWS Deployment

Please see the AWS deployment guide in the `aws-deployment` directory.
