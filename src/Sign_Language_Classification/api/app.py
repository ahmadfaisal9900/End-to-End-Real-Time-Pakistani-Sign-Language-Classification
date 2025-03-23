import io
import cv2
import base64
import numpy as np
import time
from fastapi import FastAPI, File, UploadFile, HTTPException, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, List

from ..components.model_inference import SignLanguageInference

# Initialize the model
inference_model = SignLanguageInference()

# Create FastAPI application
app = FastAPI(
    title="Sign Language Classification API",
    description="API for predicting sign language gestures from images",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Add response_time middleware
@app.middleware("http")
async def add_response_time(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

class PredictionResponse(BaseModel):
    """Response model for predictions"""
    label: str
    confidence: float

class Base64ImageRequest(BaseModel):
    """Request model for base64 encoded images"""
    image: str
    content_type: Optional[str] = "image/jpeg"

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Serve the frontend application"""
    # Redirect to the static index.html
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta http-equiv="refresh" content="0;url=/static/index.html">
        <title>Redirecting...</title>
    </head>
    <body>
        <p>Redirecting to the application...</p>
    </body>
    </html>
    """
    return html_content

@app.get("/api", response_class=JSONResponse)
def api_info():
    """Return API information in JSON format"""
    return {
        "message": "Sign Language Classification API",
        "version": "1.0.0",
        "endpoints": {
            "/predict": "Upload an image file to get prediction",
            "/predict/base64": "Send base64 encoded image for prediction",
            "/health": "Health check endpoint",
            "/docs": "Swagger UI API documentation",
            "/redoc": "ReDoc API documentation"
        }
    }

# Optimize predict endpoint for faster response in real-time scenarios
@app.post("/predict", response_model=PredictionResponse)
async def predict_from_file(file: UploadFile = File(...)):
    """Predict sign language from uploaded image file"""
    try:
        # Read and decode image more efficiently
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        
        # Use IMREAD_REDUCED_COLOR_8 for faster processing in real-time mode
        # This will reduce the image resolution to 1/8 of original
        if file.headers.get("X-Real-Time") == "true":
            image = cv2.imdecode(nparr, cv2.IMREAD_REDUCED_COLOR_8)
        else:
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Make prediction
        label, confidence = inference_model.predict_single_image(image)
        
        # Return prediction
        return PredictionResponse(label=label, confidence=confidence)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict/base64", response_model=PredictionResponse)
async def predict_from_base64(request: Base64ImageRequest):
    """Predict sign language from base64 encoded image"""
    try:
        # Decode base64 image
        image_data = base64.b64decode(request.image)
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid base64 image")
        
        # Make prediction
        label, confidence = inference_model.predict_single_image(image)
        
        # Return prediction
        return PredictionResponse(label=label, confidence=confidence)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# Add websocket endpoint for real-time streaming
@app.websocket("/ws/detect")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_bytes()
            nparr = np.frombuffer(data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_REDUCED_COLOR_8)
            
            if image is not None:
                label, confidence = inference_model.predict_single_image(image)
                await websocket.send_json({"label": label, "confidence": confidence})
    except WebSocketDisconnect:
        pass
    except Exception as e:
        await websocket.close(code=1011, reason=str(e))

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}
