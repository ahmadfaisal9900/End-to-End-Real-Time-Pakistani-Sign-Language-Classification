import os
import yaml
import cv2
import torch
import numpy as np
import transformers
from transformers import AutoImageProcessor, AutoModelForImageClassification
from ..config.configuration import Configuration, load_config

class SignLanguageInference:
    def __init__(self, model_path=None):
        """
        Initialize the inference component
        
        Args:
            model_path: Path to the trained model, if None use path from config
        """
        # Load configuration
        with open('params.yaml', 'r') as f:
            self.config = yaml.safe_load(f)
            
        # Use environment variable for model path if available
        env_model_dir = os.environ.get('MODEL_DIR')
        
        if model_path:
            self.model_path = model_path
        elif env_model_dir and os.path.exists(env_model_dir):
            # Use the container mounted model path
            self.model_path = env_model_dir
            print(f"Using model from mounted directory: {self.model_path}")
        else:
            # Use the default path from params.yaml
            self.model_path = self.config['paths']['model_output_dir']
            print(f"Using model from default path: {self.model_path}")
        
        # Load model components
        try:
            self.image_processor = AutoImageProcessor.from_pretrained(self.model_path)
            self.model = AutoModelForImageClassification.from_pretrained(self.model_path)
            self.labels = self.config['dataset']['labels']
            print(f"Model loaded successfully from {self.model_path}")
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise
        
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
    
    def get_sign_language_label(self, class_index):
        """Map class index to label"""
        return self.labels[class_index]
    
    def predict_single_image(self, image):
        """
        Predict sign language from a single image
        
        Args:
            image: Input image (OpenCV format)
            
        Returns:
            predicted label, confidence score
        """
        # Preprocess the image
        inputs = self.image_processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Perform inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
        
        # Get prediction and confidence
        probabilities = torch.nn.functional.softmax(logits, dim=-1)[0]
        predicted_class = torch.argmax(logits, dim=-1).item()
        confidence = probabilities[predicted_class].item()
        
        # Get the label
        sign_language_label = self.get_sign_language_label(predicted_class)
        
        return sign_language_label, confidence
    
    def run_realtime_inference(self):
        """Run real-time inference using webcam"""
        # Define the video capture
        cap = cv2.VideoCapture(0)  # You may change the parameter to the appropriate device index
        
        if not cap.isOpened():
            print("Error: Could not open webcam.")
            return
            
        print("Starting real-time sign language detection...")
        print("Press 'q' to quit")
        
        # Set up real-time inference
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Perform inference
            sign_language_label, confidence = self.predict_single_image(frame)
            
            # Display the results
            display_text = f"{sign_language_label} ({confidence:.2f})"
            cv2.putText(frame, display_text, (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow('Real-time Sign Language Detection', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
                break
        
        # Release the capture
        cap.release()
        cv2.destroyAllWindows()
