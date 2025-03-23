import requests
import base64
import cv2
import argparse

def predict_from_image(api_url, image_path):
    """Send an image to the prediction API"""
    # Load and prepare the image
    with open(image_path, "rb") as image_file:
        files = {"file": image_file}
        
        # Make the API request
        response = requests.post(f"{api_url}/predict", files=files)
        
        if response.status_code == 200:
            result = response.json()
            print(f"Predicted sign: {result['label']} (Confidence: {result['confidence']:.2f})")
            return result
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
            return None

def main():
    parser = argparse.ArgumentParser(description="Test Sign Language Classification API")
    parser.add_argument("--url", default="http://localhost:8000", help="API server URL")
    parser.add_argument("--image", required=True, help="Path to the image file")
    
    args = parser.parse_args()
    
    predict_from_image(args.url, args.image)

if __name__ == "__main__":
    main()
