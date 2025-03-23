import os
import logging
from ..components.model_inference import SignLanguageInference
from ..config.configuration import load_config

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')

class InferencePipeline:
    def __init__(self):
        self.config = load_config()
        
    def run_pipeline(self):
        """Execute the inference pipeline"""
        logging.info("Starting inference pipeline")
        
        model_path = self.config.paths.model_output_dir
        
        # Check if model exists
        if not os.path.exists(model_path):
            logging.error(f"Model not found at {model_path}. Please train the model first.")
            return False
            
        # Initialize the inference component
        inference = SignLanguageInference(model_path)
        
        # Run real-time inference
        logging.info("Running real-time inference...")
        inference.run_realtime_inference()
        
        logging.info("Inference pipeline completed")
        return True

def main():
    """Entry point for inference"""
    try:
        pipeline = InferencePipeline()
        pipeline.run_pipeline()
    except Exception as e:
        logging.exception(e)
        raise e

if __name__ == "__main__":
    main()
