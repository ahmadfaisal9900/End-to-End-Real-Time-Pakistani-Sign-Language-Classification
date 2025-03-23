import logging
import argparse
import subprocess
import sys
from src.Sign_Language_Classification.pipeline.train_pipeline import TrainPipeline
from src.Sign_Language_Classification.pipeline.inference_pipeline import InferencePipeline

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s]: %(message)s:"
)

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Sign Language Classification")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "inference", "api"],
        default="inference",
        help="Mode to run: 'train' to train a new model, 'inference' for real-time inference, 'api' to start the API server (default: inference)"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host for API server (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for API server (default: 8000)"
    )
    
    args = parser.parse_args()
    
    try:
        if args.mode == "train":
            logging.info("Running in training mode...")
            pipeline = TrainPipeline()
            pipeline.run_pipeline()
        elif args.mode == "api":
            logging.info("Starting API server...")
            # Run the API server using run_api.py
            cmd = [sys.executable, "run_api.py", "--host", args.host, "--port", str(args.port)]
            subprocess.run(cmd)
        else:  # inference mode
            logging.info("Running in inference mode...")
            pipeline = InferencePipeline()
            pipeline.run_pipeline()
    except Exception as e:
        logging.exception(e)
        raise e

if __name__ == "__main__":
    main()
