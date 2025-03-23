import os
import logging
from ..components.data_ingestion import DataIngestion
from ..config.configuration import load_config

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')

def main():
    """Data ingestion pipeline"""
    config = load_config()
    
    logging.info("Data ingestion started")
    data_ingestion = DataIngestion(config)
    frames_dir = data_ingestion.process_videos()
    logging.info(f"Data ingestion completed. Frames saved to: {frames_dir}")
    
    return frames_dir

if __name__ == "__main__":
    main()
