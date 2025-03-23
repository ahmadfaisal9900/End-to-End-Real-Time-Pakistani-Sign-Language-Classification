import os
import torch
from ..components.data_ingestion import DataIngestion
from ..components.data_preparation import DataPreparation
from ..components.model_trainer import ModelTrainer
from ..config.configuration import load_config
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')

class TrainPipeline:
    def __init__(self):
        self.config = load_config()
        
    def run_pipeline(self):
        """Execute the full training pipeline"""
        logging.info("Starting training pipeline")
        
        # Step 1: Process videos into frames
        logging.info("Step 1: Processing videos into frames")
        data_ingestion = DataIngestion(self.config)
        if not os.path.exists(self.config.paths.output_dir) or len(os.listdir(self.config.paths.output_dir)) == 0:
            frames_dir = data_ingestion.process_videos()
            logging.info(f"Frames extracted and saved to {frames_dir}")
        else:
            logging.info(f"Using existing frames from {self.config.paths.output_dir}")
            
        # Step 2: Prepare dataset
        logging.info("Step 2: Preparing dataset")
        data_preparation = DataPreparation(self.config)
        dataset = data_preparation.create_dataset()
        collator = data_preparation.create_data_collator()
        logging.info(f"Dataset prepared with {len(dataset)} samples")
        
        # Step 3: Train model
        logging.info("Step 3: Training model")
        trainer = ModelTrainer(self.config)
        model = trainer.train(dataset, collator)
        logging.info(f"Model trained and saved to {self.config.paths.model_output_dir}")
        
        logging.info("Training pipeline completed successfully")
        return model
