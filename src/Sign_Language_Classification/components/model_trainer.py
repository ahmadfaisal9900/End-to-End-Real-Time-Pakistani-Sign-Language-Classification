import torch
import numpy as np
from transformers import (
    ViTForImageClassification,
    ViTImageProcessor, 
    TrainingArguments,
    Trainer
)
from datasets import load_metric
from ..config.configuration import Configuration

class ModelTrainer:
    def __init__(self, config: Configuration):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.metric = load_metric("accuracy")
        
    def get_model(self):
        """Initialize the ViT model"""
        model_name = self.config.model.model_name_or_path
        num_classes = self.config.model.num_classes
        
        model = ViTForImageClassification.from_pretrained(
            model_name, 
            num_labels=num_classes
        )
        return model.to(self.device)
    
    def get_processor(self):
        """Initialize the ViT feature extractor"""
        model_name = self.config.model.model_name_or_path
        return ViTImageProcessor.from_pretrained(model_name)
    
    def compute_metrics(self, eval_pred):
        """Compute accuracy metrics"""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return self.metric.compute(predictions=predictions, references=labels)
    
    def get_training_args(self):
        """Setup training arguments"""
        training_config = self.config.training
        output_dir = self.config.paths.model_output_dir
        
        return TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=training_config.batch_size,
            num_train_epochs=training_config.epochs,
            evaluation_strategy=training_config.evaluation_strategy,
            eval_steps=training_config.eval_steps,
            save_steps=training_config.save_steps,
            logging_steps=training_config.logging_steps,
            learning_rate=training_config.learning_rate,
            save_total_limit=training_config.save_total_limit,
            load_best_model_at_end=training_config.load_best_model_at_end,
            metric_for_best_model=training_config.metric_for_best_model,
            remove_unused_columns=False,
            disable_tqdm=False,
            push_to_hub=False,
        )
    
    def train(self, train_dataset, data_collator):
        """Train the model"""
        model = self.get_model()
        processor = self.get_processor()
        training_args = self.get_training_args()
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=train_dataset,  # Using same dataset for evaluation
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
            tokenizer=processor
        )
        
        train_results = trainer.train()
        trainer.save_model(self.config.paths.model_output_dir)
        trainer.log_metrics("train", train_results.metrics)
        trainer.save_metrics("train", train_results.metrics)
        trainer.save_state()
        
        return model
