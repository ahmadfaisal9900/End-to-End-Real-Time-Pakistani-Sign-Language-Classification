import os
import yaml
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

@dataclass
class PathConfig:
    dataset_dir: str
    output_dir: str
    model_output_dir: str

@dataclass
class VideoProcessingConfig:
    n_frames_per_video: int
    output_size: List[int]
    frame_step: int

@dataclass
class ImageTransformConfig:
    resize_dimensions: List[int]

@dataclass
class DatasetConfig:
    labels: List[str]

@dataclass
class ModelConfig:
    model_name_or_path: str
    num_classes: int

@dataclass
class TrainingConfig:
    batch_size: int
    epochs: int
    learning_rate: float
    evaluation_strategy: str
    eval_steps: int
    save_steps: int
    logging_steps: int
    save_total_limit: int
    load_best_model_at_end: bool
    metric_for_best_model: str

@dataclass
class Configuration:
    paths: PathConfig
    video_processing: VideoProcessingConfig
    image_transform: ImageTransformConfig
    dataset: DatasetConfig
    model: ModelConfig
    training: TrainingConfig

def load_config():
    config_path = Path("params.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    return Configuration(
        paths=PathConfig(**config["paths"]),
        video_processing=VideoProcessingConfig(**config["video_processing"]),
        image_transform=ImageTransformConfig(**config["image_transform"]),
        dataset=DatasetConfig(**config["dataset"]),
        model=ModelConfig(**config["model"]),
        training=TrainingConfig(**config["training"])
    )