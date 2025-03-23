import os
import cv2
import random
import numpy as np
from tqdm import tqdm
from pathlib import Path
from ..config.configuration import Configuration

class DataIngestion:
    def __init__(self, config: Configuration):
        self.config = config
        
    def format_frames(self, frame, output_size):
        """
        Resize an image from a video to the specified output size.
        
        Args:
          frame: Image to be resized.
          output_size: Pixel size of the output frame image.
        
        Return:
          Resized frame.
        """
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert frame to RGB color space
        frame_resized = cv2.resize(frame_rgb, tuple(output_size), interpolation=cv2.INTER_LINEAR)  # Resize frame
        return frame_resized
    
    def frames_from_video_file(self, video_path, n_frames, label, output_size, frame_step):
        """
        Creates frames from a video file and assigns a label to each frame.
        
        Args:
          video_path: File path to the video.
          n_frames: Number of frames to be created per video file.
          label: Label for the video file.
          output_size: Pixel size of the output frame image.
          frame_step: Step between consecutive frames.
        
        Return:
          A list of frames extracted from the video.
        """
        # Read each video frame by frame
        result = []
        src = cv2.VideoCapture(str(video_path))
        
        video_length = src.get(cv2.CAP_PROP_FRAME_COUNT)
        need_length = min(1 + (n_frames - 1) * frame_step, video_length)
        
        start = 0
        if need_length < video_length:
            max_start = video_length - need_length
            start = random.randint(0, max_start + 1)
            
        src.set(cv2.CAP_PROP_POS_FRAMES, start)
        for _ in range(n_frames):
            ret, frame = src.read()
            if ret:
                frame = self.format_frames(frame, output_size)
                result.append(frame)
            else:
                result.append(np.zeros((output_size[1], output_size[0], 3), dtype=np.uint8))
        src.release()
        
        return result
    
    def save_frames(self, frames, labels, output_dir, video_number):
        """
        Save frames as individual image files with corresponding labels.
        
        Args:
          frames: List of frames to be saved.
          labels: List of labels corresponding to the frames.
          output_dir: Directory where frames will be saved.
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        for i, (frame, label) in enumerate(zip(frames, labels)):
            label_dir = os.path.join(output_dir, label)
            if not os.path.exists(label_dir):
                os.makedirs(label_dir)
            cv2.imwrite(os.path.join(label_dir, f"video_{video_number}_frame_{i}.jpg"), 
                        cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    
    def process_videos(self):
        """Process all videos and save frames to the output directory"""
        dataset_dir = self.config.paths.dataset_dir
        output_dir = self.config.paths.output_dir
        n_frames = self.config.video_processing.n_frames_per_video
        output_size = tuple(self.config.video_processing.output_size)
        frame_step = self.config.video_processing.frame_step
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        video_number = 1
        for label in os.listdir(dataset_dir):
            label_dir = os.path.join(dataset_dir, label)
            if not os.path.isdir(label_dir):
                continue
                
            for video_file in tqdm(os.listdir(label_dir), desc=f"Processing {label} videos"):
                video_path = os.path.join(label_dir, video_file)
                if not os.path.isfile(video_path):
                    continue
                    
                frames = self.frames_from_video_file(video_path, n_frames, label, output_size, frame_step)
                self.save_frames(frames, [label] * n_frames, output_dir, video_number)
                video_number += 1
                
        return output_dir
