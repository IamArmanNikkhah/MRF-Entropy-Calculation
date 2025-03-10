"""
Video processing implementation for MRF-based entropy calculation.

This module provides functions to:
1. Extract frames from a video
2. Process frames and compute entropy for each tile
3. Generate output video with overlaid entropy values
4. Export entropy values to CSV
"""

import os
import cv2
import numpy as np
import csv
from tqdm import tqdm
import torch

from src.fibonacci_lattice import (
    generate_fibonacci_points,
    compute_voronoi_regions,
    map_to_erp_rectangles,
    extract_tile_from_frame,
    visualize_tiling
)
from src.mrf_model import QuadwiseMRF
from src.entropy_calculator import EntropyCalculator

class VideoProcessor:
    def __init__(self, input_path, output_path, num_tiles=55, use_gpu=True):
        """
        Initialize with video paths and parameters.
        
        Parameters:
        -----------
        input_path : str
            Path to input video file
        output_path : str
            Path to output video file
        num_tiles : int
            Number of tiles to partition the sphere into (default: 55)
        use_gpu : bool
            Whether to use GPU acceleration if available
        """
        self.input_path = input_path
        self.output_path = output_path
        self.num_tiles = num_tiles
        self.use_gpu = use_gpu and torch.cuda.is_available()
        
        # Check if input video exists
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input video file not found: {input_path}")
        
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Initialize MRF model and entropy calculator
        self.mrf_model = QuadwiseMRF()
        self.entropy_calculator = EntropyCalculator(self.mrf_model)
        
        # Initialize GPU tensors if using GPU
        if self.use_gpu:
            self.device = torch.device("cuda")
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            self.device = torch.device("cpu")
            print("Using CPU for computations")
    
    def extract_frames(self, sample_rate=1.0):
        """
        Extract frames from input video.
        
        Parameters:
        -----------
        sample_rate : float
            Fraction of frames to extract (default: 1.0)
            
        Returns:
        --------
        frames : list of ndarrays
            List of extracted video frames
        fps : float
            Frames per second of the input video
        """
        # Open video file
        cap = cv2.VideoCapture(self.input_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video file: {self.input_path}")
        
        # Get video properties
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Video properties: {width}x{height}, {fps} fps, {frame_count} frames")
        
        # Calculate number of frames to sample
        sample_count = int(frame_count * sample_rate)
        sample_indices = np.linspace(0, frame_count - 1, sample_count, dtype=int)
        
        # Extract frames
        frames = []
        for i in tqdm(range(frame_count), desc="Extracting frames"):
            ret, frame = cap.read()
            if not ret:
                break
                
            if i in sample_indices:
                frames.append(frame)
        
        cap.release()
        
        print(f"Extracted {len(frames)} frames")
        return frames, fps
    
    def generate_tiles(self, frame):
        """
        Generate tiles for a frame using Fibonacci lattice.
        
        Parameters:
        -----------
        frame : ndarray
            Video frame
            
        Returns:
        --------
        tiles : list of ndarrays
            List of tiles extracted from the frame
        tile_regions : list of dicts
            List of tile region definitions
        """
        # Get frame dimensions
        height, width = frame.shape[:2]
        
        # Generate Fibonacci lattice points
        points_3d, points_2d = generate_fibonacci_points(self.num_tiles)
        
        # Compute Voronoi regions
        sv = compute_voronoi_regions(points_3d)
        
        # Map to ERP rectangles
        tile_regions = map_to_erp_rectangles(sv, points_2d, width, height)
        
        # Extract tiles
        tiles = []
        for region in tile_regions:
            tile, mask = extract_tile_from_frame(frame, region)
            tiles.append(tile)
        
        return tiles, tile_regions
    
    def estimate_empirical_distribution(self, frames_sample):
        """
        Estimate empirical distribution from frame samples.
        
        Parameters:
        -----------
        frames_sample : list of ndarrays
            Sample of video frames
            
        Returns:
        --------
        None (updates self.mrf_model.empirical_dist)
        """
        print("Estimating empirical distribution...")
        return self.mrf_model.estimate_empirical_distribution(frames_sample)
    
    def compute_frame_entropy(self, frame, tile_regions, custom_potential=None):
        """
        Compute entropy for each tile in a frame.
        
        Parameters:
        -----------
        frame : ndarray
            Video frame
        tile_regions : list of dicts
            List of tile region definitions
        custom_potential : function, optional
            Custom smoothness function to replace the default
            
        Returns:
        --------
        entropies : list of float
            Entropy value for each tile
        """
        entropies = []
        
        for region in tile_regions:
            # Extract tile
            tile, _ = extract_tile_from_frame(frame, region)
            
            # Compute entropy
            entropy = self.entropy_calculator.compute_entropy(tile, custom_potential)
            
            # Round to 4 decimal places
            entropy = round(entropy, 4)
            
            entropies.append(entropy)
        
        return entropies
    
    def process_video(self, custom_potential=None, sample_rate=0.1):
        """
        Process the entire video.
        
        Parameters:
        -----------
        custom_potential : function, optional
            Custom smoothness function to replace the default
        sample_rate : float
            Fraction of frames to sample for empirical distribution (default: 0.1)
            
        Returns:
        --------
        frame_entropies : list of lists
            List of entropy values for each frame
        fps : float
            Frames per second of the input video
        """
        # Extract frames
        frames, fps = self.extract_frames()
        
        # Sample frames for empirical distribution
        sample_indices = np.random.choice(len(frames), int(len(frames) * sample_rate), replace=False)
        frames_sample = [frames[i] for i in sample_indices]
        
        # Estimate empirical distribution
        self.estimate_empirical_distribution(frames_sample)
        
        # Generate tiles for first frame (tile regions are the same for all frames)
        _, tile_regions = self.generate_tiles(frames[0])
        
        # Process each frame
        frame_entropies = []
        for i, frame in enumerate(tqdm(frames, desc="Processing frames")):
            # Compute entropy for each tile
            entropies = self.compute_frame_entropy(frame, tile_regions, custom_potential)
            
            # Store entropy values
            frame_entropies.append([i] + entropies)  # Add frame number at the beginning
        
        return frame_entropies, fps
    
    def export_csv(self, frame_entropies, csv_path=None):
        """
        Export per-frame entropy values to CSV.
        
        Parameters:
        -----------
        frame_entropies : list of lists
            List of entropy values for each frame
        csv_path : str, optional
            Path to output CSV file (default: same as output_path but with .csv extension)
            
        Returns:
        --------
        csv_path : str
            Path to the generated CSV file
        """
        if csv_path is None:
            csv_path = os.path.splitext(self.output_path)[0] + ".csv"
        
        # Write header
        header = ["Frame"] + [f"Tile_{i+1}_Entropy" for i in range(self.num_tiles)]
        
        # Write data
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(frame_entropies)
        
        print(f"Entropy values exported to {csv_path}")
        return csv_path
    
    def generate_output_video(self, frame_entropies, fps=30):
        """
        Generate output video with overlaid entropy values.
        
        Parameters:
        -----------
        frame_entropies : list of lists
            List of entropy values for each frame
        fps : float
            Frames per second for the output video
            
        Returns:
        --------
        output_path : str
            Path to the generated video file
        """
        # Extract frames
        frames, _ = self.extract_frames()
        
        # Generate tiles for first frame (tile regions are the same for all frames)
        _, tile_regions = self.generate_tiles(frames[0])
        
        # Get frame dimensions
        height, width = frames[0].shape[:2]
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(self.output_path, fourcc, fps, (width, height))
        
        # Process each frame
        for i, (frame, entropies) in enumerate(tqdm(zip(frames, frame_entropies), desc="Generating output video", total=len(frames))):
            # Calculate average entropy
            avg_entropy = np.mean(entropies[1:])  # Skip frame number (first element)
            
            # Create copy of frame for visualization
            vis_frame = frame.copy()
            
            # Visualize tile boundaries
            vis_frame = visualize_tiling(vis_frame, tile_regions)
            
            # Overlay average entropy
            text = f"Avg Entropy: {avg_entropy:.4f}"
            cv2.putText(vis_frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Write frame to output video
            out.write(vis_frame)
        
        # Release video writer
        out.release()
        
        print(f"Output video generated at {self.output_path}")
        return self.output_path
    
    def run(self, custom_potential=None, sample_rate=0.1):
        """
        Run the entire pipeline.
        
        Parameters:
        -----------
        custom_potential : function, optional
            Custom smoothness function to replace the default
        sample_rate : float
            Fraction of frames to sample for empirical distribution (default: 0.1)
            
        Returns:
        --------
        result : dict
            Dictionary containing paths to output files
        """
        # Process video and compute entropy
        frame_entropies, fps = self.process_video(custom_potential, sample_rate)
        
        # Export entropy values to CSV
        csv_path = self.export_csv(frame_entropies)
        
        # Generate output video
        output_path = self.generate_output_video(frame_entropies, fps)
        
        return {
            "video_path": output_path,
            "csv_path": csv_path
        }
