"""
Video processing implementation for MRF-based entropy calculation.

This module provides functions to:
1. Extract frames from a video
2. Process frames and compute entropy for each tile
3. Generate output video with overlaid entropy values
4. Export entropy values to CSV

Optimized for GPU acceleration and parallel processing.
"""

import os
import cv2
import numpy as np
import csv
from tqdm import tqdm
import torch
import time
import concurrent.futures
from functools import partial
import gc

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
    def __init__(self, input_path, output_path, num_tiles=55, use_gpu=True, batch_size=8, gpu_memory_fraction=0.7):
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
        batch_size : int
            Number of frames to process in a batch (default: 8)
        gpu_memory_fraction : float
            Fraction of GPU memory to use (default: 0.7)
        """
        self.input_path = input_path
        self.output_path = output_path
        self.num_tiles = num_tiles
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.batch_size = batch_size
        self.gpu_memory_fraction = gpu_memory_fraction
        
        # Check if input video exists
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input video file not found: {input_path}")
        
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Initialize GPU device
        if self.use_gpu:
            self.device = torch.device("cuda")
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
            
            # Set optimal tensor precision for the GPU
            torch.set_float32_matmul_precision('high')
            
            # Determine optimal batch size based on available GPU memory with less conservative estimate
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # in GB
            
            # Use user-specified GPU memory fraction (default 70%)
            usable_mem = gpu_mem * self.gpu_memory_fraction
            
            # More aggressive memory estimation - assume 0.5GB per frame instead of 2GB
            memory_per_frame = 0.5  # Conservative estimate was 2GB
            dynamic_batch_size = max(1, int(usable_mem / memory_per_frame))
            
            # Use the minimum of user-specified batch size and our calculated one
            self.batch_size = min(self.batch_size, dynamic_batch_size) if self.batch_size > 0 else dynamic_batch_size
            
            print(f"Using batch size of {self.batch_size} based on {self.gpu_memory_fraction*100:.0f}% of available GPU memory ({usable_mem:.1f}GB)")
        else:
            self.device = torch.device("cpu")
            print("Using CPU for computations")
        
        # Initialize MRF model and entropy calculator with GPU acceleration
        self.mrf_model = QuadwiseMRF()
        self.entropy_calculator = EntropyCalculator(self.mrf_model)
        
        # Pre-generate tile mapping for efficiency
        self.tile_regions = None
    
    def extract_frames(self, sample_rate=1.0, max_frames=None):
        """
        Extract frames from input video with optimized memory usage.
        
        Parameters:
        -----------
        sample_rate : float
            Fraction of frames to extract (default: 1.0)
        max_frames : int, optional
            Maximum number of frames to extract (default: None, extract all)
            
        Returns:
        --------
        frames : list of ndarrays
            List of extracted video frames
        fps : float
            Frames per second of the input video
        """
        start_time = time.time()
        
        # Open video file
        cap = cv2.VideoCapture(self.input_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video file: {self.input_path}")
        
        # Get video properties
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Apply max_frames limit if specified
        if max_frames is not None and max_frames < frame_count:
            frame_count = max_frames
            print(f"Limited to {max_frames} frames")
        
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
            
            if max_frames is not None and i >= max_frames - 1:
                break
        
        cap.release()
        
        elapsed = time.time() - start_time
        print(f"Extracted {len(frames)} frames in {elapsed:.2f} seconds")
        return frames, fps
    
    def generate_tiles(self, frame):
        """
        Generate tiles for a frame using Fibonacci lattice.
        Uses cached tile regions for better performance.
        
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
        
        # Generate tile regions if not already cached
        if self.tile_regions is None:
            start_time = time.time()
            print("Generating Fibonacci lattice tiling...")
            
            # Generate Fibonacci lattice points
            points_3d, points_2d = generate_fibonacci_points(self.num_tiles)
            
            # Compute Voronoi regions
            sv = compute_voronoi_regions(points_3d)
            
            # Map to ERP rectangles
            self.tile_regions = map_to_erp_rectangles(sv, points_2d, width, height)
            
            elapsed = time.time() - start_time
            print(f"Tile regions generated in {elapsed:.2f} seconds")
        
        # Extract tiles in parallel with threading
        tiles = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Define extraction function
            extract_func = partial(extract_tile_from_frame, frame)
            
            # Submit all extraction tasks
            future_to_region = {executor.submit(extract_func, region): i 
                              for i, region in enumerate(self.tile_regions)}
            
            # Process results as they complete
            for future in concurrent.futures.as_completed(future_to_region):
                tile, mask = future.result()
                tiles.append(tile)
                
        return tiles, self.tile_regions
    
    def estimate_empirical_distribution(self, frames_sample):
        """
        Estimate empirical distribution from frame samples.
        Uses GPU-accelerated implementation.
        
        Parameters:
        -----------
        frames_sample : list of ndarrays
            Sample of video frames
            
        Returns:
        --------
        None (updates self.mrf_model.empirical_dist)
        """
        start_time = time.time()
        print("Estimating empirical distribution with GPU acceleration...")
        
        # Use the GPU-accelerated implementation in the MRF model
        result = self.mrf_model.estimate_empirical_distribution(frames_sample)
        
        elapsed = time.time() - start_time
        print(f"Empirical distribution estimation completed in {elapsed:.2f} seconds")
        
        return result
    
    def compute_frame_entropy(self, frame, tile_regions, custom_potential=None, use_simplified=True):
        """
        Compute entropy for each tile in a frame.
        Optimized with GPU acceleration.
        
        Parameters:
        -----------
        frame : ndarray
            Video frame
        tile_regions : list of dicts
            List of tile region definitions
        custom_potential : function, optional
            Custom smoothness function to replace the default
        use_simplified : bool
            Whether to use simplified entropy calculation (much faster but less accurate)
            
        Returns:
        --------
        entropies : list of float
            Entropy value for each tile
        """
        # Extract all tiles first
        tiles = []
        for region in tile_regions:
            tile, _ = extract_tile_from_frame(frame, region)
            tiles.append(tile)
        
        # Batch compute entropy for all tiles
        if use_simplified:
            entropies = self.entropy_calculator.simplified_batch_compute_entropy(tiles, custom_potential)
        else:
            entropies = self.entropy_calculator.batch_compute_entropy(tiles, custom_potential)
        
        # Round to 4 decimal places
        entropies = [round(entropy, 4) for entropy in entropies]
        
        return entropies
    
    def process_frame_batch(self, frames, tile_regions, custom_potential=None, use_simplified=True):
        """
        Process a batch of frames in parallel.
        
        Parameters:
        -----------
        frames : list of ndarrays
            List of video frames to process
        tile_regions : list of dicts
            List of tile region definitions
        custom_potential : function, optional
            Custom smoothness function to replace the default
            
        Returns:
        --------
        batch_entropies : list of lists
            List of entropy values for each frame in the batch
        """
        batch_entropies = []
        
        # Process each frame in the batch
        for i, frame in enumerate(frames):
            # Compute entropy for each tile
            entropies = self.compute_frame_entropy(frame, tile_regions, custom_potential, use_simplified)
            
            # Store entropy values
            batch_entropies.append(entropies)
            
        return batch_entropies
    
    def process_video(self, custom_potential=None, sample_rate=0.1, use_simplified=True, max_frames=None):
        """
        Process the entire video with batch processing.
        
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
        start_time = time.time()
        
        # Extract frames with max_frames limit if specified
        frames, fps = self.extract_frames(max_frames=max_frames)
        
        # Sample frames for empirical distribution
        sample_indices = np.random.choice(len(frames), int(len(frames) * sample_rate), replace=False)
        frames_sample = [frames[i] for i in sample_indices]
        
        # Estimate empirical distribution
        self.estimate_empirical_distribution(frames_sample)
        
        # Clear the sample frames to free memory
        frames_sample = None
        gc.collect()
        if self.use_gpu:
            torch.cuda.empty_cache()
        
        # Generate tile regions (only need to do this once)
        _, tile_regions = self.generate_tiles(frames[0])
        
        # Process frames in batches
        frame_entropies = []
        total_batches = int(np.ceil(len(frames) / self.batch_size))
        
        for batch_idx in tqdm(range(total_batches), desc="Processing frame batches"):
            # Get batch of frames
            start_idx = batch_idx * self.batch_size
            end_idx = min((batch_idx + 1) * self.batch_size, len(frames))
            batch_frames = frames[start_idx:end_idx]
            
            # Process the batch with the simplified method if requested
            batch_entropies = self.process_frame_batch(batch_frames, tile_regions, 
                                                      custom_potential, use_simplified)
            
            # Add frame numbers to entropy values
            for i, entropies in enumerate(batch_entropies):
                frame_num = start_idx + i
                frame_entropies.append([frame_num] + entropies)
                
            # Free memory after each batch
            gc.collect()
            if self.use_gpu:
                torch.cuda.empty_cache()
        
        total_time = time.time() - start_time
        print(f"Video processing completed in {total_time:.2f} seconds")
        print(f"Average time per frame: {total_time / len(frames):.2f} seconds")
        
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
        start_time = time.time()
        
        if csv_path is None:
            csv_path = os.path.splitext(self.output_path)[0] + ".csv"
        
        # Write header
        header = ["Frame"] + [f"Tile_{i+1}_Entropy" for i in range(self.num_tiles)]
        
        # Write data
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(frame_entropies)
        
        elapsed = time.time() - start_time
        print(f"Entropy values exported to {csv_path} in {elapsed:.2f} seconds")
        return csv_path
    
    def generate_output_video(self, frame_entropies, fps=30):
        """
        Generate output video with overlaid entropy values.
        Uses tile region caching for efficiency.
        
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
        start_time = time.time()
        
        # Extract frames
        frames, _ = self.extract_frames()
        
        # Use cached tile regions or generate if needed
        if self.tile_regions is None:
            _, self.tile_regions = self.generate_tiles(frames[0])
        
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
            vis_frame = visualize_tiling(vis_frame, self.tile_regions)
            
            # Overlay average entropy
            text = f"Avg Entropy: {avg_entropy:.4f}"
            cv2.putText(vis_frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Write frame to output video
            out.write(vis_frame)
        
        # Release video writer
        out.release()
        
        elapsed = time.time() - start_time
        print(f"Output video generated at {self.output_path} in {elapsed:.2f} seconds")
        return self.output_path
    
    def run(self, custom_potential=None, sample_rate=0.1, use_simplified=True, max_frames=None):
        """
        Run the entire pipeline with optimized processing.
        
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
        # Set CUDA benchmark mode for optimal performance
        if self.use_gpu:
            torch.backends.cudnn.benchmark = True
            
        total_start_time = time.time()
        
        # Process video and compute entropy with simplified method if requested
        frame_entropies, fps = self.process_video(
            custom_potential=custom_potential, 
            sample_rate=sample_rate,
            use_simplified=use_simplified,
            max_frames=max_frames
        )
        
        # Export entropy values to CSV
        csv_path = self.export_csv(frame_entropies)
        
        # Generate output video
        output_path = self.generate_output_video(frame_entropies, fps)
        
        total_elapsed = time.time() - total_start_time
        print(f"Total processing completed in {total_elapsed:.2f} seconds")
        
        # Clean up GPU memory
        if self.use_gpu:
            torch.cuda.empty_cache()
            
        return {
            "video_path": output_path,
            "csv_path": csv_path
        }
