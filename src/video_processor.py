import os
import cv2
import numpy as np
import csv
from tqdm import tqdm
import torch

from src.fibonacci_lattice import generate_fibonacci_points, compute_voronoi_regions, map_to_erp_rectangles, extract_tile_from_frame, visualize_tiling
from src.mrf_model import QuadwiseMRF
from src.entropy_calculator import EntropyCalculator

class VideoProcessor:
    def __init__(self, input_path, output_path, num_tiles=55, use_gpu=True):
        self.input_path = input_path
        self.output_path = output_path
        self.num_tiles = num_tiles
        self.use_gpu = use_gpu and torch.cuda.is_available()
        
        # Initialize models
        self.mrf_model = QuadwiseMRF()
        self.entropy_calculator = EntropyCalculator(self.mrf_model)
        
        # Initialize GPU if available
        if self.use_gpu:
            self.device = torch.device("cuda")
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            self.device = torch.device("cpu")
            print("Using CPU for computations")
    
    def extract_frames(self, sample_rate=1.0):
        cap = cv2.VideoCapture(self.input_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video file: {self.input_path}")
        
        # Get video properties
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Extract frames
        frames = []
        for i in range(frame_count):
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        
        cap.release()
        return frames, fps
    
    def generate_tiles(self, frame):
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
            tile, _ = extract_tile_from_frame(frame, region)
            tiles.append(tile)
        
        return tiles, tile_regions
    
    def compute_frame_entropy(self, frame, tile_regions, custom_potential=None):
        entropies = []
        
        for region in tile_regions:
            tile, _ = extract_tile_from_frame(frame, region)
            entropy = self.entropy_calculator.compute_entropy(tile, custom_potential)
            entropy = round(entropy, 4)
            entropies.append(entropy)
        
        return entropies
    
    def export_csv(self, frame_entropies, csv_path=None):
        if csv_path is None:
            csv_path = os.path.splitext(self.output_path)[0] + ".csv"
        
        header = ["Frame"] + [f"Tile_{i+1}_Entropy" for i in range(self.num_tiles)]
        
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(frame_entropies)
        
        print(f"Entropy values exported to {csv_path}")
        return csv_path
    
    def run(self, custom_potential=None, sample_rate=0.1):
        # Extract frames
        frames, fps = self.extract_frames()
        
        # Sample frames for empirical distribution
        sample_indices = np.random.choice(len(frames), int(len(frames) * sample_rate), replace=False)
        frames_sample = [frames[i] for i in sample_indices]
        
        # Estimate empirical distribution
        self.mrf_model.estimate_empirical_distribution(frames_sample)
        
        # Generate tiles for first frame
        _, tile_regions = self.generate_tiles(frames[0])
        
        # Process each frame
        frame_entropies = []
        for i, frame in enumerate(frames):
            entropies = self.compute_frame_entropy(frame, tile_regions, custom_potential)
            frame_entropies.append([i] + entropies)
        
        # Export entropy values to CSV
        csv_path = self.export_csv(frame_entropies)
        
        # Generate output video with overlay
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(self.output_path, fourcc, fps, (frames[0].shape[1], frames[0].shape[0]))
        
        for i, (frame, entropies) in enumerate(zip(frames, frame_entropies)):
            avg_entropy = np.mean(entropies[1:])
            vis_frame = visualize_tiling(frame, tile_regions)
            cv2.putText(vis_frame, f"Avg Entropy: {avg_entropy:.4f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            out.write(vis_frame)
        
        out.release()
        
        return {
            "video_path": self.output_path,
            "csv_path": csv_path
        }