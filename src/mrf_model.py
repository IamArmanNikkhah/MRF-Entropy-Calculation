"""
Quadwise Markov Random Field (MRF) implementation for modeling pixel dependencies.

This module provides functions to:
1. Define cliques (2×2 non-overlapping pixel blocks)
2. Quantize pixel intensities to discrete levels
3. Estimate empirical distributions from frame samples
4. Compute energy functions for cliques

Optimized for GPU acceleration using PyTorch.
"""

import numpy as np
import torch
import cv2
import itertools
from collections import defaultdict
import time
from tqdm import tqdm

class QuadwiseMRF:
    def __init__(self, beta=0.1, epsilon=1e-8):
        """
        Initialize MRF with parameters.
        
        Parameters:
        -----------
        beta : float
            Smoothness strength parameter (default: 0.1)
        epsilon : float
            Small value to prevent log(0) (default: 1e-8)
        """
        self.beta = beta
        self.epsilon = epsilon
        self.empirical_dist = None  # Will be populated by estimate_empirical_distribution
        
        # Initialize device for PyTorch operations
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Precompute all possible clique configurations (5^4 = 625 possible configurations for 5 intensity levels)
        self.all_configurations = list(itertools.product(range(5), repeat=4))
        
        # Cache for defined cliques to avoid recomputation
        self.cliques_cache = {}
        
    def define_cliques(self, tile, enforce_periodicity=True):
        """
        Define 2×2 non-overlapping pixel blocks with adjacency constraints.
        Uses cached cliques for the same tile dimensions to avoid recomputation.
        
        Parameters:
        -----------
        tile : ndarray
            Tile extracted from a video frame
        enforce_periodicity : bool
            Whether to enforce spherical periodicity at boundaries
            
        Returns:
        --------
        cliques : list of tuples
            Each tuple contains the (i,j) coordinates of the four pixels in a clique
        """
        height, width = tile.shape[:2]
        
        # Check if cliques for this tile dimension are already cached
        cache_key = (height, width, enforce_periodicity)
        if cache_key in self.cliques_cache:
            return self.cliques_cache[cache_key]
        
        cliques = []
        
        # Create 2×2 non-overlapping blocks
        for i in range(0, height-1, 2):
            for j in range(0, width-1, 2):
                clique = [(i, j), (i, j+1), (i+1, j), (i+1, j+1)]
                cliques.append(clique)
        
        # Handle boundary conditions with spherical periodicity if required
        if enforce_periodicity and height % 2 == 1:
            # Handle bottom row
            for j in range(0, width-1, 2):
                # Bottom row connects to top row
                clique = [(height-1, j), (height-1, j+1), (0, j), (0, j+1)]
                cliques.append(clique)
        
        if enforce_periodicity and width % 2 == 1:
            # Handle rightmost column
            for i in range(0, height-1, 2):
                # Rightmost column connects to leftmost column
                clique = [(i, width-1), (i, 0), (i+1, width-1), (i+1, 0)]
                cliques.append(clique)
        
        # Special case for bottom-right corner if both dimensions are odd
        if enforce_periodicity and height % 2 == 1 and width % 2 == 1:
            clique = [(height-1, width-1), (height-1, 0), (0, width-1), (0, 0)]
            cliques.append(clique)
        
        # Cache the cliques for future use
        self.cliques_cache[cache_key] = cliques
        
        return cliques
    
    def quantize_intensities(self, frame):
        """
        Quantize grayscale values to 5 discrete levels {0,1,2,3,4}.
        Optimized with PyTorch for GPU acceleration.
        
        Parameters:
        -----------
        frame : ndarray
            Grayscale frame with pixel values in range [0, 255]
            
        Returns:
        --------
        quantized : ndarray
            Frame with pixel values quantized to [0, 1, 2, 3, 4]
        """
        # Ensure frame is grayscale
        if len(frame.shape) > 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Convert to PyTorch tensor and move to GPU
        frame_tensor = torch.from_numpy(frame).float().to(self.device)
        
        # Quantize using floor division by 51 (255/5)
        # This maps [0-50] to 0, [51-101] to 1, etc.
        quantized_tensor = torch.floor(frame_tensor / 51).clamp(0, 4).byte()
        
        # Move back to CPU and convert to numpy for compatibility with other functions
        quantized = quantized_tensor.cpu().numpy()
        
        return quantized
    
    def batch_quantize_frames(self, frames, batch_size=16):
        """
        Quantize multiple frames in batches for faster processing.
        
        Parameters:
        -----------
        frames : list of ndarrays
            List of video frames
        batch_size : int
            Number of frames to process at once
            
        Returns:
        --------
        quantized_frames : list of ndarrays
            List of quantized frames
        """
        quantized_frames = []
        
        # Process frames in batches
        for i in range(0, len(frames), batch_size):
            batch = frames[i:i+batch_size]
            
            # Convert each frame to grayscale if needed
            gray_batch = []
            for frame in batch:
                if len(frame.shape) > 2:
                    gray_batch.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
                else:
                    gray_batch.append(frame)
            
            # Stack frames into a single tensor
            batch_tensor = torch.from_numpy(np.stack(gray_batch)).float().to(self.device)
            
            # Quantize all frames at once
            quantized_batch = torch.floor(batch_tensor / 51).clamp(0, 4).byte()
            
            # Move back to CPU and convert to numpy
            quantized_batch_np = quantized_batch.cpu().numpy()
            
            # Add each quantized frame to the result list
            for quant_frame in quantized_batch_np:
                quantized_frames.append(quant_frame)
        
        return quantized_frames
        
    def estimate_empirical_distribution(self, frames, sample_rate=0.1):
        """
        Estimate empirical distribution from frame samples.
        GPU-accelerated with PyTorch.
        
        Parameters:
        -----------
        frames : list of ndarrays
            List of grayscale video frames
        sample_rate : float
            Fraction of frames to sample (default: 0.1)
            
        Returns:
        --------
        empirical_dist : dict
            Dictionary mapping clique configurations to probabilities
        """
        start_time = time.time()
        print("Estimating empirical distribution with GPU acceleration...")
        
        # Initialize counts for all possible configurations (5^4 = 625)
        counts_tensor = torch.zeros(5, 5, 5, 5, dtype=torch.float32, device=self.device)
        total_cliques = 0
        
        # Get a representative frame to define cliques once
        if not frames:
            raise ValueError("No frames provided for empirical distribution estimation")
        
        # Batch quantize all frames for better performance
        quantized_frames = self.batch_quantize_frames(frames)
        
        # Get cliques from the first frame (same for all frames)
        cliques = self.define_cliques(quantized_frames[0])
        
        # Process each frame - shows a progress bar
        for frame_idx, quantized_frame in enumerate(tqdm(quantized_frames, desc="Processing empirical distribution")):
            # Extract clique values and update counts
            for clique in cliques:
                values = tuple(quantized_frame[i, j] for i, j in clique)
                # Update counts tensor (indices are the clique values)
                counts_tensor[values] += 1
                total_cliques += 1
        
        # Convert counts to probabilities
        probs_tensor = counts_tensor / total_cliques
        
        # Move tensor to CPU and convert to dictionary
        probs_np = probs_tensor.cpu().numpy()
        
        # Convert to dictionary for compatibility with existing code
        empirical_dist = {}
        for idx, prob in np.ndenumerate(probs_np):
            if prob > 0:  # Only store non-zero probabilities
                empirical_dist[idx] = float(prob)
        
        self.empirical_dist = empirical_dist
        
        elapsed = time.time() - start_time
        print(f"Empirical distribution estimated in {elapsed:.2f} seconds")
        
        return empirical_dist
    
    def smoothness_function_default(self, clique_values):
        """
        Default smoothness function: sum of squared differences between all 6 pairs.
        
        Parameters:
        -----------
        clique_values : tuple of int or tensor
            The four values in a clique (x_i, x_j, x_k, x_l)
            
        Returns:
        --------
        smoothness : float or tensor
            Smoothness value computed for the clique
        """
        if isinstance(clique_values, torch.Tensor):
            # If input is a tensor, use vectorized operations
            x = clique_values
            # Calculate all pairwise differences
            diffs = torch.combinations(x, 2)
            a = diffs[:, 0]
            b = diffs[:, 1]
            # Sum of squared differences
            return torch.sum((a.float() - b.float()) ** 2)
        else:
            # Legacy mode for tuple input
            x_i, x_j, x_k, x_l = clique_values
            
            # All possible pairs in a 2×2 clique
            pairs = [
                (x_i, x_j), (x_i, x_k), (x_i, x_l),
                (x_j, x_k), (x_j, x_l), (x_k, x_l)
            ]
            
            # Sum squared differences for all pairs using float to prevent overflow
            smoothness = sum((float(a) - float(b)) ** 2 for a, b in pairs)
            return smoothness
    
    def compute_energy(self, clique_values, custom_potential=None):
        """
        Compute energy function for a clique.
        
        Energy function: V_c(x_i, x_j, x_k, x_l) = -log(p_data(x_i, x_j, x_k, x_l) + ε) + β·f(x_i, x_j, x_k, x_l)
        
        Parameters:
        -----------
        clique_values : tuple of int or tensor
            The four values in a clique (x_i, x_j, x_k, x_l)
        custom_potential : function, optional
            Custom smoothness function to replace the default
            
        Returns:
        --------
        energy : float or tensor
            Energy value computed for the clique
        """
        if self.empirical_dist is None:
            raise ValueError("Empirical distribution not estimated. Call estimate_empirical_distribution first.")
        
        if isinstance(clique_values, torch.Tensor):
            # For tensor input, convert to tuple for dictionary lookup
            clique_tuple = tuple(clique_values.cpu().numpy().tolist())
            p_data = self.empirical_dist.get(clique_tuple, 0) + self.epsilon
            
            # Compute the smoothness term on GPU
            if custom_potential:
                smoothness = custom_potential(clique_values)
            else:
                smoothness = self.smoothness_function_default(clique_values)
            
            # Calculate energy on GPU
            energy = -torch.log(torch.tensor(p_data, device=self.device)) + self.beta * smoothness
            
        else:
            # Legacy mode for tuple input
            p_data = self.empirical_dist.get(clique_values, 0) + self.epsilon
            
            # Compute the smoothness term
            if custom_potential:
                smoothness = custom_potential(clique_values)
            else:
                smoothness = self.smoothness_function_default(clique_values)
            
            # Calculate energy
            energy = -np.log(p_data) + self.beta * smoothness
        
        return energy
    
    def compute_potential(self, clique_values, custom_potential=None):
        """
        Compute potential function for a clique: φ_c(X_c) = exp(-V_c(X_c)).
        
        Parameters:
        -----------
        clique_values : tuple of int or tensor
            The four values in a clique (x_i, x_j, x_k, x_l)
        custom_potential : function, optional
            Custom smoothness function to replace the default
            
        Returns:
        --------
        potential : float or tensor
            Potential value computed for the clique
        """
        energy = self.compute_energy(clique_values, custom_potential)
        
        if isinstance(energy, torch.Tensor):
            potential = torch.exp(-energy)
        else:
            potential = np.exp(-energy)
            
        return potential
    
    def extract_clique_values(self, quantized_frame, clique):
        """
        Extract values for a clique from a quantized frame.
        
        Parameters:
        -----------
        quantized_frame : ndarray or tensor
            Frame with quantized pixel values
        clique : list of tuples
            List of (i,j) coordinates for the four pixels in the clique
            
        Returns:
        --------
        values : tuple or tensor
            Tuple of values (x_i, x_j, x_k, x_l) for the clique
        """
        if isinstance(quantized_frame, torch.Tensor):
            # For tensor input, extract values directly
            values = torch.tensor([quantized_frame[i, j] for i, j in clique], device=self.device)
            return values
        else:
            # Legacy mode for numpy array input
            return tuple(quantized_frame[i, j] for i, j in clique)
    
    def batch_compute_potentials(self, quantized_frame, cliques, custom_potential=None):
        """
        Compute potentials for all cliques in a frame in batch mode.
        
        Parameters:
        -----------
        quantized_frame : ndarray
            Frame with quantized pixel values
        cliques : list of list of tuples
            List of cliques, where each clique is a list of (i,j) coordinates
        custom_potential : function, optional
            Custom smoothness function to replace the default
            
        Returns:
        --------
        potentials : list of float
            List of potential values for each clique
        """
        # Convert frame to tensor if it's not already
        if not isinstance(quantized_frame, torch.Tensor):
            frame_tensor = torch.from_numpy(quantized_frame).to(self.device)
        else:
            frame_tensor = quantized_frame
            
        # Initialize potentials list
        potentials = []
        
        # Process cliques in batches for better GPU utilization
        batch_size = 128  # Adjust based on GPU memory
        for i in range(0, len(cliques), batch_size):
            batch_cliques = cliques[i:i+batch_size]
            batch_values = []
            
            # Extract values for each clique in the batch
            for clique in batch_cliques:
                values = self.extract_clique_values(frame_tensor, clique)
                batch_values.append(values)
                
            # Stack tensors for batch processing if using tensors
            if isinstance(batch_values[0], torch.Tensor):
                batch_values_tensor = torch.stack(batch_values)
                
                # Compute potentials for the batch (this would require modifying compute_potential)
                # For now, process each clique individually
                batch_potentials = [self.compute_potential(values, custom_potential) 
                                   for values in batch_values]
                
            else:
                # Process each clique individually for tuple values
                batch_potentials = [self.compute_potential(values, custom_potential) 
                                   for values in batch_values]
                
            potentials.extend(batch_potentials)
            
        return potentials
