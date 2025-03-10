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
        self.probs_tensor = None    # GPU tensor version of empirical distribution
        
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
        Estimate empirical distribution from frame samples with improved GPU utilization.
        
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
        
        # Create tensor to hold clique coordinates
        # Each clique has 4 pixels, each pixel has 2 coordinates (i,j)
        # Shape: [num_cliques, 4, 2]
        clique_coords = torch.zeros(len(cliques), 4, 2, dtype=torch.long, device=self.device)
        
        # Populate clique coordinates tensor
        for cidx, clique in enumerate(cliques):
            for pidx, (i, j) in enumerate(clique):
                clique_coords[cidx, pidx, 0] = i  # row
                clique_coords[cidx, pidx, 1] = j  # column
        
        # Process each frame with tensor operations - shows a progress bar
        for frame_idx, quantized_frame in enumerate(tqdm(quantized_frames, desc="Processing empirical distribution")):
            # Convert to tensor for GPU processing
            if not isinstance(quantized_frame, torch.Tensor):
                frame_tensor = torch.from_numpy(quantized_frame).to(self.device)
            else:
                frame_tensor = quantized_frame
                
            # Process all cliques in batch where possible
            for cidx, clique in enumerate(cliques):
                # Extract the 4 pixel values for this clique
                i1, j1 = clique[0]
                i2, j2 = clique[1]
                i3, j3 = clique[2]
                i4, j4 = clique[3]
                
                # Get the values
                v1 = int(frame_tensor[i1, j1].item())
                v2 = int(frame_tensor[i2, j2].item())
                v3 = int(frame_tensor[i3, j3].item())
                v4 = int(frame_tensor[i4, j4].item())
                
                # Update counts tensor
                counts_tensor[v1, v2, v3, v4] += 1
                total_cliques += 1
        
        # Convert counts to probabilities
        probs_tensor = counts_tensor / total_cliques
        
        # Store GPU tensor version for faster lookups
        self.probs_tensor = probs_tensor
        
        # Also create dict version for compatibility
        probs_np = probs_tensor.cpu().numpy()
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
        Compute potentials for all cliques in a frame using true batch operations.
        GPU-optimized for significantly faster processing.
        
        Parameters:
        -----------
        quantized_frame : ndarray or tensor
            Frame with quantized pixel values
        cliques : list of list of tuples
            List of cliques, where each clique is a list of (i,j) coordinates
        custom_potential : function, optional
            Custom smoothness function to replace the default
            
        Returns:
        --------
        potentials : torch.Tensor or list
            Tensor or list of potential values for each clique
        """
        # Convert frame to tensor if it's not already
        if not isinstance(quantized_frame, torch.Tensor):
            frame_tensor = torch.from_numpy(quantized_frame).to(self.device)
        else:
            frame_tensor = quantized_frame
            
        # Use larger batch size for much better GPU utilization
        batch_size = 256  # Increased from 128
        
        # Process batches with tensor operations for GPU acceleration
        all_potentials = []
        
        for i in range(0, len(cliques), batch_size):
            batch_cliques = cliques[i:i+batch_size]
            batch_size_actual = len(batch_cliques)
            
            # Create tensor to hold all batch clique values [batch_size, 4]
            batch_values = torch.zeros(batch_size_actual, 4, dtype=torch.long, device=self.device)
            
            # Extract all clique values at once
            for j, clique in enumerate(batch_cliques):
                for k, (r, c) in enumerate(clique):
                    batch_values[j, k] = frame_tensor[r, c]
            
            # Use the GPU tensor version of empirical distribution if available and no custom potential
            if self.probs_tensor is not None and custom_potential is None:
                # Compute potentials for the entire batch at once
                batch_energies = torch.zeros(batch_size_actual, device=self.device)
                
                for j in range(batch_size_actual):
                    # Get probability from probs_tensor
                    v1, v2, v3, v4 = batch_values[j]
                    p_data = self.probs_tensor[v1, v2, v3, v4] + self.epsilon
                    
                    # Compute smoothness using vectorized operations
                    x = batch_values[j].float()
                    
                    # All possible pairs in a 2×2 clique (6 pairs)
                    pairs = torch.tensor([[0,1], [0,2], [0,3], [1,2], [1,3], [2,3]], device=self.device)
                    a = x[pairs[:, 0]]
                    b = x[pairs[:, 1]]
                    
                    # Sum of squared differences
                    smoothness = torch.sum((a - b) ** 2)
                    
                    # Calculate energy
                    batch_energies[j] = -torch.log(p_data) + self.beta * smoothness
                
                # Convert all energies to potentials at once
                batch_potentials = torch.exp(-batch_energies)
                all_potentials.append(batch_potentials)
            else:
                # Fallback to individual processing when using custom potential
                batch_potentials = []
                for values in batch_values:
                    potential = self.compute_potential(values, custom_potential)
                    if isinstance(potential, torch.Tensor):
                        batch_potentials.append(potential)
                    else:
                        batch_potentials.append(torch.tensor(potential, device=self.device))
                
                all_potentials.append(torch.stack(batch_potentials))
        
        # Combine all batches
        if len(all_potentials) == 1:
            return all_potentials[0]
        else:
            return torch.cat(all_potentials)
