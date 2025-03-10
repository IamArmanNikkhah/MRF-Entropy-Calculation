"""
Quadwise Markov Random Field (MRF) implementation for modeling pixel dependencies.

This module provides functions to:
1. Define cliques (2×2 non-overlapping pixel blocks)
2. Quantize pixel intensities to discrete levels
3. Estimate empirical distributions from frame samples
4. Compute energy functions for cliques
"""

import numpy as np
import cv2
from collections import defaultdict

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
        
    def define_cliques(self, tile, enforce_periodicity=True):
        """
        Define 2×2 non-overlapping pixel blocks with adjacency constraints.
        
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
        
        return cliques
    
    def quantize_intensities(self, frame):
        """
        Quantize grayscale values to 5 discrete levels {0,1,2,3,4}.
        
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
        
        # Quantize using floor division by 51 (255/5)
        # This maps [0-50] to 0, [51-101] to 1, etc.
        quantized = np.floor(frame / 51).astype(np.uint8)
        
        # Ensure values are in range [0-4] by clipping any value of 5 to 4
        quantized = np.clip(quantized, 0, 4)
        
        return quantized
    
    def estimate_empirical_distribution(self, frames, sample_rate=0.1):
        """
        Estimate empirical distribution from frame samples.
        
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
        # Initialize dictionary to count occurrences of each clique configuration
        clique_counts = defaultdict(int)
        total_cliques = 0
        
        # Divide frames into 10 non-overlapping windows
        num_frames = len(frames)
        window_size = max(1, num_frames // 10)
        windows = [frames[i:i+window_size] for i in range(0, num_frames, window_size)]
        
        # Process each frame
        for window_idx, window in enumerate(windows):
            # Sample frames from other windows
            for frame_idx, frame in enumerate(window):
                # Quantize frame
                quantized_frame = self.quantize_intensities(frame)
                
                # Get all cliques from the frame
                cliques = self.define_cliques(quantized_frame)
                
                # For each clique, extract values and count occurrences
                for clique in cliques:
                    values = tuple(quantized_frame[i, j] for i, j in clique)
                    clique_counts[values] += 1
                    total_cliques += 1
        
        # Convert counts to probabilities
        empirical_dist = {}
        for values, count in clique_counts.items():
            empirical_dist[values] = count / total_cliques
            
        self.empirical_dist = empirical_dist
        return empirical_dist
    
    def smoothness_function_default(self, clique_values):
        """
        Default smoothness function: sum of squared differences between all 6 pairs.
        
        Parameters:
        -----------
        clique_values : tuple of int
            The four values in a clique (x_i, x_j, x_k, x_l)
            
        Returns:
        --------
        smoothness : float
            Smoothness value computed for the clique
        """
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
        clique_values : tuple of int
            The four values in a clique (x_i, x_j, x_k, x_l)
        custom_potential : function, optional
            Custom smoothness function to replace the default
            
        Returns:
        --------
        energy : float
            Energy value computed for the clique
        """
        if self.empirical_dist is None:
            raise ValueError("Empirical distribution not estimated. Call estimate_empirical_distribution first.")
        
        # Get probability from empirical distribution
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
        clique_values : tuple of int
            The four values in a clique (x_i, x_j, x_k, x_l)
        custom_potential : function, optional
            Custom smoothness function to replace the default
            
        Returns:
        --------
        potential : float
            Potential value computed for the clique
        """
        energy = self.compute_energy(clique_values, custom_potential)
        potential = np.exp(-energy)
        return potential
    
    def extract_clique_values(self, quantized_frame, clique):
        """
        Extract values for a clique from a quantized frame.
        
        Parameters:
        -----------
        quantized_frame : ndarray
            Frame with quantized pixel values
        clique : list of tuples
            List of (i,j) coordinates for the four pixels in the clique
            
        Returns:
        --------
        values : tuple
            Tuple of values (x_i, x_j, x_k, x_l) for the clique
        """
        return tuple(quantized_frame[i, j] for i, j in clique)
