"""
Entropy calculation implementation for Markov Random Fields.

This module provides functions to:
1. Compute the partition function using Bethe approximation
2. Calculate expected potentials
3. Compute entropy for a tile
"""

import numpy as np
import itertools
from collections import defaultdict

class EntropyCalculator:
    def __init__(self, mrf_model):
        """
        Initialize with MRF model.
        
        Parameters:
        -----------
        mrf_model : QuadwiseMRF
            The MRF model to use for entropy calculation
        """
        self.mrf = mrf_model
        
    def compute_expected_potential(self, quantized_tile, cliques, custom_potential=None):
        """
        Compute E[log φ_c(X_c)] using empirical distribution.
        
        Parameters:
        -----------
        quantized_tile : ndarray
            Tile with quantized pixel values
        cliques : list of list of tuples
            List of cliques, where each clique is a list of (i,j) coordinates
        custom_potential : function, optional
            Custom smoothness function to replace the default
            
        Returns:
        --------
        expected_potential : float
            Expected value of log potential function
        """
        # Initialize expected potential
        expected_potential = 0.0
        
        # Iterate over all cliques
        for clique in cliques:
            # Extract clique values
            values = self.mrf.extract_clique_values(quantized_tile, clique)
            
            # Get probability from empirical distribution
            prob = self.mrf.empirical_dist.get(values, 0) + self.mrf.epsilon
            
            # Compute log potential
            potential = self.mrf.compute_potential(values, custom_potential)
            log_potential = np.log(potential + self.mrf.epsilon)
            
            # Add weighted log potential to expected potential
            expected_potential += prob * log_potential
            
        return expected_potential
    
    def compute_pixel_marginals(self, quantized_tile, cliques):
        """
        Compute marginal distributions for individual pixels.
        
        Parameters:
        -----------
        quantized_tile : ndarray
            Tile with quantized pixel values
        cliques : list of list of tuples
            List of cliques, where each clique is a list of (i,j) coordinates
            
        Returns:
        --------
        marginals : dict
            Dictionary mapping pixel coordinates to their marginal distributions
        clique_degrees : dict
            Dictionary mapping pixel coordinates to their clique degrees
        """
        # Initialize marginals and clique degrees
        marginals = defaultdict(lambda: np.zeros(5))  # 5 discrete levels (0-4)
        clique_degrees = defaultdict(int)
        
        # Count occurrences of each pixel value
        for clique in cliques:
            for i, j in clique:
                clique_degrees[(i, j)] += 1
                
                # Get pixel value
                value = quantized_tile[i, j]
                
                # Update marginal distribution
                marginals[(i, j)][value] += 1
        
        # Normalize marginals to get probabilities
        for pixel, counts in marginals.items():
            total = np.sum(counts)
            if total > 0:
                marginals[pixel] = counts / total
                
        return marginals, clique_degrees
    
    def compute_partition_function_bethe(self, quantized_tile, cliques, custom_potential=None):
        """
        Approximate log partition function using Bethe approximation.
        
        log Z ≈ ∑_c E[log φ_c(X_c)] - ∑_p (d_p - 1)E[log q_p(x_p)]
        
        Parameters:
        -----------
        quantized_tile : ndarray
            Tile with quantized pixel values
        cliques : list of list of tuples
            List of cliques, where each clique is a list of (i,j) coordinates
        custom_potential : function, optional
            Custom smoothness function to replace the default
            
        Returns:
        --------
        log_z : float
            Approximated log partition function
        """
        # Compute expected potential
        expected_potential_sum = self.compute_expected_potential(quantized_tile, cliques, custom_potential)
        
        # Compute pixel marginals and clique degrees
        marginals, clique_degrees = self.compute_pixel_marginals(quantized_tile, cliques)
        
        # Compute the second term in Bethe approximation
        marginal_entropy_term = 0.0
        for pixel, marginal in marginals.items():
            # Get clique degree for this pixel
            d_p = clique_degrees[pixel]
            
            # Skip if degree is 1 (no correction needed)
            if d_p <= 1:
                continue
            
            # Compute entropy of marginal distribution
            pixel_entropy = 0.0
            for p in marginal:
                if p > 0:
                    pixel_entropy -= p * np.log(p)
            
            # Add weighted entropy to marginal entropy term
            marginal_entropy_term += (d_p - 1) * pixel_entropy
        
        # Compute log partition function
        log_z = expected_potential_sum + marginal_entropy_term
        
        return log_z
    
    def compute_entropy_bethe(self, quantized_tile, cliques, custom_potential=None):
        """
        Calculate entropy H(X) for a tile using Bethe approximation.
        
        H(X) = log Z - ∑_c E[log φ_c(X_c)]
        
        Parameters:
        -----------
        quantized_tile : ndarray
            Tile with quantized pixel values
        cliques : list of list of tuples
            List of cliques, where each clique is a list of (i,j) coordinates
        custom_potential : function, optional
            Custom smoothness function to replace the default
            
        Returns:
        --------
        entropy : float
            Entropy value for the tile
        """
        # Compute log partition function
        log_z = self.compute_partition_function_bethe(quantized_tile, cliques, custom_potential)
        
        # Compute expected potential
        expected_potential_sum = self.compute_expected_potential(quantized_tile, cliques, custom_potential)
        
        # Compute entropy
        entropy = log_z - expected_potential_sum
        
        return entropy
    
    def compute_entropy(self, tile, custom_potential=None, method='bethe'):
        """
        Compute entropy for a tile.
        
        Parameters:
        -----------
        tile : ndarray
            Tile extracted from a video frame
        custom_potential : function, optional
            Custom smoothness function to replace the default
        method : str
            Method to use for entropy calculation ('bethe' for Bethe approximation)
            
        Returns:
        --------
        entropy : float
            Entropy value for the tile
        """
        # Quantize tile
        quantized_tile = self.mrf.quantize_intensities(tile)
        
        # Define cliques
        cliques = self.mrf.define_cliques(quantized_tile)
        
        # Compute entropy using specified method
        if method == 'bethe':
            entropy = self.compute_entropy_bethe(quantized_tile, cliques, custom_potential)
        else:
            raise ValueError(f"Unknown entropy calculation method: {method}")
        
        return entropy
