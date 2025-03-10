"""
Entropy calculation implementation for Markov Random Fields.

This module provides functions to:
1. Compute the partition function using Bethe approximation
2. Calculate expected potentials
3. Compute entropy for a tile

Optimized for GPU acceleration using PyTorch.
"""

import numpy as np
import torch
import itertools
from collections import defaultdict
import time
import torch.cuda
import sys
import cv2

# Global debug flag for detailed timing (disabled by default for better performance)
DEBUG_TIMING = False

def profile_gpu_memory():
    """Print GPU memory usage statistics."""
    if not torch.cuda.is_available():
        return "No CUDA available"
    
    # Get memory statistics
    mem_allocated = torch.cuda.memory_allocated() / (1024**2)  # MB
    mem_reserved = torch.cuda.memory_reserved() / (1024**2)    # MB
    max_mem = torch.cuda.max_memory_allocated() / (1024**2)    # MB
    
    return f"GPU Memory: {mem_allocated:.1f}MB allocated, {mem_reserved:.1f}MB reserved, {max_mem:.1f}MB peak"

def time_function(func):
    """Decorator to time functions when DEBUG_TIMING is enabled."""
    def wrapper(*args, **kwargs):
        if not DEBUG_TIMING:
            return func(*args, **kwargs)
        
        # Only record start time, without synchronization
        start_time = time.time()
        
        # Run the function
        result = func(*args, **kwargs)
        
        # Only synchronize once at the end for accurate timing
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end_time = time.time()
        
        # Print timing information
        func_name = func.__name__
        elapsed = end_time - start_time
        memory_info = profile_gpu_memory() if torch.cuda.is_available() else ""
        print(f"{func_name} took {elapsed:.4f}s | {memory_info}", file=sys.stderr)
        
        return result
    return wrapper

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
        
        # Use the same device as the MRF model
        self.device = self.mrf.device
        
        # Cache for entropy calculations to avoid redundant computations
        self.entropy_cache = {}
        
        # For performance tracking
        self.profile_times = defaultdict(list)
        
    @time_function
    def compute_expected_potential(self, quantized_tile, cliques, custom_potential=None):
        """
        Compute E[log φ_c(X_c)] using empirical distribution.
        GPU-accelerated with PyTorch.
        
        Parameters:
        -----------
        quantized_tile : ndarray or tensor
            Tile with quantized pixel values
        cliques : list of list of tuples
            List of cliques, where each clique is a list of (i,j) coordinates
        custom_potential : function, optional
            Custom smoothness function to replace the default
            
        Returns:
        --------
        expected_potential : float or tensor
            Expected value of log potential function
        """
        # Convert to tensor if it's not already
        if not isinstance(quantized_tile, torch.Tensor):
            quantized_tensor = torch.from_numpy(quantized_tile).to(self.device)
        else:
            quantized_tensor = quantized_tile
            
        # Initialize expected potential on GPU
        expected_potential = torch.tensor(0.0, device=self.device)
        
        # Process cliques in batches for better GPU utilization
        batch_size = 64  # Adjust based on GPU memory
        for i in range(0, len(cliques), batch_size):
            batch_cliques = cliques[i:i+batch_size]
            batch_expected = torch.tensor(0.0, device=self.device)
            
            # Use the MRF's batch compute potentials function
            batch_potentials = self.mrf.batch_compute_potentials(quantized_tensor, batch_cliques, custom_potential)
            
            # Process each potential from the batch
            for clique_idx, clique in enumerate(batch_cliques):
                # Extract clique values
                values = self.mrf.extract_clique_values(quantized_tile, clique)
                
                # Get probability from empirical distribution
                prob = self.mrf.empirical_dist.get(values, 0) + self.mrf.epsilon
                prob_tensor = torch.tensor(prob, device=self.device)
                
                # Get potential from batch result
                potential = batch_potentials[clique_idx]
                
                # Compute log potential
                if isinstance(potential, torch.Tensor):
                    log_potential = torch.log(potential + self.mrf.epsilon)
                else:
                    log_potential = torch.tensor(np.log(potential + self.mrf.epsilon), device=self.device)
                
                # Add weighted log potential to batch expected potential
                batch_expected += prob_tensor * log_potential
                
            # Add batch result to total
            expected_potential += batch_expected
            
        return expected_potential
    
    @time_function
    def compute_pixel_marginals(self, quantized_tile, cliques):
        """
        Compute marginal distributions for individual pixels.
        GPU-accelerated with PyTorch.
        
        Parameters:
        -----------
        quantized_tile : ndarray or tensor
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
        # Convert to tensor if it's not already
        if not isinstance(quantized_tile, torch.Tensor):
            quantized_tensor = torch.from_numpy(quantized_tile).to(self.device)
        else:
            quantized_tensor = quantized_tile
            
        # Create a unique set of pixel coordinates from all cliques
        pixel_coords = set()
        for clique in cliques:
            for i, j in clique:
                pixel_coords.add((i, j))
        
        # Initialize tensors for marginals and clique degrees
        height, width = quantized_tile.shape[:2]
        
        # Use dictionaries for sparse representation
        marginals = defaultdict(lambda: torch.zeros(5, device=self.device))  # 5 discrete levels (0-4)
        clique_degrees = defaultdict(int)
        
        # Count occurrences of each pixel value
        for clique in cliques:
            for i, j in clique:
                clique_degrees[(i, j)] += 1
                
                # Get pixel value
                if isinstance(quantized_tensor, torch.Tensor):
                    value = int(quantized_tensor[i, j].item())
                else:
                    value = quantized_tile[i, j]
                
                # Update marginal distribution
                marginals[(i, j)][value] += 1
        
        # Normalize marginals to get probabilities
        for pixel, counts in marginals.items():
            total = torch.sum(counts)
            if total > 0:
                marginals[pixel] = counts / total
                
        return marginals, clique_degrees
    
    @time_function
    def compute_partition_function_bethe(self, quantized_tile, cliques, custom_potential=None):
        """
        Approximate log partition function using Bethe approximation.
        GPU-accelerated with PyTorch.
        
        log Z ≈ ∑_c E[log φ_c(X_c)] - ∑_p (d_p - 1)E[log q_p(x_p)]
        
        Parameters:
        -----------
        quantized_tile : ndarray or tensor
            Tile with quantized pixel values
        cliques : list of list of tuples
            List of cliques, where each clique is a list of (i,j) coordinates
        custom_potential : function, optional
            Custom smoothness function to replace the default
            
        Returns:
        --------
        log_z : float or tensor
            Approximated log partition function
        """
        # Compute expected potential
        expected_potential_sum = self.compute_expected_potential(quantized_tile, cliques, custom_potential)
        
        # Compute pixel marginals and clique degrees
        marginals, clique_degrees = self.compute_pixel_marginals(quantized_tile, cliques)
        
        # Compute the second term in Bethe approximation
        marginal_entropy_term = torch.tensor(0.0, device=self.device)
        for pixel, marginal in marginals.items():
            # Get clique degree for this pixel
            d_p = clique_degrees[pixel]
            
            # Skip if degree is 1 (no correction needed)
            if d_p <= 1:
                continue
            
            # Compute entropy of marginal distribution
            pixel_entropy = torch.tensor(0.0, device=self.device)
            for p in marginal:
                if p > 0:
                    pixel_entropy -= p * torch.log(p + self.mrf.epsilon)
            
            # Add weighted entropy to marginal entropy term
            marginal_entropy_term += (d_p - 1) * pixel_entropy
        
        # Compute log partition function
        if isinstance(expected_potential_sum, torch.Tensor):
            log_z = expected_potential_sum + marginal_entropy_term
        else:
            # Convert to tensor if needed
            expected_potential_tensor = torch.tensor(expected_potential_sum, device=self.device)
            log_z = expected_potential_tensor + marginal_entropy_term
        
        return log_z
    
    @time_function
    def compute_entropy_bethe(self, quantized_tile, cliques, custom_potential=None):
        """
        Calculate entropy H(X) for a tile using Bethe approximation.
        GPU-accelerated with PyTorch.
        
        H(X) = log Z - ∑_c E[log φ_c(X_c)]
        
        Parameters:
        -----------
        quantized_tile : ndarray or tensor
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
        if isinstance(log_z, torch.Tensor) and isinstance(expected_potential_sum, torch.Tensor):
            entropy = log_z - expected_potential_sum
        elif isinstance(log_z, torch.Tensor):
            expected_potential_tensor = torch.tensor(expected_potential_sum, device=self.device)
            entropy = log_z - expected_potential_tensor
        else:
            log_z_tensor = torch.tensor(log_z, device=self.device)
            expected_potential_tensor = torch.tensor(expected_potential_sum, device=self.device)
            entropy = log_z_tensor - expected_potential_tensor
        
        # Convert to float if it's a tensor
        if isinstance(entropy, torch.Tensor):
            entropy = entropy.item()
        
        return entropy
    
    @time_function
    def compute_entropy(self, tile, custom_potential=None, method='bethe'):
        """
        Compute entropy for a tile.
        GPU-accelerated with PyTorch and cached for performance.
        
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
        # Create a hash for the tile and custom_potential to use as cache key
        # Simple checksum for cache key - not perfect but fast
        if isinstance(tile, np.ndarray):
            cache_key = hash(tile.tobytes()) + hash(str(custom_potential)) + hash(method)
        else:
            # For tensor input, convert to numpy first
            cache_key = hash(tile.cpu().numpy().tobytes()) + hash(str(custom_potential)) + hash(method)
        
        # Check if result is in cache
        if cache_key in self.entropy_cache:
            return self.entropy_cache[cache_key]
        
        # Start timing
        start_time = time.time()
        
        # Quantize tile with batch GPU processing
        quantized_tile = self.mrf.quantize_intensities(tile)
        
        # Define cliques (now uses caching internally)
        cliques = self.mrf.define_cliques(quantized_tile)
        
        # Compute entropy using specified method
        if method == 'bethe':
            entropy = self.compute_entropy_bethe(quantized_tile, cliques, custom_potential)
        else:
            raise ValueError(f"Unknown entropy calculation method: {method}")
        
        # Cache the result
        self.entropy_cache[cache_key] = entropy
        
        # Performance logging (uncomment for debugging)
        # elapsed = time.time() - start_time
        # print(f"Entropy calculated in {elapsed:.4f} seconds")
        
        return entropy
    
    @time_function
    def batch_compute_entropy(self, tiles, custom_potential=None, method='bethe'):
        """
        Compute entropy for multiple tiles in batch mode.
        Optimized for better GPU utilization with sub-batching.
        
        Parameters:
        -----------
        tiles : list of ndarrays
            List of tiles extracted from video frames
        custom_potential : function, optional
            Custom smoothness function to replace the default
        method : str
            Method to use for entropy calculation ('bethe' for Bethe approximation)
            
        Returns:
        --------
        entropies : list of float
            List of entropy values for each tile
        """
        print(f"Batch computing entropy for {len(tiles)} tiles...")
        
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            print(f"Initial GPU memory: {profile_gpu_memory()}")
            
        start_time = time.time()
        
        # Process tiles in smaller sub-batches to maximize GPU utilization
        sub_batch_size = 5  # Process 5 tiles at once
        entropies = []
        
        # Quantize all tiles first to improve GPU utilization
        quantize_start = time.time()
        quantized_tiles = []
        for tile in tiles:
            quantized_tiles.append(self.mrf.quantize_intensities(tile))
        
        quantize_time = time.time() - quantize_start
        print(f"Tile quantization took {quantize_time:.2f}s | {profile_gpu_memory()}")
        
        # Process tiles in sub-batches
        for batch_idx in range(0, len(tiles), sub_batch_size):
            batch_start = time.time()
            
            # Get current sub-batch
            batch_end = min(batch_idx + sub_batch_size, len(tiles))
            cur_batch = quantized_tiles[batch_idx:batch_end]
            
            # Process all tiles in the sub-batch
            batch_entropies = []
            for idx, quantized_tile in enumerate(cur_batch):
                # Define cliques (uses caching internally)
                cliques = self.mrf.define_cliques(quantized_tile)
                
                # Compute entropy
                if method == 'bethe':
                    entropy = self.compute_entropy_bethe(quantized_tile, cliques, custom_potential)
                else:
                    raise ValueError(f"Unknown entropy calculation method: {method}")
                
                # Convert to scalar if needed
                if isinstance(entropy, torch.Tensor):
                    batch_entropies.append(entropy.item())
                else:
                    batch_entropies.append(entropy)
            
            # Add batch results to total
            entropies.extend(batch_entropies)
            
            batch_time = time.time() - batch_start
            print(f"  Sub-batch {batch_idx//sub_batch_size + 1}/{(len(tiles)+sub_batch_size-1)//sub_batch_size} "
                  f"processed in {batch_time:.2f}s | {profile_gpu_memory()}")
            
            # Only free memory after each sub-batch instead of each tile
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        elapsed = time.time() - start_time
        print(f"Batch entropy calculation completed in {elapsed:.2f} seconds")
        print(f"Final GPU memory: {profile_gpu_memory()}")
        
        return entropies
    
    @time_function
    def simplified_batch_compute_entropy(self, tiles, custom_potential=None):
        """
        A simplified, faster entropy computation using a reduced model.
        This uses a simple approximation to get quick results.
        
        Parameters:
        -----------
        tiles : list of ndarrays
            List of tiles extracted from video frames
        custom_potential : function, optional
            Custom smoothness function to replace the default
            
        Returns:
        --------
        entropies : list of float
            List of entropy values for each tile
        """
        print(f"Computing simplified entropy for {len(tiles)} tiles...")
        
        # Convert all tiles to grayscale tensors and move to GPU
        tile_tensors = []
        for tile in tiles:
            if len(tile.shape) > 2:
                gray = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY)
            else:
                gray = tile
            tile_tensors.append(torch.from_numpy(gray).float().to(self.device))
        
        # Compute a simplified entropy metric (spatial gradient entropy)
        entropies = []
        for tile_tensor in tile_tensors:
            # Calculate gradients
            gx = tile_tensor[:, 1:] - tile_tensor[:, :-1]  # Shape: [height, width-1]
            gy = tile_tensor[1:, :] - tile_tensor[:-1, :]  # Shape: [height-1, width]
            
            # Debug shape information if debug flag is on
            if DEBUG_TIMING:
                print(f"Tile tensor shape: {tile_tensor.shape}")
                print(f"gx shape: {gx.shape}, gy shape: {gy.shape}")
            
            # Align dimensions - take common region for both gradients
            # This ensures we're using dimensions that are compatible
            # The result will be of shape [height-1, width-1]
            grad_mag = torch.sqrt(gx[:-1, :]**2 + gy[:, :-1]**2 + 1e-10)
            
            # Normalize
            grad_mag = grad_mag / grad_mag.max()
            
            # Simple entropy calculation based on gradient distribution
            bins = 10
            hist = torch.histc(grad_mag, bins=bins, min=0, max=1)
            hist = hist / hist.sum()
            
            # Shannon entropy
            entropy = -torch.sum(hist * torch.log2(hist + 1e-10))
            entropies.append(entropy.item())
        
        # Free GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        return entropies
