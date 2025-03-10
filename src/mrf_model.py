import numpy as np
import cv2
from collections import defaultdict

class QuadwiseMRF:
    def __init__(self, beta=0.1, epsilon=1e-8):
        self.beta = beta
        self.epsilon = epsilon
        self.empirical_dist = None
        
    def define_cliques(self, tile, enforce_periodicity=True):
        height, width = tile.shape[:2]
        cliques = []
        
        for i in range(0, height-1, 2):
            for j in range(0, width-1, 2):
                clique = [(i, j), (i, j+1), (i+1, j), (i+1, j+1)]
                cliques.append(clique)
        
        return cliques
    
    def quantize_intensities(self, frame):
        if len(frame.shape) > 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        quantized = np.floor(frame / 51).astype(np.uint8)
        return quantized
    
    def estimate_empirical_distribution(self, frames, sample_rate=0.1):
        clique_counts = defaultdict(int)
        total_cliques = 0
        
        num_frames = len(frames)
        window_size = max(1, num_frames // 10)
        windows = [frames[i:i+window_size] for i in range(0, num_frames, window_size)]
        
        for window in windows:
            for frame in window:
                quantized_frame = self.quantize_intensities(frame)
                cliques = self.define_cliques(quantized_frame)
                
                for clique in cliques:
                    values = tuple(quantized_frame[i, j] for i, j in clique)
                    clique_counts[values] += 1
                    total_cliques += 1
        
        empirical_dist = {}
        for values, count in clique_counts.items():
            empirical_dist[values] = count / total_cliques
            
        self.empirical_dist = empirical_dist
        return empirical_dist
    
    def smoothness_function_default(self, clique_values):
        x_i, x_j, x_k, x_l = clique_values
        pairs = [(x_i, x_j), (x_i, x_k), (x_i, x_l), (x_j, x_k), (x_j, x_l), (x_k, x_l)]
        smoothness = sum((a - b) ** 2 for a, b in pairs)
        return smoothness
    
    def compute_energy(self, clique_values, custom_potential=None):
        if self.empirical_dist is None:
            raise ValueError("Empirical distribution not estimated. Call estimate_empirical_distribution first.")
        
        p_data = self.empirical_dist.get(clique_values, 0) + self.epsilon
        
        if custom_potential:
            smoothness = custom_potential(clique_values)
        else:
            smoothness = self.smoothness_function_default(clique_values)
        
        energy = -np.log(p_data) + self.beta * smoothness
        return energy
    
    def compute_potential(self, clique_values, custom_potential=None):
        energy = self.compute_energy(clique_values, custom_potential)
        potential = np.exp(-energy)
        return potential
    
    def extract_clique_values(self, quantized_frame, clique):
        return tuple(quantized_frame[i, j] for i, j in clique)