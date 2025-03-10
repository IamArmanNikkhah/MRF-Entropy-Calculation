import numpy as np
from collections import defaultdict

class EntropyCalculator:
    def __init__(self, mrf_model):
        self.mrf = mrf_model
        
    def compute_expected_potential(self, quantized_tile, cliques, custom_potential=None):
        expected_potential = 0.0
        
        for clique in cliques:
            values = self.mrf.extract_clique_values(quantized_tile, clique)
            prob = self.mrf.empirical_dist.get(values, 0) + self.mrf.epsilon
            potential = self.mrf.compute_potential(values, custom_potential)
            log_potential = np.log(potential + self.mrf.epsilon)
            expected_potential += prob * log_potential
            
        return expected_potential
    
    def compute_pixel_marginals(self, quantized_tile, cliques):
        marginals = defaultdict(lambda: np.zeros(5))
        clique_degrees = defaultdict(int)
        
        for clique in cliques:
            for i, j in clique:
                clique_degrees[(i, j)] += 1
                value = quantized_tile[i, j]
                marginals[(i, j)][value] += 1
        
        for pixel, counts in marginals.items():
            total = np.sum(counts)
            if total > 0:
                marginals[pixel] = counts / total
                
        return marginals, clique_degrees
    
    def compute_partition_function_bethe(self, quantized_tile, cliques, custom_potential=None):
        expected_potential_sum = self.compute_expected_potential(quantized_tile, cliques, custom_potential)
        
        marginals, clique_degrees = self.compute_pixel_marginals(quantized_tile, cliques)
        
        marginal_entropy_term = 0.0
        for pixel, marginal in marginals.items():
            d_p = clique_degrees[pixel]
            
            if d_p <= 1:
                continue
            
            pixel_entropy = 0.0
            for p in marginal:
                if p > 0:
                    pixel_entropy -= p * np.log(p)
            
            marginal_entropy_term += (d_p - 1) * pixel_entropy
        
        log_z = expected_potential_sum + marginal_entropy_term
        
        return log_z
    
    def compute_entropy_bethe(self, quantized_tile, cliques, custom_potential=None):
        log_z = self.compute_partition_function_bethe(quantized_tile, cliques, custom_potential)
        expected_potential_sum = self.compute_expected_potential(quantized_tile, cliques, custom_potential)
        entropy = log_z - expected_potential_sum
        
        return entropy
    
    def compute_entropy(self, tile, custom_potential=None, method='bethe'):
        quantized_tile = self.mrf.quantize_intensities(tile)
        cliques = self.mrf.define_cliques(quantized_tile)
        
        if method == 'bethe':
            entropy = self.compute_entropy_bethe(quantized_tile, cliques, custom_potential)
        else:
            raise ValueError(f"Unknown entropy calculation method: {method}")
        
        return entropy