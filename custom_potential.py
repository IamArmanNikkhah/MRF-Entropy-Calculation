"""
Plugin interface for custom energy functions in the MRF model.

This module allows users to define custom smoothness functions for the energy calculation
in the Markov Random Field model. The default smoothness function computes the sum of
squared differences between all 6 pixel pairs in a 2×2 clique. Users can define alternative
functions here that will be used in the entropy calculation.
"""

import numpy as np

def custom_smoothness_function(clique_values):
    """
    Example custom smoothness function.
    
    This is a template that users can modify to implement custom smoothness functions
    for the energy calculation in the MRF model.
    
    Parameters:
    -----------
    clique_values : tuple of int
        The four values in a clique (x_i, x_j, x_k, x_l)
        
    Returns:
    --------
    smoothness : float
        Smoothness value computed for the clique
    """
    # Default implementation (same as in QuadwiseMRF)
    x_i, x_j, x_k, x_l = clique_values
    
    # All possible pairs in a 2×2 clique
    pairs = [
        (x_i, x_j), (x_i, x_k), (x_i, x_l),
        (x_j, x_k), (x_j, x_l), (x_k, x_l)
    ]
    
    # Sum squared differences for all pairs
    smoothness = sum((a - b) ** 2 for a, b in pairs)
    
    return smoothness

# Alternative smoothness function examples:

def absolute_difference_smoothness(clique_values):
    """
    Alternative smoothness function using absolute differences instead of squared differences.
    
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
    
    # Sum absolute differences for all pairs
    smoothness = sum(abs(a - b) for a, b in pairs)
    
    return smoothness

def variance_smoothness(clique_values):
    """
    Smoothness function based on the variance of the clique values.
    
    Parameters:
    -----------
    clique_values : tuple of int
        The four values in a clique (x_i, x_j, x_k, x_l)
        
    Returns:
    --------
    smoothness : float
        Smoothness value computed for the clique
    """
    # Calculate variance
    mean = sum(clique_values) / len(clique_values)
    variance = sum((x - mean) ** 2 for x in clique_values) / len(clique_values)
    
    return variance

def max_difference_smoothness(clique_values):
    """
    Smoothness function using the maximum difference between any pair in the clique.
    
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
    
    # Maximum difference between any pair
    smoothness = max(abs(a - b) for a, b in pairs)
    
    return smoothness

# Select which function to use as the custom potential
# Uncomment the line corresponding to the function you want to use

# custom_potential = custom_smoothness_function
# custom_potential = absolute_difference_smoothness
# custom_potential = variance_smoothness
# custom_potential = max_difference_smoothness

# By default, use the standard smoothness function
custom_potential = custom_smoothness_function
