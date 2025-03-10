"""
Fibonacci Lattice implementation for equirectangular (ERP) 360° video tiling.

This module provides functions to:
1. Generate Fibonacci lattice points on a sphere
2. Compute Voronoi regions around these points
3. Map these regions to 2D ERP frames

The Fibonacci lattice provides an approximately equal-area distribution of points on a sphere.
"""

import numpy as np
from scipy.spatial import Voronoi, SphericalVoronoi
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def generate_fibonacci_points(N):
    """
    Generate N points on a sphere using the Fibonacci lattice algorithm.
    
    Parameters:
    -----------
    N : int
        Number of points to generate (ideally a Fibonacci number like 21, 34, 55)
    
    Returns:
    --------
    points_3d : ndarray of shape (N, 3)
        3D Cartesian coordinates of points on the unit sphere
    points_2d : ndarray of shape (N, 2)
        2D coordinates (longitude, latitude) in radians
    """
    # Golden ratio
    phi = (1 + np.sqrt(5)) / 2
    
    # Arrays to hold spherical and cartesian coordinates
    points_2d = np.zeros((N, 2))  # (longitude, latitude)
    points_3d = np.zeros((N, 3))  # (x, y, z)
    
    for i in range(N):
        # Calculate longitude (θ) and latitude (φ)
        longitude = 2 * np.pi * i / phi % (2 * np.pi)
        latitude = np.arccos(1 - 2 * (i + 0.5) / N)
        
        # Store spherical coordinates (longitude, latitude)
        points_2d[i, 0] = longitude
        points_2d[i, 1] = latitude
        
        # Convert to Cartesian coordinates
        x = np.sin(latitude) * np.cos(longitude)
        y = np.sin(latitude) * np.sin(longitude)
        z = np.cos(latitude)
        
        # Store Cartesian coordinates
        points_3d[i, 0] = x
        points_3d[i, 1] = y
        points_3d[i, 2] = z
        
    return points_3d, points_2d

def compute_voronoi_regions(points_3d):
    """
    Compute Voronoi regions around each lattice point on the sphere.
    
    Parameters:
    -----------
    points_3d : ndarray of shape (N, 3)
        3D Cartesian coordinates of points on the unit sphere
    
    Returns:
    --------
    sv : SphericalVoronoi object
        Contains Voronoi regions on the sphere
    """
    # Create SphericalVoronoi object
    sv = SphericalVoronoi(points_3d, radius=1.0, center=np.array([0, 0, 0]))
    sv.sort_vertices_of_regions()
    return sv

def map_to_erp_rectangles(sv, points_2d, width, height):
    """
    Map spherical Voronoi regions to rectangular regions in 2D ERP frame.
    
    Parameters:
    -----------
    sv : SphericalVoronoi object
        Contains Voronoi regions on the sphere
    points_2d : ndarray of shape (N, 2)
        2D coordinates (longitude, latitude) in radians
    width : int
        Width of the ERP frame in pixels
    height : int
        Height of the ERP frame in pixels
    
    Returns:
    --------
    tile_regions : list of lists
        Each inner list contains the pixel coordinates defining a tile in the ERP frame
    """
    # Initialize list to store regions
    tile_regions = []
    
    # For each region in the Voronoi diagram
    for i, region in enumerate(sv.regions):
        # Initialize list to store pixel coordinates for this region
        region_pixels = []
        
        # Convert region vertices from 3D to 2D (longitude, latitude)
        region_vertices_3d = sv.vertices[region]
        region_vertices_2d = []
        
        for vertex in region_vertices_3d:
            # Convert from Cartesian to spherical coordinates
            r = np.sqrt(np.sum(vertex**2))
            latitude = np.arccos(vertex[2] / r)
            longitude = np.arctan2(vertex[1], vertex[0]) % (2 * np.pi)
            
            # Convert to ERP pixel coordinates
            x = int((longitude / (2 * np.pi)) * width) % width
            y = int((latitude / np.pi) * height)
            
            region_vertices_2d.append((x, y))
        
        # Get the seed point (center of the region) in ERP coordinates
        center_longitude, center_latitude = points_2d[i]
        center_x = int((center_longitude / (2 * np.pi)) * width) % width
        center_y = int((center_latitude / np.pi) * height)
        
        # Add the region pixels
        tile_regions.append({
            'vertices': region_vertices_2d,
            'center': (center_x, center_y)
        })
    
    return tile_regions

def extract_tile_from_frame(frame, tile_region):
    """
    Extract a tile from a frame based on its region definition.
    
    Parameters:
    -----------
    frame : ndarray
        Video frame (2D array for grayscale, 3D for color)
    tile_region : dict
        Contains 'vertices' defining the tile boundary
    
    Returns:
    --------
    tile : ndarray
        Extracted tile from the frame
    mask : ndarray
        Binary mask showing which pixels belong to the tile
    """
    # Create a mask for the tile
    height, width = frame.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    
    # Convert vertices to numpy array for polygon filling
    vertices = np.array(tile_region['vertices'], dtype=np.int32)
    
    # Fill polygon
    import cv2
    cv2.fillPoly(mask, [vertices], 1)
    
    # Extract tile using the mask
    if len(frame.shape) == 3:  # Color frame
        tile = np.zeros_like(frame)
        for c in range(frame.shape[2]):
            tile[:,:,c] = frame[:,:,c] * mask
    else:  # Grayscale frame
        tile = frame * mask
    
    return tile, mask

def visualize_tiling(erp_frame, tile_regions):
    """
    Visualize the tiling on an ERP frame.
    
    Parameters:
    -----------
    erp_frame : ndarray
        ERP-projected video frame
    tile_regions : list of dicts
        Each dict contains 'vertices' defining a tile boundary
    
    Returns:
    --------
    visualization : ndarray
        Frame with tile boundaries overlaid
    """
    import cv2
    visualization = erp_frame.copy()
    
    # Draw each tile boundary
    for region in tile_regions:
        vertices = np.array(region['vertices'], dtype=np.int32)
        cv2.polylines(visualization, [vertices], True, (0, 255, 0), 2)
        
        # Draw the center point
        center_x, center_y = region['center']
        cv2.circle(visualization, (center_x, center_y), 5, (0, 0, 255), -1)
    
    return visualization
