import numpy as np
from scipy.spatial import Voronoi, SphericalVoronoi
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def generate_fibonacci_points(N):
    phi = (1 + np.sqrt(5)) / 2
    points_2d = np.zeros((N, 2))
    points_3d = np.zeros((N, 3))
    
    for i in range(N):
        longitude = 2 * np.pi * i / phi % (2 * np.pi)
        latitude = np.arccos(1 - 2 * (i + 0.5) / N)
        
        points_2d[i, 0] = longitude
        points_2d[i, 1] = latitude
        
        x = np.sin(latitude) * np.cos(longitude)
        y = np.sin(latitude) * np.sin(longitude)
        z = np.cos(latitude)
        
        points_3d[i, 0] = x
        points_3d[i, 1] = y
        points_3d[i, 2] = z
        
    return points_3d, points_2d