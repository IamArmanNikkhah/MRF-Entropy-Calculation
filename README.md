# MRF-Entropy-Calculation

A system to compute spatial entropy for equirectangular (ERP) 360° videos using Markov Random Fields.

## Overview

This project implements a framework for analyzing spatial entropy in 360° videos by partitioning equirectangular video frames into equal-area tiles using the Fibonacci lattice algorithm and modeling pixel dependencies through quadwise Markov Random Fields (MRFs). The system outputs entropy metrics and generates a processed video with overlaid results.

## Features

- **Fibonacci Lattice Tiling**: Divides ERP-projected sphere into equal-area tiles
- **Quadwise MRF Structure**: Models dependencies using 2×2 non-overlapping pixel blocks
- **Statistical Modeling**: Quantizes pixel intensities and estimates empirical distributions
- **Entropy Computation**: Calculates entropy using Bethe approximation
- **CSV Output**: Generates detailed per-frame entropy metrics for all tiles
- **Processed Video**: Creates a visualization with entropy values and tile boundaries overlaid

## Requirements

- Python 3.7+
- OpenCV
- NumPy
- SciPy
- PyTorch
- tqdm

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/MRF-Entropy-Calculation.git
   cd MRF-Entropy-Calculation
   ```

2. Install the required dependencies:
   ```
   pip install numpy scipy opencv-python torch tqdm matplotlib
   ```

## Usage

Run the main script with the following arguments:

```
python main.py --input <input_video> --output <output_video> [options]
```

### Command-line Arguments

- `--input`, `-i`: Path to input 360° video (required)
- `--output`, `-o`: Path to output video with entropy visualization (required)
- `--num-tiles`, `-n`: Number of tiles to partition the sphere into (default: 55)
- `--custom-potential`, `-c`: Use custom potential function from custom_potential.py
- `--sample-rate`, `-s`: Fraction of frames to sample for empirical distribution (default: 0.1)
- `--no-gpu`: Disable GPU acceleration even if available

### Example

```
python main.py --input 360_video.mp4 --output entropy_video.mp4 --num-tiles 55
```

## Implementation Details

### Video Partitioning

The system partitions the sphere into N equal-area tiles using the Fibonacci lattice algorithm:

1. Generate N points on the sphere using:
   - θᵢ = 2πi/φ (longitude)
   - φᵢ = arccos(1 - 2i/N) (latitude)
   - where φ = (1+√5)/2 (golden ratio)

2. Compute Voronoi regions around each lattice point to define equal-area tiles.

3. Map each tile's Voronoi region to a rectangular region in the 2D ERP frame.

### MRF Structure

- Uses 2×2 non-overlapping pixel blocks as cliques
- Enforces adjacency constraints (Manhattan distance ≤1)
- Handles boundaries with spherical periodicity

### Statistical Modeling

- Quantizes grayscale values (0-255) to 5 discrete levels (0-4)
- Estimates empirical distribution by sampling frames
- Defines energy function: V_c(x_i, x_j, x_k, x_ℓ) = -log(p_data(x_i, x_j, x_k, x_ℓ) + ε) + β·f(x_i, x_j, x_k, x_ℓ)

### Entropy Computation

Calculates entropy using the formula:
H(X) = log Z - ∑_c E[log φ_c(X_c)]

Where Z is approximated using the Bethe approximation.

## Project Structure

```
MRF-Entropy-Calculation/
├── src/
│   ├── fibonacci_lattice.py    # Fibonacci lattice tiling implementation
│   ├── mrf_model.py            # MRF structure and energy functions
│   ├── entropy_calculator.py   # Entropy computation algorithms
│   └── video_processor.py      # Video frame extraction and processing
├── custom_potential.py         # Plugin interface for custom energy functions
├── main.py                     # Main execution script
└── README.md                   # Documentation
```

## Customizing Potential Functions

You can customize the smoothness function used in the energy calculation by modifying the `custom_potential.py` file. Several example implementations are provided:

- `custom_smoothness_function`: Default function using squared differences
- `absolute_difference_smoothness`: Uses absolute differences
- `variance_smoothness`: Based on variance of clique values
- `max_difference_smoothness`: Uses maximum difference between any pair

To use your custom function, uncomment the appropriate line at the bottom of the file and run with the `--custom-potential` flag.

## Output

The system generates two types of output:

1. **CSV File**: Per-frame entropy for all tiles with 4 decimal places:
   ```
   Frame, Tile_1_Entropy, ..., Tile_N_Entropy
   0, 2.4567, ..., 3.1021
   ```

2. **Processed Video**: Shows the original video with tile boundaries and average entropy per frame overlaid.

## Performance

The system is optimized to process at least 30 frames per second for 1080p video on an RTX 3090 GPU.
