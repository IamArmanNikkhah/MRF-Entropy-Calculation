#!/usr/bin/env python3
"""
Main script for MRF-based spatial entropy calculation on 360° videos.

This script provides a command-line interface to run the entire entropy calculation pipeline.
It partitions equirectangular video frames into equal-area tiles using a Fibonacci lattice,
models pixel dependencies via quadwise Markov Random Fields, and computes entropy metrics.

Usage:
    python main.py --input <input_video> --output <output_video> [--num-tiles <n>] 
                   [--custom-potential] [--sample-rate <r>] [--no-gpu]

Example:
    python main.py --input 360_video.mp4 --output entropy_video.mp4 --num-tiles 55
"""

import os
import sys
import argparse
import importlib.util
import time

from src.video_processor import VideoProcessor
import custom_potential

def is_fibonacci_number(n):
    """Check if a number is in the Fibonacci sequence."""
    a, b = 0, 1
    while b < n:
        a, b = b, a + b
    return b == n

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Calculate spatial entropy for equirectangular 360° videos using Markov Random Fields."
    )
    
    # Required arguments
    parser.add_argument("--input", "-i", required=True, help="Path to input 360° video")
    parser.add_argument("--output", "-o", required=True, help="Path to output video with entropy visualization")
    
    # Optional arguments
    parser.add_argument("--num-tiles", "-n", type=int, default=55, 
                        help="Number of tiles to partition the sphere into (ideally a Fibonacci number, default: 55)")
    parser.add_argument("--custom-potential", "-c", action="store_true", 
                        help="Use custom potential function from custom_potential.py")
    parser.add_argument("--sample-rate", "-s", type=float, default=0.1, 
                        help="Fraction of frames to sample for empirical distribution (default: 0.1)")
    parser.add_argument("--no-gpu", action="store_true", 
                        help="Disable GPU acceleration even if available")
    
    return parser.parse_args()

def main():
    """Main function to run the entropy calculation pipeline."""
    # Parse command-line arguments
    args = parse_arguments()
    
    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' does not exist")
        sys.exit(1)
    
    # Check if number of tiles is a Fibonacci number (approximately)
    if not is_fibonacci_number(args.num_tiles):
        print(f"Warning: {args.num_tiles} is not a Fibonacci number. Recommended values are 13, 21, 34, 55, 89, 144.")
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Get custom potential function if requested
    potential_func = None
    if args.custom_potential:
        if hasattr(custom_potential, 'custom_potential'):
            potential_func = custom_potential.custom_potential
            print(f"Using custom potential function: {potential_func.__name__}")
        else:
            print("Warning: custom_potential.py does not define 'custom_potential'. Using default.")
    
    # Initialize video processor
    processor = VideoProcessor(
        input_path=args.input,
        output_path=args.output,
        num_tiles=args.num_tiles,
        use_gpu=not args.no_gpu
    )
    
    # Run the entropy calculation pipeline
    print(f"\nProcessing '{args.input}' using {args.num_tiles} tiles...\n")
    start_time = time.time()
    
    try:
        result = processor.run(
            custom_potential=potential_func,
            sample_rate=args.sample_rate
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"\nProcessing completed in {duration:.2f} seconds")
        print(f"Output video saved to: {result['video_path']}")
        print(f"Entropy values saved to: {result['csv_path']}")
        
    except Exception as e:
        print(f"Error during processing: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
