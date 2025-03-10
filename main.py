#!/usr/bin/env python3
"""
Main script for MRF-based spatial entropy calculation on 360° videos.

This script provides a command-line interface to run the entire entropy calculation pipeline.
It partitions equirectangular video frames into equal-area tiles using a Fibonacci lattice,
models pixel dependencies via quadwise Markov Random Fields, and computes entropy metrics.

Usage:
    python main.py --input <input_video> --output <output_video> [--num-tiles <n>] 
                   [--custom-potential] [--sample-rate <r>] [--no-gpu] [--batch-size <b>]
                   [--max-frames <m>] [--profile] [--simplified] [--debug]

Example:
    python main.py --input 360_video.mp4 --output entropy_video.mp4 --num-tiles 34 --batch-size 2 --simplified
"""

import os
import sys
import argparse
import importlib.util
import time
import torch
import gc
import cProfile
import pstats
import io
from pstats import SortKey

from src.video_processor import VideoProcessor
from src.entropy_calculator import DEBUG_TIMING
import custom_potential

def is_fibonacci_number(n):
    """Check if a number is in the Fibonacci sequence."""
    a, b = 0, 1
    while b < n:
        a, b = b, a + b
    return b == n

def print_gpu_info():
    """Print detailed information about the available CUDA GPU."""
    if not torch.cuda.is_available():
        print("No CUDA GPU available")
        return
    
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # in GB
    current_device = torch.cuda.current_device()
    device_count = torch.cuda.device_count()
    
    print("\n===== GPU Information =====")
    print(f"GPU: {gpu_name}")
    print(f"Total memory: {gpu_mem:.2f} GB")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"PyTorch CUDA: {torch.backends.cudnn.version()}")
    print(f"Current device: {current_device}")
    print(f"Device count: {device_count}")
    print(f"CUDNN enabled: {torch.backends.cudnn.enabled}")
    print("===========================\n")

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Calculate spatial entropy for equirectangular 360° videos using Markov Random Fields."
    )
    
    # Required arguments
    parser.add_argument("--input", "-i", required=True, help="Path to input 360° video")
    parser.add_argument("--output", "-o", required=True, help="Path to output video with entropy visualization")
    
    # Optional arguments
    parser.add_argument("--num-tiles", "-n", type=int, default=34, 
                        help="Number of tiles to partition the sphere into (ideally a Fibonacci number, default: 34)")
    parser.add_argument("--custom-potential", "-c", action="store_true", 
                        help="Use custom potential function from custom_potential.py")
    parser.add_argument("--sample-rate", "-s", type=float, default=0.1, 
                        help="Fraction of frames to sample for empirical distribution (default: 0.1)")
    parser.add_argument("--no-gpu", action="store_true", 
                        help="Disable GPU acceleration even if available")
    parser.add_argument("--batch-size", "-b", type=int, default=2,
                        help="Number of frames to process in a batch (default: 2)")
    parser.add_argument("--max-frames", "-m", type=int, default=None,
                        help="Maximum number of frames to process (default: process all frames)")
    parser.add_argument("--profile", "-p", action="store_true",
                        help="Run the script with cProfile to identify bottlenecks")
    parser.add_argument("--simplified", "-S", action="store_true",
                        help="Use simplified entropy calculation (faster but less accurate)")
    parser.add_argument("--debug", "-d", action="store_true",
                        help="Enable detailed debug output for performance analysis")
    
    return parser.parse_args()

def setup_debug_flags(debug_enabled):
    """Set up debug flags in the codebase."""
    # Import the module
    import src.entropy_calculator as ec
    
    # Set the debug flag
    ec.DEBUG_TIMING = debug_enabled
    
    if debug_enabled:
        print("Debug mode enabled - detailed timing information will be shown")

def run_with_profiling(args, potential_func):
    """Run the video processing with profiling enabled."""
    print("\nRunning with profiling enabled...\n")
    
    # Create a profile object
    pr = cProfile.Profile()
    pr.enable()
    
    # Initialize video processor
    processor = VideoProcessor(
        input_path=args.input,
        output_path=args.output,
        num_tiles=args.num_tiles,
        use_gpu=not args.no_gpu,
        batch_size=args.batch_size
    )
    
    # Run the entropy calculation pipeline with simplified mode if requested
    result = processor.run(
        custom_potential=potential_func,
        sample_rate=args.sample_rate,
        use_simplified=args.simplified,
        max_frames=args.max_frames
    )
    
    # Disable profiling
    pr.disable()
    
    # Print profile results
    print("\n===== Profiling Results =====")
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats(SortKey.CUMULATIVE)
    ps.print_stats(20)  # Print top 20 functions by cumulative time
    print(s.getvalue())
    
    return result

def main():
    """Main function to run the entropy calculation pipeline."""
    # Parse command-line arguments
    args = parse_arguments()
    
    # Set up debug flags if requested
    setup_debug_flags(args.debug)
    
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
    
    # Print GPU information if available
    if not args.no_gpu and torch.cuda.is_available():
        print_gpu_info()

    # Print processing mode
    if args.simplified:
        print("\n⚠️ Using SIMPLIFIED entropy calculation mode (faster but less accurate)")
    
    # Get custom potential function if requested
    potential_func = None
    if args.custom_potential:
        if hasattr(custom_potential, 'custom_potential'):
            potential_func = custom_potential.custom_potential
            print(f"Using custom potential function: {potential_func.__name__}")
        else:
            print("Warning: custom_potential.py does not define 'custom_potential'. Using default.")
    
    # Run with profiling if requested
    if args.profile:
        start_time = time.time()
        try:
            result = run_with_profiling(args, potential_func)
            end_time = time.time()
            duration = end_time - start_time
        except Exception as e:
            print(f"Error during profiled processing: {str(e)}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    else:
        # Initialize video processor
        processor = VideoProcessor(
            input_path=args.input,
            output_path=args.output,
            num_tiles=args.num_tiles,
            use_gpu=not args.no_gpu,
            batch_size=args.batch_size
        )
        
        # Run the entropy calculation pipeline
        print(f"\nProcessing '{args.input}' using {args.num_tiles} tiles...\n")
        start_time = time.time()
        
        try:
            result = processor.run(
                custom_potential=potential_func,
                sample_rate=args.sample_rate,
                use_simplified=args.simplified,
                max_frames=args.max_frames
            )
            
            end_time = time.time()
            duration = end_time - start_time
        except Exception as e:
            print(f"Error during processing: {str(e)}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
    # Print summary
    print(f"\nTotal processing completed in {duration:.2f} seconds")
    print(f"Output video saved to: {result['video_path']}")
    print(f"Entropy values saved to: {result['csv_path']}")
    
    # Clean up
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
