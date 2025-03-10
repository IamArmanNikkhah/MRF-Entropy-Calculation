# Running MRF-Entropy-Calculation in Google Colab

This guide provides specific instructions for running the optimized MRF-Entropy-Calculation code in Google Colab to achieve better performance.

## Key Performance Improvements

1. **Simplified Entropy Calculation**: Added a fast approximation mode that is much quicker but slightly less accurate
2. **GPU Memory Optimization**: Improved tensor handling to prevent out-of-memory errors
3. **Parallelized Processing**: Better batch handling and concurrent operations
4. **Reduced Computational Complexity**: Options to limit frames and tiles for faster processing
5. **Profiling & Debugging**: Added detailed timing outputs to identify bottlenecks

## Recommended Settings for Google Colab with Tesla T4

For optimal performance in Google Colab with a Tesla T4 GPU:

```bash
!python main.py --input /content/your_video.mp4 --output entropy_video.mp4 --num-tiles 34 --batch-size 2 --simplified
```

### Command-line Options Explanation:

- `--num-tiles 34`: Reduced from 55 to 34 (Fibonacci number) to decrease computational load by ~40%
- `--batch-size 2`: Small batch size to prevent GPU memory issues on Tesla T4
- `--simplified`: Uses a faster approximation algorithm for entropy calculation

## Testing and Debugging Options

If you want to test with a small portion of your video:

```bash
!python main.py --input /content/your_video.mp4 --output entropy_video.mp4 --num-tiles 34 --batch-size 1 --simplified --max-frames 50
```

For detailed timing information on each step:

```bash
!python main.py --input /content/your_video.mp4 --output entropy_video.mp4 --simplified --debug
```

## Speed vs. Accuracy Trade-offs

- **High Accuracy (Slower)**: Omit the `--simplified` flag for the original Bethe approximation algorithm
- **Balanced Performance**: Use `--num-tiles 34` with `--simplified` 
- **Maximum Speed**: Use `--num-tiles 21` with `--simplified` and `--batch-size 1`

## Troubleshooting

If you encounter CUDA memory errors:
1. Reduce batch size to 1: `--batch-size 1`
2. Use fewer tiles: `--num-tiles 21` (next Fibonacci number down)
3. Ensure you're using `--simplified` mode

If you encounter other errors or slow processing:
1. Add `--debug` to see detailed timing information
2. Use `--profile` to identify bottlenecks
3. Try limiting frame count with `--max-frames 100` for testing

## Memory Management

The Tesla T4 in Google Colab has ~16GB of VRAM, but not all of this is available for your computations. For optimal results:

1. Restart your runtime before running the script to clear memory
2. Don't run other CUDA operations in the same notebook
3. Use the simplified algorithm when possible

## Full List of Available Options

```
--input, -i: Path to input 360° video (required)
--output, -o: Path to output video with entropy visualization (required)
--num-tiles, -n: Number of tiles to partition the sphere into (default: 34)
--custom-potential, -c: Use custom potential function from custom_potential.py
--sample-rate, -s: Fraction of frames to sample for empirical distribution (default: 0.1)
--no-gpu: Disable GPU acceleration even if available
--batch-size, -b: Number of frames to process in a batch (default: 2)
--max-frames, -m: Maximum number of frames to process
--simplified, -S: Use simplified entropy calculation (faster but less accurate)
--debug, -d: Enable detailed debug output for performance analysis
--profile, -p: Run with cProfile to identify bottlenecks
