# Video Frame Organizer

A Python tool for reconstructing corrupted or shuffled videos by intelligently reordering frames using structural similarity analysis and machine learning clustering techniques.

## Overview

This tool addresses the problem of corrupted videos where frames have been shuffled or contain anomalous content. It uses computer vision and machine learning algorithms to:

1. **Remove anomalous frames** using DBSCAN clustering
2. **Calculate frame similarities** using Structural Similarity Index (SSIM)
3. **Find optimal frame ordering** through similarity-based sequencing
4. **Reconstruct the video** with proper frame continuity

## Features

- **Automatic anomaly detection**: Removes corrupted or significantly different frames
- **Intelligent frame sequencing**: Uses SSIM to determine optimal frame order
- **Consistency optimization**: Evaluates multiple sequences to find the smoothest reconstruction
- **Command-line interface**: Easy-to-use CLI with validation and error handling
- **Flexible output**: Customizable output paths and video parameters

## Requirements

```bash
pip install opencv-python matplotlib scikit-learn numpy tqdm scikit-image
```

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd video-frame-organizer
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```bash
python main.py -i input_corrupted_video.mp4 -o output_reconstructed_video.mp4
```

### Command Line Options

- `-i, --input`: Path to the input corrupted video file (required)
- `-o, --output`: Path for the output reconstructed video file (required)

### Examples

```bash
# Basic reconstruction
python main.py -i corrupted_video.mp4 -o fixed_video.mp4

# Using full argument names
python main.py --input input.mp4 --output output.mp4
```

## How It Works

### 1. Frame Extraction
- Extracts all frames from the input video
- Resizes frames to 256x256 pixels for consistent processing
- Converts to BGR color format for OpenCV compatibility

### 2. Anomaly Detection
- Uses DBSCAN clustering on flattened pixel values
- Automatically determines optimal epsilon parameter
- Removes frames labeled as outliers (-1 cluster)

### 3. Similarity Analysis
- Computes pairwise SSIM between all filtered frames
- Creates similarity matrix for frame comparison
- Generates sorted similarity indices for each frame

### 4. Sequence Optimization
- Tests all possible starting frames
- Builds sequences by selecting most similar unused frames
- Calculates consistency scores based on similarity transitions

### 5. Video Reconstruction
- Selects the sequence with the lowest consistency score
- Reconstructs video using optimal frame ordering
- Outputs MP4 format with 30 FPS

## Algorithm Details

The reconstruction process relies on several key algorithms:

- **DBSCAN Clustering**: Density-based clustering to identify and remove anomalous frames
- **Structural Similarity Index (SSIM)**: Measures perceptual similarity between frames
- **Greedy Sequencing**: Builds frame sequences by always choosing the most similar next frame
- **Consistency Scoring**: Evaluates sequence quality based on similarity transition smoothness

## Performance Considerations

- Processing time scales quadratically with the number of frames (O(n²))
- Memory usage depends on video length and frame resolution
- Recommended for videos with up to 1000-2000 frames for reasonable processing times
- Consider reducing frame resolution for very large videos

## Limitations

- Assumes frames belong to a single continuous sequence
- May not work well with videos containing scene cuts or transitions
- Performance degrades with very long videos due to computational complexity
- Requires sufficient frame similarity for effective reconstruction

## File Structure

```
video-frame-organizer/
├── main.py              # Main application script
├── README.md           # This file
├── requirements.txt    # Python dependencies
└── examples/           # Example videos (if included)
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built using OpenCV for video processing
- Scikit-learn for machine learning algorithms
- Scikit-image for SSIM calculation
- NumPy for efficient numerical computations

## Support

If you encounter issues or have questions:

1. Check the [Issues](../../issues) page for existing solutions
2. Create a new issue with detailed description and error messages
3. Include sample video files if possible (ensure no sensitive content)

---

**Note**: This tool is designed for educational and research purposes. Results may vary depending on video content and corruption patterns.
