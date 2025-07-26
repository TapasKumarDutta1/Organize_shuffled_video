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


## Installation

1. Clone this repository:
```bash
git clone https://github.com/TapasKumarDutta1/Organize_shuffled_video.git
cd Organize_shuffled_video
```

2. Install dependencies:
```bash
pip install installation.tar.gz
```

## Usage

### Examples

```bash
# Basic reconstruction
python main.py -i corrupted_video.mp4 -o fixed_video.mp4

# Using full argument names
python main.py --input corrupted_video.mp4 --output output.mp4
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

## File Structure

```
video-frame-organizer/
├── main.py                 # Main application script
├── README.md              # This documentation
├── LICENSE                # MIT License
├── installation.tar.gz    # Pre-built package for pip installation
└── corrupted_video.mp4    # Sample corrupted video for testing
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

