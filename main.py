import argparse
from matplotlib import pyplot as plt
import cv2
from sklearn.cluster import *
import numpy as np
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
import os
"testing1"
"testin2"
"testin3"
"testin4"
"testing5"

def extract_frames(vid_name):
    """
    Extract all frames from a video file and resize them to a standard size.

    Args:
        vid_name (str): Path to the input video file. Defaults to "/content/corrupted_video.mp4"

    Returns:
        list: List of numpy arrays, where each array represents a frame image
              resized to (256, 256, 3) in BGR color format

    Purpose:
        Reads a video file frame by frame and converts them into a list of images
        with standardized dimensions for further processing.
    """
    vidcap = cv2.VideoCapture(vid_name)
    success, image = vidcap.read()
    count = 0
    ftr_ls = []
    img_ls = []

    while success:
        if success:
            img_ls.append(cv2.resize(image,(256,256)))
            success, image = vidcap.read()
    return img_ls


def remove_anomalies(img_ls):
    """
    Remove anomalous/corrupted frames from the image list using DBSCAN clustering.

    Args:
        img_ls (list): List of numpy arrays representing video frames,
                      each with shape (256, 256, 3)

    Returns:
        list: Filtered list of numpy arrays containing only non-anomalous frames
              (frames that belong to the main cluster, excluding outliers)

    Purpose:
        Uses DBSCAN clustering to identify and remove corrupted or significantly
        different frames by flattening images and clustering them based on pixel values.
        Frames labeled as -1 (noise/outliers) are removed.
    """
    for i in range(10000, 100000, 1000):
        kmeans1 = DBSCAN(eps=i).fit([i.reshape(-1) for i in img_ls])
        if len(np.unique(kmeans1.labels_)) == 2:
            break
    img_filtered = []
    for i, check in zip(img_ls, kmeans1.labels_):
        if check != -1:
            img_filtered.append(i)
    return img_filtered


def get_similarity_matrix(img_filtered):
    """
    Compute pairwise structural similarity (SSIM) matrix between all filtered frames.

    Args:
        img_filtered (list): List of numpy arrays representing filtered video frames,
                            each with shape (256, 256, 3)

    Returns:
        tuple: A tuple containing:
            - similarity (numpy.ndarray): 2D array of shape (n_frames, n_frames)
                                        containing SSIM values between frame pairs (float64)
            - sim_argsort (numpy.ndarray): 2D array of shape (n_frames, n_frames)
                                         containing indices of frames sorted by similarity
                                         for each frame (int64)

    Purpose:
        Calculates structural similarity between every pair of frames to determine
        which frames are most similar to each other. This helps in ordering frames
        in a sequence that maintains visual continuity.
    """
    n_good = len(img_filtered)
    similarity = np.asarray([[0.0] * n_good] * n_good).astype(float)
    for en1, i in tqdm(enumerate(img_filtered)):
        for en2, j in enumerate(img_filtered):
            similarity[en1][en2] = ssim(
                cv2.cvtColor(i, cv2.COLOR_BGR2GRAY),
                cv2.cvtColor(j, cv2.COLOR_BGR2GRAY),
                multichannel=False,
            )
    sim_argsort = np.argsort(similarity, -1)
    return similarity, sim_argsort


def find_best_track_for_start(start, sim_argsort):
    """
    Find the optimal sequence of frames starting from a given frame index.

    Args:
        start (int): Index of the starting frame for the sequence
        sim_argsort (numpy.ndarray): 2D array containing frame indices sorted by
                                   similarity for each frame, shape (n_frames, n_frames)

    Returns:
        list: List of integers representing the optimal sequence of frame indices
              starting from the given start frame, includes all available frames

    Purpose:
        Creates a sequence of frames by always choosing the most similar unused frame
        as the next frame in the sequence. This helps reconstruct a smooth video
        from shuffled frames.
    """
    track = [start]
    while len(track) != len(sim_argsort[0]):
        for i in sim_argsort[track[-1]][::-1]:
            if i not in track:
                track.append(i.item())
                break
    return track


def consistent_change(track, similarity):
    """
    Calculate the consistency score for a given frame sequence.

    Args:
        track (list): List of integers representing frame indices in sequence order
        similarity (numpy.ndarray): 2D similarity matrix of shape (n_frames, n_frames)
                                  containing SSIM values between frame pairs

    Returns:
        float: Consistency score representing the total variation in similarity
               between consecutive frames (lower is better)

    Purpose:
        Measures how smoothly the similarity changes between consecutive frames
        in a sequence. A lower score indicates more consistent transitions,
        which suggests a better frame ordering.
    """
    cost = []
    for i, j in zip(track[:-1], track[1:]):
        cost.append(similarity[i][j].item())
    jump = 0
    for i, j in zip(cost[:-1], cost[1:]):
        jump += abs(i - j)
    return jump


def get_best_track(similarity, sim_argsort):
    """
    Find the best frame sequence by evaluating all possible starting frames.

    Args:
        similarity (numpy.ndarray): 2D similarity matrix of shape (n_frames, n_frames)
                                  containing SSIM values between frame pairs
        sim_argsort (numpy.ndarray): 2D array containing frame indices sorted by
                                   similarity for each frame

    Returns:
        tuple: A tuple containing:
            - track_dk (dict): Dictionary mapping starting frame indices (int) to
                              their corresponding optimal sequences (list of int)
            - cost_dk (dict): Dictionary mapping starting frame indices (int) to
                             their consistency scores (float)

    Purpose:
        Evaluates all possible starting frames and finds their optimal sequences,
        then calculates consistency scores to determine which sequence provides
        the smoothest video reconstruction.
    """
    track_dk = {}
    cost_dk = {}
    for start in range(similarity.shape[0]):
        track = find_best_track_for_start(start, sim_argsort)
        track_dk[start] = track
        cost_dk[start] = consistent_change(track, similarity)
    return track_dk, cost_dk


def images_to_video(image_ls, output_path, fps=30):
    """
    Convert a list of images to a video file.

    Args:
        image_ls (list): List of numpy arrays representing frames, each with
                        shape (height, width, 3) in BGR format
        output_path (str): Path where the output video file will be saved
        fps (int): Frames per second for the output video. Defaults to 30

    Returns:
        None: Function saves video to disk and prints confirmation message

    Purpose:
        Takes a sequence of image frames and writes them to a video file
        using OpenCV's VideoWriter with MP4 format.
    """
    height, width, _ = image_ls[0].shape

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame in image_ls:
        out.write(frame)

    out.release()
    print(f"Video saved to {output_path}")


def video_from_order(img_filtered, cost_dk, track_dk, output):
    """
    Create a video using the best frame sequence determined by consistency analysis.

    Args:
        img_filtered (list): List of numpy arrays representing filtered frames
        cost_dk (dict): Dictionary mapping starting frame indices to consistency scores
        track_dk (dict): Dictionary mapping starting frame indices to frame sequences
        output (str): Path for the output video file

    Returns:
        None: Function creates and saves the video file

    Purpose:
        Selects the frame sequence with the lowest consistency score (best ordering)
        and creates a video from those frames in the determined order.
    """
    video_ls = []
    for i in track_dk[min(cost_dk, key=cost_dk.get)]:
        video_ls.append(img_filtered[i])
    images_to_video(video_ls, output)


def organize_frames_for_shuffled_frames_outliers(input, output):
    """
    Complete pipeline to reconstruct a video from corrupted/shuffled frames.

    Args:
        input (str): Path to the input corrupted video file
        output (str): Path where the reconstructed video will be saved

    Returns:
        None: Function processes the video and saves the result

    Purpose:
        Main function that orchestrates the entire video reconstruction process:
        1. Extracts frames from corrupted video
        2. Removes anomalous frames using clustering
        3. Computes frame similarities
        4. Finds optimal frame ordering
        5. Creates reconstructed video with proper frame sequence
    """
    frames = extract_frames(input)
    img_filtered = remove_anomalies(frames)
    similarity_matrix, similarity_sorted = get_similarity_matrix(img_filtered)
    track_dk, cost_dk = get_best_track(similarity_matrix, similarity_sorted)
    video_from_order(img_filtered, cost_dk, track_dk, output)



def main():
    """
    Command-line interface for the video frame organizer.
    """
    parser = argparse.ArgumentParser(
        description="Reconstruct corrupted/shuffled videos by organizing frames using structural similarity",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py -i corrupted_video.mp4 -o fixed_video.mp4
  python main.py --input input.mp4 --output output.mp4 --fps 24
  python main.py -i video.mp4 -o output.mp4 --fps 60
        """
    )
    
    parser.add_argument(
        "-i", "--input",
        required=True,
        help="Path to the input corrupted video file"
    )
    
    parser.add_argument(
        "-o", "--output", 
        required=True,
        help="Path for the output reconstructed video file"
    )
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' does not exist.")
        sys.exit(1)
    
    # Validate output directory
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        print(f"Error: Output directory '{output_dir}' does not exist.")
        sys.exit(1)
    
    # Run the video reconstruction
    organize_frames_for_shuffled_frames_outliers(args.input, args.output)


if __name__ == "__main__":
    main()
