from moviepy.editor import VideoFileClip, concatenate_videoclips
import os
import argparse

def insert_video_gap(video1_path, video2_path, frame_start, frame_end, output_path):
    """
    Inserts the second video into a gap between two frames of the first video.

    Parameters:
    - video1_path (str): Path to the first video.
    - video2_path (str): Path to the second video.
    - frame_start (int): Frame number where the gap starts.
    - frame_end (int): Frame number where the gap ends.
    - output_path (str): Path to save the output video.
    """
    # Load video clips
    print("Loading video 1...")
    video1 = VideoFileClip(video1_path)
    print("Loading video 2...")
    video2 = VideoFileClip(video2_path)

    # Calculate start and end times of the frames (in seconds)
    fps = video1.fps  # Frames per second of the first video
    start_time = frame_start / fps
    end_time = frame_end / fps

    # Cut video1 into two parts: before and after the gap
    print("Cutting video 1...")
    part1 = video1.subclip(0, start_time)  # From start to frame_start
    part2 = video1.subclip(end_time, video1.duration)  # From frame_end to end

    # Combine: part1 -> video2 -> part2
    print("Concatenating videos...")
    final_video = concatenate_videoclips([part1, video2, part2])

    # Write final output
    print("Saving the final video...")
    final_video.write_videofile(output_path, codec="libx264", fps=fps)

    print("Video processing complete! Final video saved to:", output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Insert a video into a gap between frames of another video.")
    parser.add_argument("--video1", type=str, required=True, help="Path to the first video.")
    parser.add_argument("--video2", type=str, required=True, help="Path to the second video.")
    parser.add_argument("--frame_start", type=int, required=True, help="Starting frame number for the gap.")
    parser.add_argument("--frame_end", type=int, required=True, help="Ending frame number for the gap.")
    parser.add_argument("--output", type=str, required=True, help="Path to save the output video.")

    args = parser.parse_args()

    # Check if input files exist
    if not os.path.exists(args.video1) or not os.path.exists(args.video2):
        print("Error: One or both video files not found.")
    else:
        # Run the video processing function
        insert_video_gap(args.video1, args.video2, args.frame_start, args.frame_end, args.output)
