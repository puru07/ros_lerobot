#!/usr/bin/env python3

import pandas as pd
import numpy as np
import cv2
import os
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
import argparse

#Converts a list of 6 robot joint angles into a 2D (x, y) position using a simplified kinematic model
#

def joint_to_cartesian(joint_angles):
    """
    Convert joint angles to approximate Cartesian coordinates.
    This is a simplified forward kinematics - you may need to adjust
    based on your specific robot configuration.
    
    Args:
        joint_angles: List of 6 joint angles [shoulder_lift_joint, elbow_joint, wrist_1_joint, wrist_2_joint, wrist_3_joint, shoulder_pan_joint]
    
    Returns:
        [x, y] coordinates
    """
    
    # Link lengths (approximate for UR5e in m)
    L1 = 0.425  # Upper arm length, Shoulder -> Elbow
    L2 = 0.392   # Forearm length, Elbow -> Wrist1
    L3 = 0.109    # Wrist length, Wrist1 -> Tip
    
    # Extract relevant joints
    shoulder_pan = joint_angles[0]
    shoulder_lift = joint_angles[1] 
    elbow = joint_angles[2]
    
    theta1 = shoulder_lift
    theta2 = elbow

    # total shoulder-lift plane angle
    total_angle = theta1 + theta2

    x = (L1 * np.cos(theta1) + L2 * np.cos(total_angle)) * np.cos(shoulder_pan)
    y = (L1 * np.cos(theta1) + L2 * np.cos(total_angle)) * np.sin(shoulder_pan)
    z = L1 * np.sin(theta1) + L2 * np.sin(total_angle)
    
    return [x, y]

#It returns a dummy action vector [vx, vy], meaning "go this much in x, and this much in y"
def create_dummy_action(current_pos, next_pos=None):
    """
    Create dummy action data. In real scenario, this would be actual control commands.
    """
    if next_pos is None:
        return [0.0, 0.0]
    
    # Simple velocity-based action
    dx = next_pos[0] - current_pos[0]
    dy = next_pos[1] - current_pos[1]
    
    # Scale to reasonable action range
    scale = 10.0
    return [dx * scale, dy * scale]

#
def calculate_reward(frame_index, total_frames):
    """
    Create dummy reward calculation.
    Replace with your actual reward function.
    """
    # Simple progression-based reward
    progress = frame_index / total_frames
    return progress

#The function converts a dataset saved in a CSV file into a Parquet format, while also creating videos from images.
def convert_csv_to_parquet(csv_file, image_dir, output_dir, episode_index=1):
    """
    Convert CSV dataset to Parquet format matching the original code structure.
    
    Args:
        csv_file: Path to input CSV file
        image_dir: Directory containing image files
        output_dir: Output directory for parquet and video files
        episode_index: Episode number for this dataset
    """
    
    # Read CSV file
    print(f"Reading CSV file: {csv_file}")
    df = pd.read_csv(csv_file)
    
    # Create output directories
    data_dir = Path(output_dir) / "data" / "chunk_000"
    video_dir = Path(output_dir) / "videos" / "chunk_000" / "observation.images"
    wrist_vid_dir = video_dir / "wrist"
    top_vid_dir = video_dir / "top"
    state_vid_dir = video_dir / "state"
    
    for dir_path in [data_dir, wrist_vid_dir, top_vid_dir, state_vid_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Prepare data for parquet
    parquet_data = []
    images_for_video = []
    
    print(f"Processing {len(df)} frames...")
    
    for idx, row in df.iterrows():
        # Extract joint angles
        joint_angles = [
            row['joint_shoulder_pan_joint'],
            row['joint_shoulder_lift_joint'],
            row['joint_elbow_joint'],
            row['joint_wrist_1_joint'],
            row['joint_wrist_2_joint'],
            row['joint_wrist_3_joint']
        ]
        
        # Convert to Cartesian coordinates
        cartesian_pos = joint_to_cartesian(joint_angles)
        
        # Create action (dummy for now)
        if idx < len(df) - 1:
            next_joint_angles = [
                df.iloc[idx + 1]['joint_shoulder_pan_joint'],
                df.iloc[idx + 1]['joint_shoulder_lift_joint'],
                df.iloc[idx + 1]['joint_elbow_joint'],
                df.iloc[idx + 1]['joint_wrist_1_joint'],
                df.iloc[idx + 1]['joint_wrist_2_joint'],
                df.iloc[idx + 1]['joint_wrist_3_joint']
            ]
            next_cartesian_pos = joint_to_cartesian(next_joint_angles)
            action = create_dummy_action(cartesian_pos, next_cartesian_pos)
        else:
            action = [0.0, 0.0]
        
        # Calculate reward and success
        reward = calculate_reward(idx, len(df))
        success = reward > 0.9  # Success threshold
        done = (idx == len(df) - 1) or success
        
        # Create parquet row
        parquet_row = {
            'observation.state': cartesian_pos,
            'action': action,
            'episode_index': episode_index,
            'frame_index': idx,
            'timestamp': row['image_timestamp'],
            'next.reward': reward,
            'next.done': done,
            'next.success': success,
            'index': idx,
            'task_index': 0
        }
        
        parquet_data.append(parquet_row)
        
        # Load image for video creation
        image_path = Path(image_dir) / row['image_filename']
        if image_path.exists():
            image = cv2.imread(str(image_path))
            if image is not None:
                # Resize to match original code format
                image = cv2.resize(image, (640, 360))
                images_for_video.append(image)
            else:
                print(f"Warning: Could not load image {image_path}")
        else:
            print(f"Warning: Image file not found {image_path}")
    
    # Create DataFrame and save as parquet
    parquet_df = pd.DataFrame(parquet_data)
    
    # Generate filename
    if episode_index <= 9:
        filename = f'episode_00000{episode_index}.parquet'
        video_filename = f'episode_00000{episode_index}.mp4'
    elif episode_index <= 99:
        filename = f'episode_0000{episode_index}.parquet'
        video_filename = f'episode_0000{episode_index}.mp4'
    elif episode_index <= 999:
        filename = f'episode_000{episode_index}.parquet'
        video_filename = f'episode_000{episode_index}.mp4'
    else:
        filename = f'episode_00{episode_index}.parquet'
        video_filename = f'episode_00{episode_index}.mp4'
    
    # Save parquet file
    table = pa.Table.from_pandas(parquet_df)
    pq.write_table(table, data_dir / filename)
    print(f"Parquet file saved: {data_dir / filename}")
    
    # Create videos
    if images_for_video:
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        fps = 10  # Match original code
        
        # Create wrist video (using the same images for all three for now)
        out1 = cv2.VideoWriter(str(wrist_vid_dir / video_filename), fourcc, fps, (640, 360))
        for frame in images_for_video:
            out1.write(frame)
        out1.release()
        print(f"Wrist video saved: {wrist_vid_dir / video_filename}")
        
        # Create top video
        out2 = cv2.VideoWriter(str(top_vid_dir / video_filename), fourcc, fps, (640, 360))
        for frame in images_for_video:
            out2.write(frame)
        out2.release()
        print(f"Top video saved: {top_vid_dir / video_filename}")
        
        # Create state video
        out3 = cv2.VideoWriter(str(state_vid_dir / video_filename), fourcc, fps, (640, 360))
        for frame in images_for_video:
            out3.write(frame)
        out3.release()
        print(f"State video saved: {state_vid_dir / video_filename}")
    
    print(f"Conversion complete! Episode {episode_index} processed.")
    print(f"Total frames: {len(df)}")
    print(f"Output directory: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Convert CSV dataset to Parquet format')
    parser.add_argument('--csv_file', required=True, help='Path to input CSV file')
    parser.add_argument('--image_dir', required=True, help='Directory containing image files')
    parser.add_argument('--output_dir', required=True, help='Output directory for converted data')
    parser.add_argument('--episode_index', type=int, default=1, help='Episode index number')
    
    args = parser.parse_args()
    
    # Verify inputs
    if not os.path.exists(args.csv_file):
        print(f"Error: CSV file not found: {args.csv_file}")
        return
    
    if not os.path.exists(args.image_dir):
        print(f"Error: Image directory not found: {args.image_dir}")
        return
    
    # Convert dataset
    convert_csv_to_parquet(
        csv_file=args.csv_file,
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        episode_index=args.episode_index
    )

if __name__ == "__main__":
    # Example usage if run directly
    if len(os.sys.argv) == 1:
        print("Example usage:")
        print("python convert_csv_to_parquet.py --csv_file dataset.csv --image_dir ./images --output_dir ./output --episode_index 1")
        print("\nOr modify the paths below and run directly:")
        
        # Uncomment and modify these paths for direct execution
        csv_file = "extracted_data/training_data.csv"
        image_dir = "extracted_data/new_image"
        output_dir = "extracted_data/modified_dataset"
        episode_index = 1
        convert_csv_to_parquet(csv_file, image_dir, output_dir, episode_index)
    else:
        main()