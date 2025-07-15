#!/usr/bin/env python3

import rclpy
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions, StorageFilter

import numpy as np
import cv2
import os
import csv
from sensor_msgs.msg import Image, JointState

def ros_image_to_cv2(msg):
    """Convert ROS Image message to OpenCV format without cv_bridge"""
    if msg.encoding == "rgb8":
        # RGB8: 8-bit RGB
        img_array = np.frombuffer(msg.data, dtype=np.uint8)
        img_array = img_array.reshape((msg.height, msg.width, 3))
        # Convert RGB to BGR for OpenCV
        return cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    elif msg.encoding == "bgr8":
        # BGR8: 8-bit BGR (OpenCV format)
        img_array = np.frombuffer(msg.data, dtype=np.uint8)
        return img_array.reshape((msg.height, msg.width, 3))
    elif msg.encoding == "mono8":
        # Grayscale
        img_array = np.frombuffer(msg.data, dtype=np.uint8)
        return img_array.reshape((msg.height, msg.width))
    elif msg.encoding == "rgba8":
        # RGBA8: 8-bit RGBA
        img_array = np.frombuffer(msg.data, dtype=np.uint8)
        img_array = img_array.reshape((msg.height, msg.width, 4))
        # Convert RGBA to BGR
        return cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
    else:
        raise ValueError(f"Unsupported image encoding: {msg.encoding}")

def main():
    # Initialize ROS2
    rclpy.init()
    
    # === CONFIGURE THESE PATHS ===
    bag_path = "lerobot_20250710_130929"  # Your bag folder path
    output_dir = os.path.join(bag_path, "extracted_data")
    sync_tolerance = 0.1  # seconds
    
    # Create output directories
    images_folder = os.path.join(output_dir, "new_image")
    os.makedirs(images_folder, exist_ok=True)
    csv_file = os.path.join(output_dir, "training_data.csv")
    
    # Setup rosbag reader
    storage_options = StorageOptions(uri=bag_path, storage_id='sqlite3')
    converter_options = ConverterOptions(
        input_serialization_format='cdr', 
        output_serialization_format='cdr'
    )
    
    reader = SequentialReader()
    
    try:
        reader.open(storage_options, converter_options)
        print(f"Successfully opened bag: {bag_path}")
    except Exception as e:
        print(f"Error opening bag file: {e}")
        return
    
    # Get available topics
    topics = reader.get_all_topics_and_types()
    print("Available topics:")
    for topic in topics:
        print(f" - {topic.name}: {topic.type}")
    
    # Verify required topics exist
    available_topic_names = [topic.name for topic in topics]
    required_topics = ['/camera/camera/color/image_raw', '/joint_states']
    
    missing_topics = [t for t in required_topics if t not in available_topic_names]
    if missing_topics:
        print(f"Error: Required topics not found: {missing_topics}")
        return
    
    # Get message types
    msg_type_image = get_message("sensor_msgs/msg/Image")
    msg_type_joint = get_message("sensor_msgs/msg/JointState")
    
    # Set topic filter
    storage_filter = StorageFilter()
    storage_filter.topics = required_topics
    reader.set_filter(storage_filter)
    
    # Storage for messages with timestamps
    image_messages = []
    joint_messages = []
    
    print("Reading bag file...")
    
    # Read all messages
    while reader.has_next():
        topic, data, timestamp = reader.read_next()
        
        # Convert timestamp from nanoseconds to seconds
        timestamp_sec = timestamp / 1e9
        
        try:
            if topic == '/camera/camera/color/image_raw':
                msg = deserialize_message(data, msg_type_image)
                image_messages.append((timestamp_sec, msg))
                
            elif topic == '/joint_states':
                msg = deserialize_message(data, msg_type_joint)
                joint_messages.append((timestamp_sec, msg))
                
        except Exception as e:
            print(f"Error deserializing message from {topic}: {e}")
            continue
    
    print(f"Found {len(image_messages)} image messages and {len(joint_messages)} joint messages")
    
    # Sort messages by timestamp
    image_messages.sort(key=lambda x: x[0])
    joint_messages.sort(key=lambda x: x[0])
    
    if not image_messages or not joint_messages:
        print("Error: No valid messages found")
        return
    
    # Synchronize messages and extract data
    synchronized_data = []
    joint_names = None
    
    print("Synchronizing and extracting data...")
    
    for img_idx, (img_timestamp, img_msg) in enumerate(image_messages):
        # Find closest joint message
        closest_joint = None
        min_time_diff = float('inf')
        
        for joint_timestamp, joint_msg in joint_messages:
            time_diff = abs(img_timestamp - joint_timestamp)
            if time_diff < min_time_diff:
                min_time_diff = time_diff
                closest_joint = (joint_timestamp, joint_msg)
        
        # Check if synchronization is within tolerance
        if closest_joint and min_time_diff <= sync_tolerance:
            joint_timestamp, joint_msg = closest_joint
            
            # Extract and save image
            try:
                # Convert ROS image to OpenCV format
                cv_image = ros_image_to_cv2(img_msg)
                
                # Save image
                image_filename = f"frame_{img_idx:06d}.png"
                image_path = os.path.join(images_folder, image_filename)
                cv2.imwrite(image_path, cv_image)
                
                # Store joint names (only once)
                if joint_names is None:
                    joint_names = list(joint_msg.name)
                
                # Prepare data row
                data_row = {
                    'frame_id': img_idx,    #Generated: Easy indexing for your dataset
                    'image_filename': image_filename,   #Generated: Links CSV rows to actual image files
                    'image_timestamp': img_timestamp,   #Extracted: Shows when data was recorded
                    'joint_timestamp': joint_timestamp, #Extracted
                    'sync_time_diff': min_time_diff     #Computed: |image_timestamp - joint_timestamp|
                }
                
                # Add joint angles
                for name, position in zip(joint_msg.name, joint_msg.position):
                    data_row[f'joint_{name}'] = position
                
                synchronized_data.append(data_row)
                
                if img_idx % 100 == 0:
                    print(f"Processed {img_idx + 1} images...")
                    
            except Exception as e:
                print(f"Error processing image {img_idx}: {e}")
                continue
    
    # Write CSV file
    if synchronized_data:
        print(f"Writing {len(synchronized_data)} synchronized data points to CSV...")
        
        with open(csv_file, 'w', newline='') as f:
            fieldnames = ['frame_id', 'image_filename', 'image_timestamp', 
                         'joint_timestamp', 'sync_time_diff']
            
            # Add joint column names
            if joint_names:
                fieldnames.extend([f'joint_{name}' for name in joint_names])
            
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(synchronized_data)
        
        print(f"Successfully extracted {len(synchronized_data)} synchronized data points")
        print(f"Images saved to: {images_folder}")
        print(f"CSV data saved to: {csv_file}")
        
        # Print statistics
        print(f"\nStatistics:")
        print(f"- Total synchronized pairs: {len(synchronized_data)}")
        print(f"- Average sync time difference: {np.mean([d['sync_time_diff'] for d in synchronized_data]):.4f} seconds")
        print(f"- Max sync time difference: {np.max([d['sync_time_diff'] for d in synchronized_data]):.4f} seconds")
        if joint_names:
            print(f"- Joint names: {joint_names}")
    
    else:
        print("No synchronized data found!")
    
    # Cleanup
    rclpy.shutdown()

if __name__ == "__main__":
    main()