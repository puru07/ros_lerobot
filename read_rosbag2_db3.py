import rclpy
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions, StorageFilter

def main():
    rclpy.init()

    #edit this path
    bag_path = "lerobot_20250710_130929"

    storage_options = StorageOptions(uri=bag_path, storage_id='sqlite3')
    converter_options = ConverterOptions(input_serialization_format='cdr', output_serialization_format='cdr')

    reader = SequentialReader()
    reader.open(storage_options, converter_options)

    msg_type_joint = get_message("sensor_msgs/msg/JointState")

    storage_filter = StorageFilter()
    storage_filter.topics = ['/joint_states']
    reader.set_filter(storage_filter)

    while reader.has_next():
        topic, data, timestamp = reader.read_next()
        msg = deserialize_message(data, msg_type_joint)
        print(f"Timestamp: {timestamp / 1e9:.3f}s, Joint Names: {msg.name}, Positions: {msg.position}")

    rclpy.shutdown()

if __name__ == "__main__":
    main()
