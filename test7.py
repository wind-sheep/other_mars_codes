#!/usr/bin/env python3

import rospy
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

def depth_callback(msg):
    # Convert the ROS Image message to a NumPy array
    bridge = CvBridge()
    try:
        # # Convert the image to a NumPy array
        # depth_image = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        # # Convert from uint16 to float and scale to meters
        # depth_in_meters = depth_image.astype(np.float32) / 1000.0
        # rospy.loginfo("Depth image received and converted to meters.")
        # # Accessing a specific pixel for demonstration
        # height, width = depth_in_meters.shape
        # # example_pixel = (int(height / 2), int(width / 2))  # Center pixel
        # example_pixel = (209, 488)
        # rospy.loginfo(f"Depth at pixel {example_pixel}: {depth_in_meters[example_pixel]} meters")

        bridge = CvBridge()
        cv_img = bridge.imgmsg_to_cv2(msg, "passthrough")
        depth_array = np.array(cv_img, dtype=np.uint16)*0.001
        example_pixel = (209, 488)
        rospy.loginfo(f"Depth at pixel {example_pixel}: {depth_array[example_pixel]} meters")

    except Exception as e:
        rospy.logerr(f"Error processing depth image: {e}")

def main():
    rospy.init_node('depth_image_subscriber', anonymous=True)
    rospy.Subscriber("/camera/depth/image_rect_raw", Image, depth_callback)
    rospy.spin()

if __name__ == "__main__":
    main()
