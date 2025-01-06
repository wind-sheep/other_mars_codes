#!/usr/bin/env python3

import rospy
import time
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
from ultralytics import YOLO

# Load your YOLO model
model = YOLO(r"/home/stu/yolo/best.pt")  # Replace with your trained model file

# Initialize the CV Bridge
bridge = CvBridge()

# Global variables to store the last detected coordinates and depth
last_x_center = None
last_y_center = None
last_depth = None

# Record start time
start_time = None

# Callback for color image processing
def image_callback(msg):
    global last_x_center, last_y_center, start_time

    if start_time is None:
        start_time = time.time()  # Start the timer

    try:
        # Convert the ROS Image message to a CV2 image
        frame = bridge.imgmsg_to_cv2(msg, "bgr8")
    except CvBridgeError as e:
        rospy.logerr(f"Error converting image: {e}")
        return

    # Perform object detection on the frame
    results = model(frame, conf=0.8)  # Confidence threshold

    # Loop through detected objects
    for result in results:
        boxes = result.boxes  # Bounding boxes
        for box in boxes:
            # Extract bounding box details
            x_center, y_center, width, height = box.xywh[0].tolist()  # Get bounding box center and size
            
            # Update the last detected x_center and y_center
            last_x_center = int(x_center)
            last_y_center = int(y_center)

            # Calculate corners of the bounding box
            x_min = int(x_center - width / 2)
            y_min = int(y_center - height / 2)
            x_max = int(x_center + width / 2)
            y_max = int(y_center + height / 2)
            
            # Draw the bounding box on the frame
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            # Draw the center point
            cv2.circle(frame, (last_x_center, last_y_center), 5, (0, 0, 255), -1)  # Red dot for center

            # Label the center point with coordinates
            label = f"({last_x_center}, {last_y_center})"
            cv2.putText(frame, label, (last_x_center + 5, last_y_center - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Display the frame
    cv2.imshow("YOLO Detection", frame)

    # Check if 5 seconds have passed
    elapsed_time = time.time() - start_time
    if elapsed_time > 5:
        rospy.signal_shutdown("5 seconds elapsed. Exiting...")

    # Close window on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        rospy.signal_shutdown("User pressed 'q'")

# Callback for depth image processing
def depth_callback(msg):
    global last_x_center, last_y_center, last_depth

    if last_x_center is None or last_y_center is None:
        return  # No object detected yet

    try:
        # Convert the ROS depth image message to a CV2 image
        depth_image = bridge.imgmsg_to_cv2(msg, "32FC1")  # Depth image in meters
    except CvBridgeError as e:
        rospy.logerr(f"Error converting depth image: {e}")
        return

    # Ensure coordinates are within image bounds
    height, width = depth_image.shape
    if 0 <= last_x_center < width and 0 <= last_y_center < height:
        # Retrieve depth value at the detected pixel
        last_depth = depth_image[last_y_center, last_x_center]
        rospy.loginfo(f"Depth at ({last_x_center}, {last_y_center}): {last_depth:.2f} meters")
    else:
        rospy.logwarn("Detected coordinates are out of depth image bounds.")

def main():
    rospy.init_node('yolo_depth_detector', anonymous=True)

    # Subscribe to the color image topic
    rospy.Subscriber("/camera/color/image_raw", Image, image_callback)

    # Subscribe to the depth image topic
    rospy.Subscriber("/camera/depth/image_rect_raw", Image, depth_callback)

    rospy.loginfo("YOLO depth detector node started. Press 'q' in the OpenCV window to exit.")

    # Keep the node running
    rospy.spin()

    # After node shutdown, print the last detected depth
    if last_x_center is not None and last_y_center is not None and last_depth is not None:
        print("\nLast detected coordinates and depth:")
        print(f"x_center: {last_x_center}, y_center: {last_y_center}, depth: {last_depth:.2f} meters")

    # Close all OpenCV windows
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
