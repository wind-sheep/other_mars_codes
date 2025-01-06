#!/usr/bin/env python3

import rospy
import time
from collections import deque
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
from ultralytics import YOLO

# Load your YOLO model
model = YOLO(r"/home/stu/yolo/best.pt")  # Replace with your trained model file

# Initialize the CV Bridge
bridge = CvBridge()

# Global variables
SODA_CLASS_ID = 0  # Replace with the correct class ID for "Soda"
CONFIDENCE_THRESHOLD = 0.9  # Confidence threshold for reliable detections
BUFFER_SIZE = 5  # Number of detections to keep in the stabilization buffer
stabilization_buffer = deque(maxlen=BUFFER_SIZE)
depth_image = None  # To store the depth image
last_stored_coordinate = None
last_stored_depth = None

# Helper function to check if coordinates are stable
def are_coordinates_stable(buffer):
    if len(buffer) < BUFFER_SIZE:
        return False  # Buffer is not yet full

    mean_x = sum(coord[0] for coord in buffer) / BUFFER_SIZE
    mean_y = sum(coord[1] for coord in buffer) / BUFFER_SIZE

    for x, y in buffer:
        if abs(x - mean_x) > 10 or abs(y - mean_y) > 10:  # Adjust tolerance
            return False
    return True

# Callback for depth image
def depth_callback(msg):
    global depth_image
    try:
        depth_image = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
    except CvBridgeError as e:
        rospy.logerr(f"Error converting depth image: {e}")

# Callback for color image processing
def image_callback(msg):
    global stabilization_buffer, depth_image, last_stored_coordinate, last_stored_depth

    try:
        frame = bridge.imgmsg_to_cv2(msg, "bgr8")
    except CvBridgeError as e:
        rospy.logerr(f"Error converting image: {e}")
        return

    results = model(frame, conf=CONFIDENCE_THRESHOLD)

    detected_soda = None
    for result in results:
        boxes = result.boxes
        for box in boxes:
            class_id = int(box.cls[0].item())
            confidence = box.conf[0].item()

            if class_id == SODA_CLASS_ID and confidence >= CONFIDENCE_THRESHOLD:
                x_center, y_center, width, height = box.xywh[0].tolist()
                detected_soda = (int(x_center), int(y_center))

                x_min = int(x_center - width / 2)
                y_min = int(y_center - height / 2)
                x_max = int(x_center + width / 2)
                y_max = int(y_center + height / 2)

                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                cv2.circle(frame, detected_soda, 5, (0, 0, 255), -1)
                cv2.putText(frame, f"Soda ({x_center:.0f}, {y_center:.0f})", 
                            (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                if depth_image is not None:
                    depth_value = depth_image[int(y_center), int(x_center)]
                    cv2.putText(frame, f"Depth: {depth_value:.2f}m", 
                                (x_min, y_max + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    if detected_soda:
        stabilization_buffer.append(detected_soda)

    if are_coordinates_stable(stabilization_buffer):
        stable_x, stable_y = map(int, [sum(coord[i] for coord in stabilization_buffer) / BUFFER_SIZE for i in range(2)])
        if depth_image is not None:
            stable_depth = depth_image[stable_y, stable_x]
            last_stored_coordinate = (stable_x, stable_y)
            last_stored_depth = stable_depth
            rospy.loginfo(f"Stable Soda Coordinates: x_center={stable_x}, y_center={stable_y}, depth={stable_depth:.2f}m")

    cv2.imshow("YOLO Detection with Depth", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        rospy.signal_shutdown("User pressed 'q'")

def main():
    global last_stored_coordinate, last_stored_depth

    rospy.init_node('yolo_depth_detector', anonymous=True)

    rospy.Subscriber("/camera/color/image_raw", Image, image_callback)
    rospy.Subscriber("/camera/depth/image_rect_raw", Image, depth_callback)

    rospy.loginfo("YOLO with Depth detector node started. Running for 5 seconds.")

    start_time = time.time()
    rate = rospy.Rate(10)  # 10 Hz loop rate

    while not rospy.is_shutdown():
        elapsed_time = time.time() - start_time
        if elapsed_time > 5.0:
            rospy.loginfo("5 seconds elapsed. Shutting down the node.")
            break
        rate.sleep()

    cv2.destroyAllWindows()

    # Print last stored coordinate and depth
    if last_stored_coordinate and last_stored_depth is not None:
        rospy.loginfo(f"\nLast Stored Coordinate: {last_stored_coordinate}")
        rospy.loginfo(f"Last Stored Depth: {last_stored_depth:.2f}m\n")
    else:
        rospy.loginfo("\nNo valid detection within the 5 seconds.\n")

if __name__ == "__main__":
    main()
