#!/usr/bin/env python3

import rospy
import time
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
from ultralytics import YOLO

# Load your YOLO model
model = YOLO(r"/home/stu/yolo/best.pt")  # Replace with your trained model file

# Initialize the CV Bridge
bridge = CvBridge()

# Global variables to store the last detected coordinates
last_x_center = None
last_y_center = None

# Record start time
start_time = None

# Callback function for processing images
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
            confidence = box.conf[0].item()  # Confidence score
            class_id = int(box.cls[0].item())  # Class ID
            
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
            cv2.putText(frame, f"Class {class_id} ({confidence:.2f})", 
                        (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            rospy.loginfo(f"Detected object: Class={class_id}, Confidence={confidence:.2f}")
            rospy.loginfo(f"Bounding Box: x_min={x_min}, y_min={y_min}, x_max={x_max}, y_max={y_max}")
            rospy.loginfo(f"Bounding Box: x_center={last_x_center}, y_center={last_y_center}, width={int(width)}, height={int(height)}")

    # Display the frame
    cv2.imshow("YOLO Detection", frame)

    # Check if 5 seconds have passed
    elapsed_time = time.time() - start_time
    if elapsed_time > 5:
        rospy.signal_shutdown("5 seconds elapsed. Exiting...")

    # Close window on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        rospy.signal_shutdown("User pressed 'q'")

def main():
    rospy.init_node('yolo_ros_detector', anonymous=True)

    # Subscribe to the image topic
    rospy.Subscriber("/camera/color/image_raw", Image, image_callback)

    rospy.loginfo("YOLO ROS detector node started. Press 'q' in the OpenCV window to exit.")

    # Keep the node running
    rospy.spin()

    # After node shutdown, print the last detected coordinates
    if last_x_center is not None and last_y_center is not None:
        print("\nLast detected coordinates:")
        print(f"x_center: {last_x_center}, y_center: {last_y_center}")

    # Close all OpenCV windows
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
