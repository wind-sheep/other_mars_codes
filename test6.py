#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
from ultralytics import YOLO
import time

# Load the YOLO model
model = YOLO(r"/home/stu/yolo/best3.pt")  # Replace with your trained model file

# Initialize the CV Bridge
bridge = CvBridge()

# Variables for storing the last valid center and depth of the closest "Soda"
last_x_center = None
last_y_center = None
last_depth = None

total_x_center=0
total_y_center=0
total_depth_value=0
counter=0

# Start time to control the 5-second limit
start_time = None

# Global depth frame to sync with the RGB frame
depth_frame = None

# Camera intrinsic matrix
is_K_empty = True
K = np.zeros((3, 3))  # Camera intrinsic matrix

# Depth callback to update the global depth frame
def depth_callback(msg):
    global depth_frame
    try:
        depth_frame = bridge.imgmsg_to_cv2(msg, "32FC1")
    except CvBridgeError as e:
        rospy.logerr(f"Error converting depth image: {e}")

# RGB callback for processing images
def rgb_callback(msg):
    global total_x_center, total_y_center, total_depth_value, counter

    try:
        # Convert the ROS Image message to a CV2 image
        frame = bridge.imgmsg_to_cv2(msg, "bgr8")
    except CvBridgeError as e:
        rospy.logerr(f"Error converting RGB image: {e}")
        return

    if depth_frame is None:
        rospy.logwarn("Depth frame is not available yet.")
        return

    # Perform object detection on the frame
    results = model(frame, conf=0.8)  # Confidence threshold

    # Loop through detected objects
    for result in results:
        for box in result.boxes:
            # Extract bounding box details
            x_center, y_center, width, height = box.xywh[0].tolist()
            confidence = box.conf[0].item()
            class_id = int(box.cls[0].item())

            # Calculate bounding box corners
            x_min = int(x_center - width / 2)
            y_min = int(y_center - height / 2)
            x_max = int(x_center + width / 2)
            y_max = int(y_center + height / 2)

            # Check if the detected object is "Soda"
            if class_id == 2:  # Assuming class_id 0 corresponds to "Soda"
                try:
                    # Retrieve depth information at the center of the bounding box
                    depth_value = depth_frame[int(y_center), int(x_center)]

                    if not np.isfinite(depth_value):
                        continue
                    
                    if int(depth_value)!=0:
                        total_x_center+=x_center
                        total_y_center+=y_center
                        total_depth_value+=depth_value
                        counter+=1
                    else:
                        rospy.logwarn("There is no depth data.")

                    # Draw bounding box and annotate
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                    cv2.putText(frame, f"Soda: {depth_value:.2f}m", 
                                (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    # Mark the center of the bounding box
                    cv2.circle(frame, (int(x_center), int(y_center)), 5, (0, 0, 255), -1)
                    cv2.putText(frame, f"Center: ({int(x_center)}, {int(y_center)})", 
                                (int(x_center) + 10, int(y_center) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    
                    # Print the coordinates and depth of the detected "Soda"
                    rospy.loginfo(f"Detected Soda at x={int(x_center)}, y={int(y_center)}, Depth={depth_value}mm")

                except IndexError:
                    rospy.logwarn("Depth value out of range.")

    # Display the frame with annotations
    cv2.imshow("YOLO Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        rospy.signal_shutdown("User pressed 'q'")

def camera_info_callback(camera_info_msg):
    global K, is_K_empty

    if is_K_empty:
        K = np.array(camera_info_msg.K).reshape(3, 3)
        is_K_empty = False
        rospy.loginfo("Camera intrinsic matrix received.")

def compute_coordinates():
    global K, total_x_center, total_y_center, total_depth_value, counter
    
    final_x_center= 0
    final_y_center= 0
    final_depth_value= 0

    if counter!=0:
        final_x_center=total_x_center/counter
        final_y_center= total_y_center/counter
        final_depth_value=total_depth_value/counter
        #final_real_depth_value= 0.2059*final_depth_value-353.19 # 41cm to 44cm
        final_real_depth_value=final_depth_value
        rospy.loginfo("")
        rospy.loginfo("pixel coordinate and depth:(%d,%d),%d"%(int(final_x_center), int(final_y_center), int(final_real_depth_value)))
        
        camera_z = final_real_depth_value / 1000  # Convert mm to m
        camera_x = camera_z * (final_x_center - K[0, 2]) / K[0, 0]
        camera_y = camera_z * (final_y_center - K[1, 2]) / K[1, 1]
        rospy.loginfo("")
        rospy.loginfo("camera coordinate:(%f,%f,%f)"%(camera_x, camera_y, camera_z))
        rospy.loginfo("")
        
        arm_x=camera_z
        arm_y=-camera_x
        arm_z=-camera_y
        
        ############################################### the final camera cordinate are (x,y,z), publish these using ROSbridge  
        
    else:
        rospy.logwarn("No valid detection to compute 3D coordinates.(No depth data)")

# Main function
def main():
    global start_time

    rospy.init_node('yolo_depth_detector', anonymous=True)

    # Subscribe to the RGB and depth topics
    rospy.Subscriber("/camera/color/image_raw", Image, rgb_callback)
    #rospy.Subscriber("/camera/depth/image_rect_raw", Image, depth_callback)
    rospy.Subscriber("/camera/aligned_depth_to_color/image_raw", Image, depth_callback)
    rospy.Subscriber("/camera/depth/camera_info", CameraInfo, camera_info_callback)

    rospy.loginfo("YOLO Depth Detector node started. Press 'q' to exit.")

    # Start timer
    start_time = time.time()

    # Keep the node running for 5 seconds
    rate = rospy.Rate(10)  # 10Hz
    while not rospy.is_shutdown():
        elapsed_time = time.time() - start_time
        if elapsed_time > 5:
            break
        rate.sleep()

    # Compute 3D coordinates
    compute_coordinates()

    # Clean up
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
