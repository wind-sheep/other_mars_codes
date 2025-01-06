#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
from ultralytics import YOLO
import time

# Load your YOLO model
model = YOLO(r"/home/stu/yolo/best.pt")  # Replace with your trained model file

# Initialize the CV Bridge
bridge = CvBridge()

# Variables for storing the last valid center and depth of the closest "Soda"
last_x_center = None
last_y_center = None
last_depth = None

min_depth = 10000 # a very large number(> the distance of the object detected)
depth_buffer_byMe = [min_depth]

# Start time to control the 5-second limit
start_time = None

# Global depth frame to sync with the RGB frame
depth_frame = None

# 全域變數
is_K_empty = True
K = np.zeros((3, 3))  # 相機內參矩陣
bridge = CvBridge()
ux=0
uy=0

# Depth callback to update the global depth frame
def depth_callback(msg):
    global depth_frame
    try:
        depth_frame = bridge.imgmsg_to_cv2(msg, "32FC1")
    except CvBridgeError as e:
        rospy.logerr(f"Error converting depth image: {e}")

# RGB callback for processing images
def rgb_callback(msg):
    global last_x_center, last_y_center, last_depth, center_buffer, depth_buffer, depth_buffer_byMe, min_depth

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
    results = model(frame, conf=0.9)  # Confidence threshold

    # Loop through detected objects
    for result in results:
        boxes = result.boxes  # Bounding boxes
        for box in boxes:
            # Extract bounding box details
            x_center, y_center, width, height = box.xywh[0].tolist()  # Get bounding box center and size
            confidence = box.conf[0].item()  # Confidence score
            class_id = int(box.cls[0].item())  # Class ID

            # Calculate corners of the bounding box
            x_min = int(x_center - width / 2)
            y_min = int(y_center - height / 2)
            x_max = int(x_center + width / 2)
            y_max = int(y_center + height / 2)

            # Check if the detected object is "Soda" (update class_id as per your model's class mapping)
            if class_id == 0:  # Assuming class_id 0 corresponds to "Soda"
                # Retrieve depth information at the center of the bounding box
                try:
                    depth_value = int(depth_frame[int(y_center), int(x_center)])

                    # Ignore invalid or infinite depth values
                    if not np.isfinite(depth_value):
                        continue

                    if depth_value==depth_buffer_byMe[0]:
                        depth_buffer_byMe.append(depth_value)
                        try:
                            if depth_buffer_byMe[2] is depth_value:
                                if depth_value<min_depth:
                                    min_depth=depth_value
                                    depth_buffer_byMe=[min_depth]
                                    last_x_center, last_y_center,last_depth=x_center, y_center,depth_value
                                    rospy.loginfo("update new min depth")
                                else:
                                    depth_buffer_byMe=[min_depth]
                                    rospy.loginfo("this is not min depth")
                        except IndexError:
                            rospy.loginfo("not having continuous 3 min depth yet")
                            
                    else:
                        depth_buffer_byMe=[depth_value]

                    # Draw bounding box and annotate
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                    cv2.putText(frame, f"Soda: {depth_value:.2f}m", 
                                (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    # Show the center of the bounding box
                    cv2.circle(frame, (int(x_center), int(y_center)), 5, (0, 0, 255), -1)
                    cv2.putText(frame, f"Center: ({int(x_center)}, {int(y_center)})", 
                                (int(x_center) + 10, int(y_center) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    
                    # Print the coordinates and depth of the detected "Soda"
                    rospy.loginfo(f"Detected Soda at x={int(x_center)}, y={int(y_center)}, Depth={depth_value:.2f}m")

                except IndexError:
                    rospy.logwarn("Depth value out of range.")

    # Display the frame with annotations
    cv2.imshow("YOLO Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        rospy.signal_shutdown("User pressed 'q'")
        
def camera_info_callback(camera_info_msg): #get K
    global K, is_K_empty
    
    rospy.loginfo("test1")

    if is_K_empty:
        K = np.array(camera_info_msg.K).reshape(3, 3)  # 獲取 3x3 的內參矩陣
        is_K_empty = False
        
# def img_callback(img_msg):
#     global K,ux,uy, last_x_center, last_y_center
    
#     rospy.loginfo("test2")

#     # Step1: 讀取深度圖 get depth of each pixel
#     try:
#         depth_image = bridge.imgmsg_to_cv2(img_msg, desired_encoding="passthrough")
#     except Exception as e:
#         rospy.logerr(f"Failed to convert image message to CV2: {e}")
#         return
                
#     # ux=486
#     # uy=210
#     ux=int(last_x_center)
#     uy=int(last_y_center)
#     z = depth_image[uy, ux] / 1000.0 # change mm to m
#     x = z * (ux - K[0, 2]) / K[0, 0]
#     y = z * (uy - K[1, 2]) / K[1, 1]
#     rospy.loginfo("coordinate:(%f,%f,%f)"%(x,y,z))

def img_callback():
    global K,ux,uy, last_x_center, last_y_center, last_depth
    
    print("test2")
                
    # ux=486
    # uy=210
    ux=int(last_x_center)
    uy=int(last_y_center)
    z = last_depth / 1000.0 # change mm to m
    x = z * (ux - K[0, 2]) / K[0, 0]
    y = z * (uy - K[1, 2]) / K[1, 1]
    print("coordinate:(%f,%f,%f)"%(x,y,z))

# Main function
def main():
    global start_time, last_x_center, last_y_center, last_depth

    rospy.init_node('yolo_depth_detector', anonymous=True)

    rospy.sleep(1)  # Wait for 2 seconds to ensure depth frames are available
    
    # Subscribe to the RGB and depth topics
    rospy.Subscriber("/camera/color/image_raw", Image, rgb_callback)
    rospy.Subscriber("/camera/depth/image_rect_raw", Image, depth_callback)
    rospy.Subscriber("/camera/depth/camera_info", CameraInfo, camera_info_callback) # get K

    rospy.loginfo("YOLO Depth Detector node started. Press 'q' to exit.")

    # Initialize the timer
    start_time = time.time()

    # Keep the node running for 5 seconds
    rate = rospy.Rate(10) # 10Hz
    while not rospy.is_shutdown():
        elapsed_time = time.time() - start_time
        if elapsed_time > 5: #second
            break
        rate.sleep()

    # Print the last stored center coordinate and depth
    if last_x_center is not None and last_y_center is not None and last_depth is not None:
        rospy.loginfo(f"\nLast closest Soda: x_center={int(last_x_center)}, y_center={int(last_y_center)}, depth={last_depth:.2f}m\n")
    
    else:
        rospy.loginfo("\nNo valid Soda detected.\n")

    # Close all OpenCV windows
    #cv2.destroyAllWindows()
    
    rospy.loginfo("Subscribing to camera info and depth topics for 3D coordinate extraction.")
    # rospy.Subscriber("/camera/depth/camera_info", CameraInfo, camera_info_callback) # get K
    #rospy.Subscriber("/camera/depth/image_rect_raw", Image, img_callback)
    img_callback()
    rospy.loginfo("test3")

if __name__ == "__main__":
    main()
