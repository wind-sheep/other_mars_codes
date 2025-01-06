#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image, CameraInfo
import numpy as np
from cv_bridge import CvBridge
from geometry_msgs.msg import Point

# 全域變數
is_K_empty = True
K = np.zeros((3, 3))  # 相機內參矩陣
bridge = CvBridge()
ux=0
uy=0
#is_first_script_done = False

# def execution_complete_callback(msg):
#     global is_first_script_done
#     is_first_script_done = True
#     rospy.loginfo("Received execution complete signal, starting the second script.")

# def coordinates_callback(msg):
#     global ux,uy
#     rospy.loginfo(f"Received coordinates: x={msg.x}, y={msg.y}")
#     ux=msg.x
#     uy=msg.y

def camera_info_callback(camera_info_msg): #get K
    global K, is_K_empty

    if is_K_empty:
        K = np.array(camera_info_msg.K).reshape(3, 3)  # 獲取 3x3 的內參矩陣
        is_K_empty = False
    
    # # print K
    # rospy.loginfo("K is :")
    # for i in range(3):
    #     for j in range(3):
    #         rospy.loginfo(K[i][j])
    
def img_callback(img_msg):
    global K,ux,uy #, is_first_script_done
    
    # # Wait until the first script is done before processing images
    # if not is_first_script_done:
    #     rospy.loginfo("Waiting for first script to finish...")
    #     return

    # Step1: 讀取深度圖 get depth of each pixel
    try:
        depth_image = bridge.imgmsg_to_cv2(img_msg, desired_encoding="passthrough")
    except Exception as e:
        rospy.logerr(f"Failed to convert image message to CV2: {e}")
        return

    #Step2: 深度圖轉點雲 turn pixel coordinate into camera coordinate
    # points = []
    # for uy in range(height):
    #     for ux in range(width):
    #         z = depth_image[uy, ux] / 1000.0  # 單位從 mm 轉換為 m
    #         if z > 0:  # 跳過無效深度值
    #             x = z * (ux - K[0, 2]) / K[0, 0]
    #             y = z * (uy - K[1, 2]) / K[1, 1]
    #             #points.append([x, y, z])
    #             rospy.loginfo("coordinate:(%d,%d,%d)",x,y,z)
                
    ux=486
    uy=210
    z = depth_image[uy, ux] / 1000.0 # change mm to m
    x = z * (ux - K[0, 2]) / K[0, 0]
    y = z * (uy - K[1, 2]) / K[1, 1]
    rospy.loginfo("coordinate:(%f,%f,%f)"%(x,y,z))

def main():
    # 初始化 ROS 節點
    rospy.init_node("depth_to_pointcloud_node", anonymous=True)
    
    # rospy.Subscriber('soda_coordinates', Point, coordinates_callback)
    
    #rospy.Subscriber("/execution_complete", Point, execution_complete_callback)

    # 訂閱話題
    rospy.Subscriber("/camera/depth/camera_info", CameraInfo, camera_info_callback) # get K
    rospy.Subscriber("/camera/depth/image_rect_raw", Image, img_callback)

    rospy.loginfo("Node is running...")
    rospy.spin()

if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
