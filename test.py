#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
import numpy as np
from cv_bridge import CvBridge

# 全域變數
bridge = CvBridge()

def img_callback(img_msg):

    # Step1: 讀取深度圖
    try:
        depth_image = bridge.imgmsg_to_cv2(img_msg, desired_encoding="passthrough")
    except Exception as e:
        rospy.logerr(f"Failed to convert image message to CV2: {e}")
        return

    height, width = depth_image.shape

    # Step2: 深度圖轉點雲
    # for uy in range(height):
    #     for ux in range(width):
    #         z = depth_image[uy, ux] / 1000.0  # 單位從 mm 轉換為 m
    z = depth_image[210,486]
    
    rospy.loginfo("width is %d",width)
    rospy.loginfo("height is %d",height)
    rospy.loginfo(z)


def main():
    # 初始化 ROS 節點
    rospy.init_node("depth_to_pointcloud_node", anonymous=True)

    # 訂閱話題
    rospy.Subscriber("/camera/depth/image_rect_raw", Image, img_callback)

    rospy.loginfo("Node is running...")
    rospy.spin()

if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass

