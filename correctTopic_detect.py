#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
import numpy as np
import pcl
from pcl import PointCloud_PointXYZ
from pcl_conversions import to_msg
from cv_bridge import CvBridge

# 全域變數
pub_point_cloud2 = None
is_K_empty = True
K = np.zeros((3, 3))  # 相機內參矩陣
bridge = CvBridge()

def img_callback(img_msg):
    global K

    # Step1: 讀取深度圖
    try:
        depth_image = bridge.imgmsg_to_cv2(img_msg, desired_encoding="passthrough")
    except Exception as e:
        rospy.logerr(f"Failed to convert image message to CV2: {e}")
        return

    height, width = depth_image.shape

    # Step2: 深度圖轉點雲
    points = []
    for uy in range(height):
        for ux in range(width):
            z = depth_image[uy, ux] / 1000.0  # 單位從 mm 轉換為 m
            if z > 0:  # 跳過無效深度值
                x = z * (ux - K[0, 2]) / K[0, 0]
                y = z * (uy - K[1, 2]) / K[1, 1]
                points.append([x, y, z])

    # Step3: 發布點雲
    if points:
        cloud = pcl.PointCloud_PointXYZ()
        cloud.from_list(points)

        try:
            point_cloud2_msg = to_msg(cloud, frame_id="world")
            pub_point_cloud2.publish(point_cloud2_msg)
        except Exception as e:
            rospy.logerr(f"Failed to publish point cloud: {e}")


def camera_info_callback(camera_info_msg):
    global K, is_K_empty

    if is_K_empty:
        K = np.array(camera_info_msg.K).reshape(3, 3)  # 獲取 3x3 的內參矩陣
        is_K_empty = False


def main():
    global pub_point_cloud2

    # 初始化 ROS 節點
    rospy.init_node("depth_to_pointcloud_node", anonymous=True)

    # 訂閱話題
    rospy.Subscriber("/camera/depth/image_rect_raw", Image, img_callback)
    rospy.Subscriber("/camera/depth/camera_info", CameraInfo, camera_info_callback)

    # 設定發布器
    pub_point_cloud2 = rospy.Publisher("/d435i_point_cloud", PointCloud2, queue_size=10)

    rospy.loginfo("Node is running...")
    rospy.spin()

if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
