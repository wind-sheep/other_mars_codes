#!/usr/bin/env python3

# import rospy
# from sensor_msgs.msg import PointCloud2
# from sensor_msgs import point_cloud2
# import struct

# def callback(msg):
#     # Extract the width, height, and data from the PointCloud2 message
#     width = msg.width
#     height = msg.height
#     data = point_cloud2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
    
#     # Compute the index of the pixel (100, 200) in the 640x480 image
#     # Here, (x, y) = (100, 200) corresponds to row 200 and column 100 in the 640x480 image.
#     # 486 210
#     index = 210 * width + 486
    
#     # Convert the data to a list (or any other format that suits your needs)
#     points = list(data)
    
#     # Get the (x, y, z) coordinates of the point at (x=100, y=200)
#     point = points[index]  # This gives you the (x, y, z) coordinates of the point
#     rospy.loginfo("3D Coordinates at (100, 200): x = %f, y = %f, z = %f", point[0], point[1], point[2])

# rospy.init_node('pointcloud_listener')
# rospy.Subscriber('/camera/depth/color/points', PointCloud2, callback)
# rospy.spin()

import rospy
from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2

def callback(msg):
    # Extract the width, height, and data from the PointCloud2 message
    width = msg.width
    height = msg.height
    
    # Use read_points to access the point cloud data. This returns a generator.
    # We can convert the generator to a list to make it indexable.
    points = list(point_cloud2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True))
    
    # Ensure that the number of points in the cloud matches the image dimensions
    if len(points) < (200 * width + 100):  # Check if the index is within range
        rospy.logwarn("The index is out of range! There might be fewer points than expected.")
        return

    # Calculate the index for pixel (x=100, y=200)
    index = 200 * width + 100
    
    # Get the (x, y, z) coordinates of the point at (100, 200)
    point = points[index]  # This should be safe now
    rospy.loginfo("3D Coordinates at (100, 200): x = %f, y = %f, z = %f", point[0], point[1], point[2])

rospy.init_node('pointcloud_listener')
rospy.Subscriber('/camera/depth/color/points', PointCloud2, callback)
rospy.spin()
