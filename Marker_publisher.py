#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import Pose
from visualization_msgs.msg import Marker
import tf
from tf.transformations import *
import numpy as np
import tf2_ros
from geometry_msgs.msg import TransformStamped

def tf_from_pose_msg(pose_msg):
    q = [pose_msg.orientation.x, pose_msg.orientation.y, pose_msg.orientation.z, pose_msg.orientation.w]
    t = [pose_msg.position.x, pose_msg.position.y, pose_msg.position.z]
    T_mat = quaternion_matrix(q)
    T_mat[:3, 3] = t

    return T_mat

def tf_from_lookupTransform(rot, trans):
    q = rot
    t = trans
    T_mat = quaternion_matrix(q)
    T_mat[:3, 3] = t

    return T_mat

def talker():
    qr_pick_pub = rospy.Publisher('kimm_aruco_publisher/pose', Pose, queue_size=1)
    qr_visual_pub = rospy.Publisher('qr_marker_viz', Marker, queue_size=1)

    rospy.init_node('qr_talker', anonymous=True)
    rate = rospy.Rate(10) # 10hz

    while not rospy.is_shutdown():
        qr_pose= Pose()
        qr_pose.position.x =  -0.1
        qr_pose.position.y =  -0.1
        qr_pose.position.z =  -0.0
        qr_pose.orientation.x = 0.0
        qr_pose.orientation.y =-0.1305
        qr_pose.orientation.z = 0.0
        qr_pose.orientation.w = 0.9914     
        T_pos = tf_from_pose_msg(qr_pose)             

        # listener = tf.TransformListener()
        # try:
        #     listener.waitForTransform('world', 'camera_color_optical_frame', rospy.Time(0), rospy.Duration(4.0))
        #     (trans,rot) = listener.lookupTransform('world', 'camera_color_optical_frame', rospy.Time(0))
        # except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
        #     continue
        # T_transform = tf_from_lookupTransform(rot, trans)

        T_transform = quaternion_matrix([1, 0, 0, 0]) #x, y, z, w
        T_transform[:3, 3] = [0.522, 0.032, 0.584]     #home position

        # q_joint7 = [0.9239, -0.3827, 0.0, 0.0]    
        T_camera_to_joint7 = quaternion_matrix([0.0, 0.0, -0.3827, 0.9239] )

        T_new_pre = np.matmul(T_transform, T_pos)
        T_new = np.matmul(T_new_pre, T_camera_to_joint7)
        q_new = quaternion_from_matrix(T_new)

        qr_new= Pose()
        qr_new.position.x =  T_new[0,3]
        qr_new.position.y =  T_new[1,3]
        qr_new.position.z =  T_new[2,3]
        qr_new.orientation.x = q_new[0]
        qr_new.orientation.y = q_new[1]
        qr_new.orientation.z = q_new[2]
        qr_new.orientation.w = q_new[3]  

        marker = Marker()
        marker.header.frame_id = 'world'
        marker.id = 100
        marker.type = marker.CUBE
        marker.action = marker.ADD
        marker.pose.position.x = qr_new.position.x
        marker.pose.position.y = qr_new.position.y
        marker.pose.position.z = qr_new.position.z
        marker.pose.orientation.x = qr_new.orientation.x
        marker.pose.orientation.y = qr_new.orientation.y
        marker.pose.orientation.z = qr_new.orientation.z
        marker.pose.orientation.w = qr_new.orientation.w

        marker.scale.x = .1
        marker.scale.y = .1
        marker.scale.z = .01
        marker.color.a = 1
        marker.color.r = 0.0
        marker.color.g = 0.9
        marker.color.b = 0.2  

        qr_pick_pub.publish(qr_new)
        qr_visual_pub.publish(marker)


        br = tf2_ros.TransformBroadcaster()
        transform = TransformStamped()
        transform.header.stamp = rospy.Time.now()
        transform.header.frame_id = "world"
        transform.child_frame_id = "marker"
        transform.transform.translation.x = qr_new.position.x
        transform.transform.translation.y = qr_new.position.y
        transform.transform.translation.z = qr_new.position.z
        transform.transform.rotation.x = qr_new.orientation.x
        transform.transform.rotation.y = qr_new.orientation.y
        transform.transform.rotation.z = qr_new.orientation.z
        transform.transform.rotation.w = qr_new.orientation.w
        
        br.sendTransform(transform)
        
        rate.sleep()



if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass