#!/usr/bin/env python
import rospy
from apriltag_ros.msg import AprilTagDetectionArray
from geometry_msgs.msg import PoseStamped

PUB = {}

def cb(msg):
    for det in msg.detections:
        if not det.id:
            continue
        tid = int(det.id[0])
        if tid not in PUB:
            continue
        out = PoseStamped()
        out.header = det.pose.header if det.pose.header.stamp.to_nsec() != 0 else msg.header
        out.pose = det.pose.pose.pose
        PUB[tid].publish(out)

def main():
    rospy.init_node("tagposes_from_detections", anonymous=True)
    ids = rospy.get_param("~ids", [0,1,2])
    prefix = rospy.get_param("~prefix", "/tags")
    for tid in ids:
        PUB[int(tid)] = rospy.Publisher("%s/tag_%d/pose" % (prefix, int(tid)), PoseStamped, queue_size=10)
    rospy.Subscriber("/apriltag_ros/tag_detections", AprilTagDetectionArray, cb, queue_size=10)
    rospy.spin()

if __name__ == "__main__":
    main()
