# Pose, velocity and cameras read straight from the ROS topics, not through HAL.
# HAL's drone node handles only a few callbacks per second and the topics send
# far more (pose 83 Hz, cameras 30 Hz), so its queues stay full. Measured
# flying at 2 m/s: HAL position changes 2 times per second and is 0.3 s old on
# average (0.55 s worst), and the images are 0.35 s old. That is around 1 m of
# error at cruise speed. Here every queue has depth 1, so we get the newest.

import math
import threading
import cv_bridge
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from geometry_msgs.msg import PoseStamped, TwistStamped
from sensor_msgs.msg import Image

NEWEST = QoSProfile(depth=1, reliability=QoSReliabilityPolicy.BEST_EFFORT)


def stamp_s(msg):
    return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9


class Sensors(Node):
    def __init__(self):
        if not rclpy.ok():
            rclpy.init()
        super().__init__("drone_fresh_sensors")
        self.pose = None    # (x, y, z, yaw), m and rad
        self.pose_stamp = 0.0   # s, sim time
        self.transform = None   # (position, rotation matrix), body -> world, to place what the camera sees
        self.velocity = (0.0, 0.0, 0.0)     # m/s
        self.images = {}    # camera name -> newest message
        self.bridge = cv_bridge.CvBridge()
        self.create_subscription(PoseStamped, "/drone/self_localization/pose", self.on_pose, NEWEST)
        self.create_subscription(TwistStamped, "/drone/self_localization/twist", self.on_twist, NEWEST)
        for name in ("frontal", "ventral"):
            self.create_subscription(Image, "/drone/%s_cam/image_raw" % name,
                                     lambda msg, name=name: self.images.__setitem__(name, msg), NEWEST)
        ex = SingleThreadedExecutor()
        ex.add_node(self)
        threading.Thread(target=ex.spin, daemon=True).start()

    def on_pose(self, msg):
        p, q = msg.pose.position, msg.pose.orientation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        self.pose = (p.x, p.y, p.z, yaw)
        rotation = np.array([
            [1 - 2 * (q.y * q.y + q.z * q.z), 2 * (q.x * q.y - q.w * q.z), 2 * (q.x * q.z + q.w * q.y)],
            [2 * (q.x * q.y + q.w * q.z), 1 - 2 * (q.x * q.x + q.z * q.z), 2 * (q.y * q.z - q.w * q.x)],
            [2 * (q.x * q.z - q.w * q.y), 2 * (q.y * q.z + q.w * q.x), 1 - 2 * (q.x * q.x + q.y * q.y)]])
        self.transform = (np.array([p.x, p.y, p.z]), rotation)
        self.pose_stamp = stamp_s(msg)

    def on_twist(self, msg):
        v = msg.twist.linear
        self.velocity = (v.x, v.y, v.z)

    # Newest frame of "frontal" or "ventral" as (bgr image, stamp), or None
    def image(self, name):
        msg = self.images.get(name)
        if msg is None:
            return None
        return self.bridge.imgmsg_to_cv2(msg, "bgr8"), stamp_s(msg)
