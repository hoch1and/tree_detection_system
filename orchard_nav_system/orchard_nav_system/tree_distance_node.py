import math

import numpy as np
import cv2

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge

from message_filters import Subscriber, ApproximateTimeSynchronizer


class TreeDistanceNode(Node):
    """
    Tree-oriented postprocessing node.

    It combines:
        tree_mask + depth_image

    and estimates 3D tree features:
        - distance to tree
        - mean X position in meters
        - mean Z depth
        - angle to tree center
        - left/right tree distribution
        - tree mask area

    Output /perception/tree_info:
        [
            distance_m,
            mean_x_m,
            mean_z_m,
            angle_rad,
            left_ratio,
            right_ratio,
            tree_area_ratio,
            valid
        ]
    """

    def __init__(self):
        super().__init__("tree_distance_node")

        self.declare_parameter("depth_topic", "/camera/depth/image_raw")
        self.declare_parameter("tree_mask_topic", "/segmentation/tree_mask")
        self.declare_parameter("tree_info_topic", "/perception/tree_info")

        self.declare_parameter("max_depth_m", 20.0)
        self.declare_parameter("min_depth_m", 0.1)
        self.declare_parameter("depth_unit_scale", 0.001)
        self.declare_parameter("distance_percentile", 20.0)

        # Camera intrinsics.
        # If fx/fy/cx/cy <= 0, they will be estimated from image size and FOV.
        self.declare_parameter("fx", 0.0)
        self.declare_parameter("fy", 0.0)
        self.declare_parameter("cx", 0.0)
        self.declare_parameter("cy", 0.0)
        self.declare_parameter("horizontal_fov_deg", 69.0)

        # To reduce CPU load, we may sample mask pixels.
        self.declare_parameter("max_points", 8000)

        # Morphological filtering for unstable masks.
        self.declare_parameter("min_mask_area_ratio", 0.001)
        self.declare_parameter("morph_kernel_size", 5)

        self.depth_topic = self.get_parameter("depth_topic").value
        self.tree_mask_topic = self.get_parameter("tree_mask_topic").value
        self.tree_info_topic = self.get_parameter("tree_info_topic").value

        self.max_depth_m = float(self.get_parameter("max_depth_m").value)
        self.min_depth_m = float(self.get_parameter("min_depth_m").value)
        self.depth_unit_scale = float(self.get_parameter("depth_unit_scale").value)
        self.distance_percentile = float(self.get_parameter("distance_percentile").value)

        self.fx_param = float(self.get_parameter("fx").value)
        self.fy_param = float(self.get_parameter("fy").value)
        self.cx_param = float(self.get_parameter("cx").value)
        self.cy_param = float(self.get_parameter("cy").value)
        self.horizontal_fov_deg = float(self.get_parameter("horizontal_fov_deg").value)

        self.max_points = int(self.get_parameter("max_points").value)
        self.min_mask_area_ratio = float(self.get_parameter("min_mask_area_ratio").value)
        self.morph_kernel_size = int(self.get_parameter("morph_kernel_size").value)

        self.bridge = CvBridge()

        self.tree_info_pub = self.create_publisher(
            Float32MultiArray,
            self.tree_info_topic,
            10
        )

        self.depth_sub = Subscriber(self, Image, self.depth_topic)
        self.tree_mask_sub = Subscriber(self, Image, self.tree_mask_topic)

        self.sync = ApproximateTimeSynchronizer(
            [self.depth_sub, self.tree_mask_sub],
            queue_size=10,
            slop=0.2
        )
        self.sync.registerCallback(self.synced_callback)

        self.get_logger().info("Tree 3D postprocessing node started.")

    def convert_depth_to_meters(self, depth_msg: Image) -> np.ndarray:
        depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
        depth = depth.astype(np.float32)

        if depth_msg.encoding == "16UC1":
            depth *= self.depth_unit_scale
        elif depth_msg.encoding == "32FC1":
            pass
        else:
            # RealSense z16 sometimes may appear with a non-standard encoding.
            finite = depth[np.isfinite(depth)]
            if finite.size > 0 and np.nanmedian(finite) > 100.0:
                depth *= self.depth_unit_scale

        return depth

    @staticmethod
    def resize_mask_if_needed(mask: np.ndarray, target_shape) -> np.ndarray:
        target_h, target_w = target_shape[:2]
        if mask.shape[:2] != (target_h, target_w):
            return cv2.resize(mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
        return mask

    def preprocess_mask(self, mask: np.ndarray) -> np.ndarray:
        binary = (mask > 0).astype(np.uint8)

        if self.morph_kernel_size > 1:
            k = self.morph_kernel_size
            kernel = np.ones((k, k), dtype=np.uint8)
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        return binary.astype(bool)

    def get_intrinsics(self, width: int, height: int):
        cx = self.cx_param if self.cx_param > 0 else width / 2.0
        cy = self.cy_param if self.cy_param > 0 else height / 2.0

        if self.fx_param > 0 and self.fy_param > 0:
            fx = self.fx_param
            fy = self.fy_param
        else:
            fov_rad = math.radians(self.horizontal_fov_deg)
            fx = width / (2.0 * math.tan(fov_rad / 2.0))
            fy = fx

        return fx, fy, cx, cy

    def mask_depth_to_points(self, tree_mask: np.ndarray, depth: np.ndarray):
        h, w = tree_mask.shape[:2]

        ys, xs = np.where(tree_mask)
        if xs.size == 0:
            return None

        z = depth[ys, xs]
        valid_depth = np.isfinite(z) & (z >= self.min_depth_m) & (z <= self.max_depth_m)

        xs = xs[valid_depth]
        ys = ys[valid_depth]
        z = z[valid_depth]

        if z.size == 0:
            return None

        if self.max_points > 0 and z.size > self.max_points:
            idx = np.linspace(0, z.size - 1, self.max_points).astype(np.int32)
            xs = xs[idx]
            ys = ys[idx]
            z = z[idx]

        fx, fy, cx, cy = self.get_intrinsics(w, h)

        x = (xs.astype(np.float32) - cx) * z / fx
        y = (ys.astype(np.float32) - cy) * z / fy

        points = np.stack([x, y, z], axis=1)
        return points, xs, ys

    def compute_tree_features(self, tree_mask: np.ndarray, depth: np.ndarray):
        h, w = tree_mask.shape[:2]
        frame_area = float(h * w)

        tree_area = int(np.count_nonzero(tree_mask))
        tree_area_ratio = tree_area / frame_area

        if tree_area_ratio < self.min_mask_area_ratio:
            return [float("inf"), 0.0, 0.0, 0.0, 0.0, 0.0, tree_area_ratio, 0.0]

        left_area = int(np.count_nonzero(tree_mask[:, :w // 2]))
        right_area = int(np.count_nonzero(tree_mask[:, w // 2:]))

        if tree_area > 0:
            left_ratio = left_area / float(tree_area)
            right_ratio = right_area / float(tree_area)
        else:
            left_ratio = 0.0
            right_ratio = 0.0

        point_result = self.mask_depth_to_points(tree_mask, depth)
        if point_result is None:
            return [float("inf"), 0.0, 0.0, 0.0, left_ratio, right_ratio, tree_area_ratio, 0.0]

        points, xs, ys = point_result

        x_values = points[:, 0]
        z_values = points[:, 2]

        # Distance: use percentile, not min, because depth is noisy.
        distance_m = float(np.percentile(z_values, self.distance_percentile))

        # Robust center estimates.
        mean_x_m = float(np.median(x_values))
        mean_z_m = float(np.median(z_values))

        # Angle from camera forward axis to tree center.
        # Negative angle = tree is left, positive = tree is right.
        angle_rad = float(math.atan2(mean_x_m, mean_z_m))

        return [
            distance_m,
            mean_x_m,
            mean_z_m,
            angle_rad,
            float(left_ratio),
            float(right_ratio),
            float(tree_area_ratio),
            1.0
        ]

    def synced_callback(self, depth_msg: Image, tree_mask_msg: Image):
        try:
            depth = self.convert_depth_to_meters(depth_msg)
            tree_mask = self.bridge.imgmsg_to_cv2(tree_mask_msg, desired_encoding="mono8")
            tree_mask = self.resize_mask_if_needed(tree_mask, depth.shape)
            tree_mask = self.preprocess_mask(tree_mask)

        except Exception as exc:
            self.get_logger().error(f"Failed to process synchronized messages: {exc}")
            return

        features = self.compute_tree_features(tree_mask, depth)

        msg = Float32MultiArray()
        msg.data = [float(v) for v in features]
        self.tree_info_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = TreeDistanceNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()