import threading
import time
from typing import Optional

import numpy as np

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image

try:
    import pyrealsense2 as rs
except ImportError:
    rs = None


class RealSenseBagPublisher(Node):
    """
    Reads Intel RealSense .bag file directly through pyrealsense2
    and publishes RGB + aligned depth frames as ROS 2 Image messages.

    Output topics:
        /camera/rgb/image_raw
        /camera/depth/image_raw
    """

    def __init__(self):
        super().__init__("realsense_bag_publisher")

        self.declare_parameter("bag_path", "/home/h/test_video.bag")
        self.declare_parameter("rgb_topic_out", "/camera/rgb/image_raw")
        self.declare_parameter("depth_topic_out", "/camera/depth/image_raw")
        self.declare_parameter("frame_id", "camera_link")
        self.declare_parameter("loop", True)
        self.declare_parameter("publish_rate", 30.0)
        self.declare_parameter("align_depth_to_color", True)

        self.bag_path = self.get_parameter("bag_path").value
        self.rgb_topic_out = self.get_parameter("rgb_topic_out").value
        self.depth_topic_out = self.get_parameter("depth_topic_out").value
        self.frame_id = self.get_parameter("frame_id").value
        self.loop = bool(self.get_parameter("loop").value)
        self.publish_rate = float(self.get_parameter("publish_rate").value)
        self.align_depth_to_color = bool(self.get_parameter("align_depth_to_color").value)

        if self.publish_rate <= 0:
            self.publish_rate = 30.0

        self.rgb_pub = self.create_publisher(Image, self.rgb_topic_out, 10)
        self.depth_pub = self.create_publisher(Image, self.depth_topic_out, 10)

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        if rs is None:
            self.get_logger().error(
                "pyrealsense2 is not installed. Run: pip install pyrealsense2"
            )
            return

        self.get_logger().info(f"RealSense bag path: {self.bag_path}")
        self.get_logger().info(f"RGB output topic: {self.rgb_topic_out}")
        self.get_logger().info(f"Depth output topic: {self.depth_topic_out}")

        self._thread = threading.Thread(target=self._play_loop, daemon=True)
        self._thread.start()

    def destroy_node(self):
        self._stop_event.set()

        if self._thread is not None:
            self._thread.join(timeout=2.0)

        super().destroy_node()

    def _make_image_msg(self, image: np.ndarray, encoding: str) -> Image:
        msg = Image()

        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id

        msg.height = int(image.shape[0])
        msg.width = int(image.shape[1])
        msg.encoding = encoding
        msg.is_bigendian = 0
        msg.step = int(image.strides[0])
        msg.data = image.tobytes()

        return msg

    def _play_once(self):
        pipeline = rs.pipeline()
        config = rs.config()

        rs.config.enable_device_from_file(
            config,
            self.bag_path,
            repeat_playback=False
        )

        profile = pipeline.start(config)

        playback = profile.get_device().as_playback()
        playback.set_real_time(False)

        align = None
        if self.align_depth_to_color:
            align = rs.align(rs.stream.color)

        sleep_dt = 1.0 / self.publish_rate

        self.get_logger().info("Started RealSense bag playback.")

        try:
            while rclpy.ok() and not self._stop_event.is_set():
                try:
                    frames = pipeline.wait_for_frames(1000)
                except RuntimeError:
                    self.get_logger().info("End of RealSense bag.")
                    break

                if align is not None:
                    frames = align.process(frames)

                color_frame = frames.get_color_frame()
                depth_frame = frames.get_depth_frame()

                if not color_frame or not depth_frame:
                    continue

                color_image = np.asanyarray(color_frame.get_data())
                depth_image = np.asanyarray(depth_frame.get_data())

                color_format = color_frame.profile.format()

                if color_format == rs.format.rgb8:
                    rgb_encoding = "rgb8"
                elif color_format == rs.format.bgr8:
                    rgb_encoding = "bgr8"
                else:
                    # Fallback. YOLO node will request bgr8 via cv_bridge.
                    rgb_encoding = "rgb8"

                # RealSense depth is usually uint16 Z16 in millimeters/depth units.
                if depth_image.dtype == np.uint16:
                    depth_encoding = "16UC1"
                elif depth_image.dtype == np.float32:
                    depth_encoding = "32FC1"
                else:
                    depth_image = depth_image.astype(np.uint16)
                    depth_encoding = "16UC1"

                rgb_msg = self._make_image_msg(color_image, rgb_encoding)
                depth_msg = self._make_image_msg(depth_image, depth_encoding)

                self.rgb_pub.publish(rgb_msg)
                self.depth_pub.publish(depth_msg)

                time.sleep(sleep_dt)

        finally:
            pipeline.stop()
            self.get_logger().info("Stopped RealSense bag playback.")

    def _play_loop(self):
        while rclpy.ok() and not self._stop_event.is_set():
            try:
                self._play_once()
            except Exception as exc:
                self.get_logger().error(f"RealSense bag playback error: {exc}")
                time.sleep(1.0)

            if not self.loop:
                break

            self.get_logger().info("Restarting RealSense bag playback...")


def main(args=None):
    rclpy.init(args=args)

    node = RealSenseBagPublisher()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
