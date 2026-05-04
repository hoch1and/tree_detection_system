import cv2

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class VideoRecorderNode(Node):
    def __init__(self):
        super().__init__("video_recorder_node")

        self.declare_parameter("image_topic", "/visualization/debug_overlay")
        self.declare_parameter("output_path", "/home/h/orchard_nav_demo.mp4")
        self.declare_parameter("fps", 25.0)

        self.image_topic = self.get_parameter("image_topic").value
        self.output_path = self.get_parameter("output_path").value
        self.fps = float(self.get_parameter("fps").value)

        self.bridge = CvBridge()

        self.latest_frame = None
        self.writer = None
        self.frame_size = None
        self.frames_written = 0

        self.sub = self.create_subscription(
            Image,
            self.image_topic,
            self.image_callback,
            10
        )

        self.timer = self.create_timer(1.0 / self.fps, self.write_frame)

        self.get_logger().info(f"Recording topic: {self.image_topic}")
        self.get_logger().info(f"Output path: {self.output_path}")
        self.get_logger().info(f"Output FPS: {self.fps}")

    def image_callback(self, msg: Image):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            self.latest_frame = frame
        except Exception as exc:
            self.get_logger().warning(f"Failed to convert image: {exc}")

    def init_writer(self, frame):
        h, w = frame.shape[:2]
        self.frame_size = (w, h)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        self.writer = cv2.VideoWriter(
            self.output_path,
            fourcc,
            self.fps,
            self.frame_size
        )

        if not self.writer.isOpened():
            self.get_logger().error("Failed to open video writer.")
            self.writer = None
            return

        self.get_logger().info(f"Video writer initialized: {w}x{h}")

    def write_frame(self):
        if self.latest_frame is None:
            return

        if self.writer is None:
            self.init_writer(self.latest_frame)

        if self.writer is None:
            return

        frame = self.latest_frame

        h, w = frame.shape[:2]
        if self.frame_size != (w, h):
            frame = cv2.resize(frame, self.frame_size)

        self.writer.write(frame)
        self.frames_written += 1

        if self.frames_written % int(self.fps * 5) == 0:
            self.get_logger().info(f"Written frames: {self.frames_written}")

    def destroy_node(self):
        if self.writer is not None:
            self.writer.release()
            self.get_logger().info(
                f"Saved video: {self.output_path}, frames: {self.frames_written}"
            )

        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = VideoRecorderNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()