import cv2
import numpy as np

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


class YOLOSegNode(Node):
    def __init__(self):
        super().__init__("yolo_seg_node")

        self.declare_parameter("model_path", "")
        self.declare_parameter("rgb_topic", "/camera/rgb/image_raw")
        self.declare_parameter("tree_mask_topic", "/segmentation/tree_mask")
        self.declare_parameter("road_mask_topic", "/segmentation/road_mask")
        self.declare_parameter("debug_image_topic", "/segmentation/debug_image")
        self.declare_parameter("tree_info_topic", "/segmentation/tree_info")

        self.declare_parameter("imgsz", 640)
        self.declare_parameter("conf", 0.35)
        self.declare_parameter("tree_class_id", 0)
        self.declare_parameter("road_class_id", 1)
        self.declare_parameter("publish_debug", True)

        self.model_path = self.get_parameter("model_path").value
        self.rgb_topic = self.get_parameter("rgb_topic").value
        self.tree_mask_topic = self.get_parameter("tree_mask_topic").value
        self.road_mask_topic = self.get_parameter("road_mask_topic").value
        self.debug_image_topic = self.get_parameter("debug_image_topic").value
        self.tree_info_topic = self.get_parameter("tree_info_topic").value

        self.imgsz = int(self.get_parameter("imgsz").value)
        self.conf = float(self.get_parameter("conf").value)
        self.tree_class_id = int(self.get_parameter("tree_class_id").value)
        self.road_class_id = int(self.get_parameter("road_class_id").value)
        self.publish_debug = bool(self.get_parameter("publish_debug").value)

        self.bridge = CvBridge()

        if YOLO is None:
            self.get_logger().error("Ultralytics is not installed. Run: pip install ultralytics")
            self.model = None
        elif not self.model_path:
            self.get_logger().error("Parameter 'model_path' is empty.")
            self.model = None
        else:
            self.get_logger().info(f"Loading YOLO model: {self.model_path}")
            self.model = YOLO(self.model_path)
            self.model.to('cuda')

        self.tree_mask_pub = self.create_publisher(Image, self.tree_mask_topic, 10)
        self.road_mask_pub = self.create_publisher(Image, self.road_mask_topic, 10)
        self.debug_pub = self.create_publisher(Image, self.debug_image_topic, 10)
        self.tree_info_pub = self.create_publisher(Float32MultiArray, self.tree_info_topic, 10)

        self.rgb_sub = self.create_subscription(
            Image,
            self.rgb_topic,
            self.image_callback,
            10
        )

        self.get_logger().info("YOLO segmentation node started.")

    def publish_tree_info(self, distance, x, y, conf, detected):
        values = [distance, x, y, conf, detected]

        clean_values = []
        for v in values:
            v = float(v)
            if not np.isfinite(v):
                v = 0.0
            clean_values.append(v)

        msg = Float32MultiArray()
        msg.data = clean_values
        self.tree_info_pub.publish(msg)

    def image_callback(self, msg: Image):
        if self.model is None:
            return

        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as exc:
            self.get_logger().error(f"Failed to convert RGB image: {exc}")
            self.publish_tree_info(0.0, 0.0, 0.0, 0.0, 0.0)
            return

        height, width = frame.shape[:2]

        try:
            result = self.model.predict(
                source=frame,
                imgsz=self.imgsz,
                conf=self.conf,
                verbose=False,
                device=0
            )[0]
        except Exception as exc:
            self.get_logger().error(f"YOLO inference failed: {exc}")
            self.publish_tree_info(0.0, 0.0, 0.0, 0.0, 0.0)
            return

        tree_mask = np.zeros((height, width), dtype=np.uint8)
        road_mask = np.zeros((height, width), dtype=np.uint8)

        best_tree_conf = 0.0
        best_tree_center_x = 0.0
        best_tree_center_y = 0.0
        tree_detected = 0.0

        if result.masks is not None and result.boxes is not None:
            masks = result.masks.data.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy().astype(int)
            self.get_logger().info(f"classes={classes}, confs={result.boxes.conf.cpu().numpy()}")
            confs = result.boxes.conf.cpu().numpy()

            for mask, cls_id, conf in zip(masks, classes, confs):
                resized_mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
                binary_mask = (resized_mask > 0.5).astype(np.uint8) * 255

                if cls_id == self.tree_class_id:
                    tree_mask = cv2.bitwise_or(tree_mask, binary_mask)

                    if conf > best_tree_conf:
                        best_tree_conf = float(conf)

                        ys, xs = np.where(binary_mask > 0)
                        if len(xs) > 0 and len(ys) > 0:
                            best_tree_center_x = float(np.mean(xs) / width)
                            best_tree_center_y = float(np.mean(ys) / height)
                            tree_detected = 1.0

                elif cls_id == self.road_class_id:
                    road_mask = cv2.bitwise_or(road_mask, binary_mask)

        tree_msg = self.bridge.cv2_to_imgmsg(tree_mask, encoding="mono8")
        tree_msg.header = msg.header

        road_msg = self.bridge.cv2_to_imgmsg(road_mask, encoding="mono8")
        road_msg.header = msg.header

        self.tree_mask_pub.publish(tree_msg)
        self.road_mask_pub.publish(road_msg)

        # Тут distance пока 0.0, потому что в RGB-сегментации нет глубины.
        # Если у тебя есть depth-нода, distance надо брать оттуда.
        if tree_detected == 1.0:
            self.publish_tree_info(
                0.0,
                best_tree_center_x,
                best_tree_center_y,
                best_tree_conf,
                1.0
            )
        else:
            self.publish_tree_info(0.0, 0.0, 0.0, 0.0, 0.0)

        if self.publish_debug:
            debug = frame.copy()

            road_overlay = np.zeros_like(debug)
            road_overlay[:, :, 1] = road_mask

            tree_overlay = np.zeros_like(debug)
            tree_overlay[:, :, 2] = tree_mask

            debug = cv2.addWeighted(debug, 1.0, road_overlay, 0.35, 0)
            debug = cv2.addWeighted(debug, 1.0, tree_overlay, 0.35, 0)

            debug_msg = self.bridge.cv2_to_imgmsg(debug, encoding="bgr8")
            debug_msg.header = msg.header
            self.debug_pub.publish(debug_msg)


def main(args=None):
    rclpy.init(args=args)
    node = YOLOSegNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()