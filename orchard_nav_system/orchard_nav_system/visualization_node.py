import math
from pathlib import Path

import cv2
import numpy as np

import rclpy
from rclpy.node import Node

from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import Twist

from PIL import Image as PILImage
from PIL import ImageDraw, ImageFont


class VisualizationNode(Node):
    def __init__(self):
        super().__init__("visualization_node")

        self.declare_parameter("rgb_topic_in", "/camera/rgb/image_raw")
        self.declare_parameter("tree_mask_topic_in", "/segmentation/tree_mask")
        self.declare_parameter("tree_info_topic_in", "/perception/tree_info")
        self.declare_parameter("cmd_vel_topic_in", "/cmd_vel")
        self.declare_parameter("debug_overlay_topic_out", "/visualization/debug_overlay")

        self.declare_parameter("mask_alpha", 0.28)
        self.declare_parameter("min_contour_area", 150.0)
        self.declare_parameter("font_size", 20)

        rgb_topic = self.get_parameter("rgb_topic_in").value
        tree_mask_topic = self.get_parameter("tree_mask_topic_in").value
        tree_info_topic = self.get_parameter("tree_info_topic_in").value
        cmd_vel_topic = self.get_parameter("cmd_vel_topic_in").value
        debug_overlay_topic = self.get_parameter("debug_overlay_topic_out").value

        self.mask_alpha = float(self.get_parameter("mask_alpha").value)
        self.min_contour_area = float(self.get_parameter("min_contour_area").value)
        self.font_size = int(self.get_parameter("font_size").value)

        self.bridge = CvBridge()

        self.latest_tree_mask = None
        self.latest_tree_info = None
        self.latest_cmd_vel = None

        self.rgb_sub = self.create_subscription(
            Image, rgb_topic, self.rgb_callback, 10
        )
        self.tree_mask_sub = self.create_subscription(
            Image, tree_mask_topic, self.tree_mask_callback, 10
        )
        self.tree_info_sub = self.create_subscription(
            Float32MultiArray, tree_info_topic, self.tree_info_callback, 10
        )
        self.cmd_vel_sub = self.create_subscription(
            Twist, cmd_vel_topic, self.cmd_vel_callback, 10
        )

        self.debug_pub = self.create_publisher(Image, debug_overlay_topic, 10)

        self.font_regular = self.load_font_regular(self.font_size)
        self.font_bold = self.load_font_bold(self.font_size + 4)
        self.font_small = self.load_font_regular(max(15, self.font_size - 2))

        self.get_logger().info("Visualization node started.")

    def load_font_regular(self, size: int):
        candidates = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
            "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
        ]
        for path in candidates:
            if Path(path).exists():
                try:
                    return ImageFont.truetype(path, size=size)
                except Exception:
                    pass
        return ImageFont.load_default()

    def load_font_bold(self, size: int):
        candidates = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
            "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
        ]
        for path in candidates:
            if Path(path).exists():
                try:
                    return ImageFont.truetype(path, size=size)
                except Exception:
                    pass
        return ImageFont.load_default()

    @staticmethod
    def bgr_to_pil(frame_bgr: np.ndarray) -> PILImage.Image:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        return PILImage.fromarray(frame_rgb)

    @staticmethod
    def pil_to_bgr(image_pil: PILImage.Image) -> np.ndarray:
        frame_rgb = np.array(image_pil)
        return cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

    def draw_text_pil(self, frame_bgr: np.ndarray, text: str, position, font, color=(0, 0, 0)):
        image_pil = self.bgr_to_pil(frame_bgr)
        draw = ImageDraw.Draw(image_pil)
        draw.text(position, text, font=font, fill=color)
        return self.pil_to_bgr(image_pil)

    def draw_panel(self, frame: np.ndarray, x1: int, y1: int, x2: int, y2: int,
                   color=(245, 245, 245), alpha=0.58) -> np.ndarray:
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        return cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0)

    def tree_mask_callback(self, msg: Image):
        try:
            mask = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            if mask is None:
                return
            if len(mask.shape) == 3:
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            if mask.dtype != np.uint8:
                mask = mask.astype(np.uint8)
            self.latest_tree_mask = mask
        except Exception as exc:
            self.get_logger().warning(f"Failed to parse tree mask: {exc}")

    def tree_info_callback(self, msg: Float32MultiArray):
        try:
            self.latest_tree_info = list(msg.data)
        except Exception as exc:
            self.get_logger().warning(f"Failed to parse tree_info: {exc}")

    def cmd_vel_callback(self, msg: Twist):
        self.latest_cmd_vel = msg

    @staticmethod
    def command_from_velocity(valid: float, linear_x: float, angular_z: float) -> str:
        if valid < 0.5:
            return "МЕДЛЕННО ВПЕРЁД"
        if abs(angular_z) < 0.08:
            return "ДВИЖЕНИЕ ПРЯМО"
        if angular_z > 0:
            return "ПОВОРОТ ВЛЕВО"
        return "ПОВОРОТ ВПРАВО"

    @staticmethod
    def command_color(valid: float, angular_z: float):
        if valid < 0.5:
            return (0, 200, 255)
        if abs(angular_z) < 0.08:
            return (0, 180, 0)
        return (0, 150, 255)

    def parse_tree_info(self):
        distance = float("inf")
        mean_x = 0.0
        mean_z = 0.0
        angle = 0.0
        left_ratio = 0.0
        right_ratio = 0.0
        area_ratio = 0.0
        valid = 0.0

        if self.latest_tree_info is not None and len(self.latest_tree_info) >= 8:
            distance = float(self.latest_tree_info[0])
            mean_x = float(self.latest_tree_info[1])
            mean_z = float(self.latest_tree_info[2])
            angle = float(self.latest_tree_info[3])
            left_ratio = float(self.latest_tree_info[4])
            right_ratio = float(self.latest_tree_info[5])
            area_ratio = float(self.latest_tree_info[6])
            valid = float(self.latest_tree_info[7])

        return distance, mean_x, mean_z, angle, left_ratio, right_ratio, area_ratio, valid

    def parse_cmd_vel(self):
        linear_x = 0.0
        angular_z = 0.0
        if self.latest_cmd_vel is not None:
            linear_x = float(self.latest_cmd_vel.linear.x)
            angular_z = float(self.latest_cmd_vel.angular.z)
        return linear_x, angular_z

    def draw_tree_mask_and_contours(self, frame: np.ndarray, distance: float, angle: float):
        h, w = frame.shape[:2]
        target_center = None

        if self.latest_tree_mask is None:
            return frame, target_center

        mask = self.latest_tree_mask
        if mask.shape[:2] != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

        binary = (mask > 0).astype(np.uint8) * 255

        color_mask = np.zeros_like(frame)
        color_mask[:, :, 1] = binary
        frame = cv2.addWeighted(frame, 1.0, color_mask, self.mask_alpha, 0)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        valid_contours = [
            contour for contour in contours
            if cv2.contourArea(contour) >= self.min_contour_area
        ]

        if not valid_contours:
            return frame, target_center

        cv2.drawContours(frame, valid_contours, -1, (0, 0, 255), 2)

        target_contour = max(valid_contours, key=cv2.contourArea)
        moments = cv2.moments(target_contour)

        if moments["m00"] != 0:
            cx = int(moments["m10"] / moments["m00"])
            cy = int(moments["m01"] / moments["m00"])
            target_center = (cx, cy)

        return frame, target_center

    def draw_info_panel(
        self,
        frame: np.ndarray,
        distance: float,
        mean_x: float,
        mean_z: float,
        angle: float,
        left_ratio: float,
        right_ratio: float,
        area_ratio: float,
        valid: float,
        linear_x: float,
        angular_z: float,
        command: str,
    ) -> np.ndarray:
        margin = 20
        panel_w = 380
        panel_h = 215

        panel_x1 = margin
        panel_y1 = margin
        panel_x2 = panel_x1 + panel_w
        panel_y2 = panel_y1 + panel_h

        frame = self.draw_panel(
            frame,
            panel_x1,
            panel_y1,
            panel_x2,
            panel_y2,
            color=(248, 248, 248),
            alpha=0.52,
        )

        title = "ДЕРЕВО ОБНАРУЖЕНО" if valid >= 0.5 else "ДЕРЕВО НЕ ОБНАРУЖЕНО"
        dist_text = f"{distance:.2f} м" if math.isfinite(distance) else "нет данных"

        lines = [
            title,
            f"Расстояние: {dist_text}",
            f"Угол: {angle:.3f} рад",
            f"Смещение X: {mean_x:.3f} м",
            f"Глубина Z: {mean_z:.3f} м",
            f"Лево / право: {left_ratio:.2f} / {right_ratio:.2f}",
            f"Площадь дерева: {area_ratio:.3f}",
            f"Лин. скорость: {linear_x:.2f} м/с",
            f"Угл. скорость: {angular_z:.2f} рад/с",
            f"Команда: {command}",
        ]

        y = panel_y1 + 14
        for i, line in enumerate(lines):
            if i == 0:
                font = self.font_bold
                step = 28
            else:
                font = self.font_small
                step = 18

            frame = self.draw_text_pil(
                frame,
                line,
                (panel_x1 + 14, y),
                font,
                color=(0, 0, 0)
            )
            y += step

        return frame

    def draw_motion_arrow(self, frame: np.ndarray, valid: float, linear_x: float, angular_z: float):
        h, w = frame.shape[:2]

        margin = 20
        panel_w = 260
        panel_h = 160

        panel_x2 = w - margin
        panel_y1 = margin
        panel_x1 = panel_x2 - panel_w
        panel_y2 = panel_y1 + panel_h

        frame = self.draw_panel(
            frame,
            panel_x1,
            panel_y1,
            panel_x2,
            panel_y2,
            color=(248, 248, 248),
            alpha=0.50,
        )

        frame = self.draw_text_pil(
            frame,
            "Направление движения",
            (panel_x1 + 2, panel_y1 + 12),
            self.font_regular,
            color=(0, 0, 0)
        )

        arrow_color = self.command_color(valid, angular_z)

        # стрелка опущена ниже
        origin = ((panel_x1 + panel_x2) // 2, panel_y1 + 125)
        arrow_len = 70

        if valid < 0.5:
            end_point = (origin[0], origin[1] - arrow_len)
        elif angular_z > 0.08:
            end_point = (origin[0] - arrow_len, origin[1])
        elif angular_z < -0.08:
            end_point = (origin[0] + arrow_len, origin[1])
        else:
            end_point = (origin[0], origin[1] - arrow_len)

        cv2.circle(frame, origin, 4, (0, 0, 0), -1)
        cv2.arrowedLine(frame, origin, end_point, arrow_color, 5, tipLength=0.28)

        return frame

    def rgb_callback(self, msg: Image):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as exc:
            self.get_logger().warning(f"Failed to parse RGB frame: {exc}")
            return

        distance, mean_x, mean_z, angle, left_ratio, right_ratio, area_ratio, valid = self.parse_tree_info()
        linear_x, angular_z = self.parse_cmd_vel()
        command = self.command_from_velocity(valid, linear_x, angular_z)

        frame, _ = self.draw_tree_mask_and_contours(frame, distance, angle)

        frame = self.draw_info_panel(
            frame,
            distance,
            mean_x,
            mean_z,
            angle,
            left_ratio,
            right_ratio,
            area_ratio,
            valid,
            linear_x,
            angular_z,
            command,
        )

        frame = self.draw_motion_arrow(frame, valid, linear_x, angular_z)

        out_msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        out_msg.header = msg.header
        self.debug_pub.publish(out_msg)


def main(args=None):
    rclpy.init(args=args)
    node = VisualizationNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()