import math

import rclpy
from rclpy.node import Node

from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import Twist


class RobotControlNode(Node):
    """
    Tree-oriented reactive controller.

    Input /perception/tree_info:
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

    Output:
        /cmd_vel

    Logic:
        - if no tree: move forward slowly
        - if tree is close and on the left: turn right
        - if tree is close and on the right: turn left
        - if tree is centered and close: slow down / stop
        - steering is based on tree 3D angle, not road center
    """

    def __init__(self):
        super().__init__("robot_control_node")

        self.declare_parameter("tree_info_topic", "/perception/tree_info")
        self.declare_parameter("cmd_vel_topic", "/cmd_vel")

        self.declare_parameter("safe_stop_distance", 0.8)
        self.declare_parameter("slow_distance", 2.5)

        self.declare_parameter("max_linear_speed", 0.35)
        self.declare_parameter("min_linear_speed", 0.08)
        self.declare_parameter("search_linear_speed", 0.12)

        self.declare_parameter("max_angular_speed", 0.8)
        self.declare_parameter("angle_kp", 1.2)
        self.declare_parameter("side_balance_kp", 0.4)

        self.declare_parameter("center_angle_zone_rad", 0.08)
        self.declare_parameter("tree_area_stop_ratio", 0.35)

        # If detections flicker, keep last valid observation for a short time.
        self.declare_parameter("hold_last_detection_sec", 0.7)

        self.tree_info_topic = self.get_parameter("tree_info_topic").value
        self.cmd_vel_topic = self.get_parameter("cmd_vel_topic").value

        self.safe_stop_distance = float(self.get_parameter("safe_stop_distance").value)
        self.slow_distance = float(self.get_parameter("slow_distance").value)

        self.max_linear_speed = float(self.get_parameter("max_linear_speed").value)
        self.min_linear_speed = float(self.get_parameter("min_linear_speed").value)
        self.search_linear_speed = float(self.get_parameter("search_linear_speed").value)

        self.max_angular_speed = float(self.get_parameter("max_angular_speed").value)
        self.angle_kp = float(self.get_parameter("angle_kp").value)
        self.side_balance_kp = float(self.get_parameter("side_balance_kp").value)

        self.center_angle_zone_rad = float(self.get_parameter("center_angle_zone_rad").value)
        self.tree_area_stop_ratio = float (self.get_parameter("tree_area_stop_ratio").value)

        self.hold_last_detection_sec = float(self.get_parameter("hold_last_detection_sec").value)

        self.declare_parameter("log_interval_sec", 1.0)
        self.declare_parameter("log_linear_delta", 0.03)
        self.declare_parameter("log_angular_delta", 0.05)
        self.declare_parameter("log_distance_delta", 0.25)

        self.log_interval_sec = float(self.get_parameter("log_interval_sec").value)
        self.log_linear_delta = float(self.get_parameter("log_linear_delta").value)
        self.log_angular_delta = float(self.get_parameter("log_angular_delta").value)
        self.log_distance_delta = float(self.get_parameter("log_distance_delta").value)

        self.last_log_time = None
        self.last_logged_state = None

        self.last_valid_info = None
        self.last_valid_time = None

        self.cmd_pub = self.create_publisher(Twist, self.cmd_vel_topic, 10)

        self.sub = self.create_subscription(
            Float32MultiArray,
            self.tree_info_topic,
            self.tree_info_callback,
            10
        )

        self.get_logger().info("Tree-oriented robot control node started.")

    @staticmethod
    def clamp(value, min_value, max_value):
        return max(min_value, min(max_value, value))

    def compute_linear_speed(self, distance_m: float, valid: bool):
        if not valid or math.isinf(distance_m):
            return self.search_linear_speed

        if distance_m <= self.safe_stop_distance:
            return 0.0

        if distance_m >= self.slow_distance:
            return self.max_linear_speed

        ratio = (distance_m - self.safe_stop_distance) / (
            self.slow_distance - self.safe_stop_distance
        )

        speed = self.min_linear_speed + ratio * (
            self.max_linear_speed - self.min_linear_speed
        )

        return self.clamp(speed, 0.0, self.max_linear_speed)

    def get_smoothed_info(self, data):
        now = self.get_clock().now()

        distance_m = float(data[0])
        mean_x_m = float(data[1])
        mean_z_m = float(data[2])
        angle_rad = float(data[3])
        left_ratio = float(data[4])
        right_ratio = float(data[5])
        tree_area_ratio = float(data[6])
        valid = bool(data[7] > 0.5)

        current = {
            "distance_m": distance_m,
            "mean_x_m": mean_x_m,
            "mean_z_m": mean_z_m,
            "angle_rad": angle_rad,
            "left_ratio": left_ratio,
            "right_ratio": right_ratio,
            "tree_area_ratio": tree_area_ratio,
            "valid": valid,
        }

        if valid:
            self.last_valid_info = current
            self.last_valid_time = now
            return current

        if self.last_valid_info is not None and self.last_valid_time is not None:
            dt = (now - self.last_valid_time).nanoseconds / 1e9
            if dt <= self.hold_last_detection_sec:
                held = dict(self.last_valid_info)
                held["valid"] = True
                return held

        return current

    def tree_info_callback(self, msg: Float32MultiArray):
        if len(msg.data) < 8:
            self.get_logger().warning("Invalid tree_info message. Expected 8 values.")
            return

        info = self.get_smoothed_info(msg.data)

        distance_m = info["distance_m"]
        angle_rad = info["angle_rad"]
        left_ratio = info["left_ratio"]
        right_ratio = info["right_ratio"]
        tree_area_ratio = info["tree_area_ratio"]
        valid = info["valid"]

        cmd = Twist()

        linear_speed = self.compute_linear_speed(distance_m, valid)

        angular = 0.0

        if valid and not math.isinf(distance_m):
            # Main tree-based steering.
            #
            # angle_rad < 0: tree is left.
            # To avoid a tree on the left, robot should turn right.
            # ROS angular.z > 0 means turn left.
            # Therefore angular follows angle_rad sign:
            #   tree left  -> angle negative -> angular negative -> turn right
            #   tree right -> angle positive -> angular positive -> turn left
            angular += self.angle_kp * angle_rad

            # Additional balancing by left/right mask distribution.
            # If more tree pixels are on the left, turn right.
            # If more tree pixels are on the right, turn left.
            side_error = right_ratio - left_ratio
            angular += self.side_balance_kp * side_error

            tree_is_close = distance_m <= self.safe_stop_distance
            tree_is_centered = abs(angle_rad) <= self.center_angle_zone_rad
            tree_is_large = tree_area_ratio >= self.tree_area_stop_ratio

            if tree_is_close and (tree_is_centered or tree_is_large):
                linear_speed = 0.0

                if abs(angle_rad) < 0.03:
                    angular = self.max_angular_speed
                else:
                    angular = self.max_angular_speed if angle_rad > 0 else -self.max_angular_speed

        else:
            # No tree detected: move forward slowly.
            # In a more advanced version, this could switch to search behavior.
            linear_speed = self.search_linear_speed
            angular = 0.0

        cmd.linear.x = self.clamp(linear_speed, 0.0, self.max_linear_speed)
        cmd.angular.z = self.clamp(
            angular,
            -self.max_angular_speed,
            self.max_angular_speed
        )
        
        if self.should_log_control(distance_m, valid, cmd.linear.x, cmd.angular.z):
            distance_text = "inf" if math.isinf(distance_m) else f"{distance_m:.2f}"

            self.get_logger().info(
                "[CONTROL] "
                f"valid={int(valid)} "
                f"dist={distance_text}m "
                f"angle={angle_rad:.3f}rad "
                f"left={left_ratio:.2f} "
                f"right={right_ratio:.2f} "
                f"linear={cmd.linear.x:.2f} "
                f"angular={cmd.angular.z:.2f}"
            )

        self.cmd_pub.publish(cmd)

    def should_log_control(self, distance_m, valid, linear_x, angular_z):
        now = self.get_clock().now()

        current_state = {
            "distance_m": distance_m,
            "valid": valid,
            "linear_x": linear_x,
            "angular_z": angular_z,
        }

        if self.last_log_time is None or self.last_logged_state is None:
            self.last_log_time = now
            self.last_logged_state = current_state
            return True     

        dt = (now - self.last_log_time).nanoseconds / 1e9

        valid_changed = current_state["valid"] != self.last_logged_state["valid"]

        linear_changed = abs(
            current_state["linear_x"] - self.last_logged_state["linear_x"]
        ) >= self.log_linear_delta

        angular_changed = abs(
            current_state["angular_z"] - self.last_logged_state["angular_z"]
        ) >= self.log_angular_delta

        if math.isinf(distance_m) or math.isinf(self.last_logged_state["distance_m"]):
            distance_changed = current_state["valid"] != self.last_logged_state["valid"]
        else:
            distance_changed = abs(
                current_state["distance_m"] - self.last_logged_state["distance_m"]
            ) >= self.log_distance_delta

        if dt >= self.log_interval_sec and (
            valid_changed or linear_changed or angular_changed or distance_changed
        ):
            self.last_log_time = now
            self.last_logged_state = current_state
            return True

        return False


def main(args=None):
    rclpy.init(args=args)
    node = RobotControlNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        stop_cmd = Twist()
        node.cmd_pub.publish(stop_cmd)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()