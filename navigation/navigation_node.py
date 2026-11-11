import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import Float32
from sensor_msgs.msg import LaserScan, Imu
from math import radians, degrees
from tf_transformations import euler_from_quaternion
import numpy as np
from sklearn.cluster import DBSCAN
from mechaship_interfaces.msg import RgbwLedColor

class NavigationNode(Node):
    def __init__(self) -> None:
        super().__init__(
            "navigation_node",
            allow_undeclared_parameters=True,
            automatically_declare_parameters_from_overrides=True,
        )

        self.front_lidar = []
        self.roll_degree = 0
        self.pitch_degree = 0
        self.yaw_degree = 0
        self.previous_servo_angle = 90.0

        # Subscribers
        self.lidar_subscription = self.create_subscription(
            LaserScan,
            "/scan",
            self.lidar_callback,
            qos_profile_sensor_data,
        )

        self.imu_subscription = self.create_subscription(
            Imu,
            "/imu",
            self.imu_callback,
            qos_profile_sensor_data,
        )

        # Publishers
        self.thruster_publisher = self.create_publisher(Float32, "/actuator/thruster/percentage", 10)
        self.servo_publisher = self.create_publisher(Float32, "/actuator/key/degree", 10)
        self.rgbw_led_publisher = self.create_publisher(
            RgbwLedColor, "/actuator/rgbwled/color", 10
        )

        self.set_white_led()

        # Timer
        self.create_timer(0.5, self.timer_callback)  # 0.5초 주기로 타이머 실행

    #

    def lidar_callback(self, data: LaserScan):
        total_ranges = len(data.ranges)
        fov_degrees = 160
        start_index = (total_ranges // 2) - (total_ranges * fov_degrees // 360 // 2)
        end_index = (total_ranges // 2) + (total_ranges * fov_degrees // 360 // 2)
        self.front_lidar = data.ranges[start_index:end_index]

    def imu_callback(self, msg: Imu):
        quaternion = (
            msg.orientation.x,
            msg.orientation.y,
            msg.orientation.z,
            msg.orientation.w,
        )
        roll_rad, pitch_rad, yaw_rad = euler_from_quaternion(quaternion)
        self.roll_degree = degrees(roll_rad) + 180
        self.pitch_degree = degrees(pitch_rad) + 180
        self.yaw_degree = degrees(yaw_rad) + 180

    def timer_callback(self):
        filtered_lidar = self.filter_lidar_data(self.front_lidar)
        clusters = self.cluster_lidar_data(filtered_lidar)

        if clusters:
            angle = self.calculate_avoidance_angle(clusters)
            self.set_thruster_and_servo(angle)
        else:
            self.set_thruster_and_servo(0)  # 기본값
            self.set_white_led()

    def filter_lidar_data(self, lidar_data):
        return [d for d in lidar_data if 0.1 < d < 10.0 and not np.isinf(d) and not np.isnan(d)]

    def cluster_lidar_data(self, lidar_data):
        if not lidar_data:
            return []

        total_ranges = len(lidar_data)
        angles = np.linspace(-80, 80, total_ranges)  # FOV 160도
        points = np.array([
            [d * np.cos(np.radians(a)), d * np.sin(np.radians(a))]
            for d, a in zip(lidar_data, angles) if d > 0
        ])

        # DBSCAN 클러스터링
        db = DBSCAN(eps=0.5, min_samples=7).fit(points)
        labels = db.labels_

        # 클러스터별 중심 계산
        clusters = {}
        for label in set(labels):
            if label == -1:  # 노이즈 포인트
                continue
            cluster_points = points[labels == label]
            center = np.mean(cluster_points, axis=0)
            clusters[label] = center

        return clusters

    def calculate_safe_zones(self, lidar_data, clusters):
        if not lidar_data or not clusters:
            return [(0, 180)], [90]  # 장애물이 없을 경우 전체가 안전 구역

        total_ranges = len(lidar_data)
        angles = np.linspace(-80, 80, total_ranges)
        unsafe_zones = []

        safety_margin = 5  # 안전 마진 (각도)
        # ⚠️ 원본 rtf 파일이 여기서 끊겨 있음 (이후 calculate_avoidance_angle, set_thruster_and_servo, set_white_led, main 없음)
