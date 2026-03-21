"""
姿态估计模块
使用 YOLOv8-Pose 检测人体骨骼关键点
"""
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path

# YOLOv8-Pose 关键点索引 (COCO 17点)
COCO_KEYPOINTS = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
]

# COCO 骨架连接
COCO_SKELETON = [
    [0, 1], [0, 2], [1, 3], [2, 4],  # 头
    [5, 6], [5, 7], [7, 9], [6, 8], [8, 10],  # 上肢
    [5, 6], [5, 11], [6, 12], [11, 12],  # 躯干
    [11, 13], [13, 15], [12, 14], [14, 16]  # 下肢
]


class PoseKeypoints:
    """姿态关键点数据类"""

    def __init__(self, keypoints: np.ndarray, scores: np.ndarray):
        """
        初始化关键点数据

        Args:
            keypoints: 关键点坐标 (N, 2) - x, y
            scores: 关键点置信度 (N,)
        """
        self.keypoints = np.array(keypoints, dtype=np.float32)
        self.scores = np.array(scores, dtype=np.float32)

    @property
    def num_keypoints(self) -> int:
        """关键点数量"""
        return len(self.keypoints)

    @property
    def is_valid(self) -> bool:
        """是否有效（至少有部分关键点）"""
        return self.num_keypoints > 0 and np.any(self.scores > 0)

    def get_confident_keypoints(self, threshold: float = 0.3) -> Tuple[np.ndarray, np.ndarray]:
        """
        获取高置信度关键点

        Args:
            threshold: 置信度阈值

        Returns:
            (关键点坐标, 置信度)
        """
        mask = self.scores >= threshold
        return self.keypoints[mask], self.scores[mask]

    def normalize(self, image_size: Tuple[int, int]) -> 'PoseKeypoints':
        """
        归一化关键点坐标到 [0, 1]

        Args:
            image_size: 图像尺寸 (width, height)

        Returns:
            归一化后的关键点
        """
        width, height = image_size
        normalized_keypoints = self.keypoints.copy()
        normalized_keypoints[:, 0] /= width
        normalized_keypoints[:, 1] /= height

        return PoseKeypoints(normalized_keypoints, self.scores.copy())

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'keypoints': self.keypoints.tolist(),
            'scores': self.scores.tolist(),
            'num_keypoints': self.num_keypoints
        }

    def __repr__(self):
        return f"PoseKeypoints(points={self.num_keypoints}, valid={self.is_valid})"


class PoseEstimator:
    """姿态估计器"""

    def __init__(self,
                 model_name: str = "yolov8n-pose.pt",
                 confidence_threshold: float = 0.5,
                 nms_threshold: float = 0.4):
        """
        初始化姿态估计器

        Args:
            model_name: YOLOv8 模型名称
            confidence_threshold: 置信度阈值
            nms_threshold: NMS 阈值
        """
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.model = None
        self._load_model()

    def _load_model(self):
        """加载 YOLOv8-Pose 模型"""
        try:
            from ultralytics import YOLO
            # 尝试加载模型，如果不存在会自动下载
            model_path = Path(__file__).parent.parent.parent / "models" / self.model_name
            if model_path.exists():
                self.model = YOLO(str(model_path))
            else:
                # 使用预训练模型
                self.model = YOLO(self.model_name)
            print(f"✓ 姿态估计模型加载成功: {self.model_name}")
        except ImportError:
            raise ImportError("请安装 ultralytics: pip install ultralytics")
        except Exception as e:
            raise RuntimeError(f"模型加载失败: {e}")

    def estimate(self, frame: np.ndarray) -> List[PoseKeypoints]:
        """
        估计帧中所有人的姿态

        Args:
            frame: 输入图像 (H, W, C)

        Returns:
            关键点列表
        """
        if self.model is None:
            raise RuntimeError("模型未加载")

        # YOLOv8-Pose 推理
        results = self.model(frame, conf=self.confidence_threshold, iou=self.nms_threshold, verbose=False)

        pose_keypoints = []

        if results and len(results) > 0:
            result = results[0]

            # 检查是否有关键点数据
            if result.keypoints is not None and len(result.keypoints) > 0:
                keypoints_data = result.keypoints.data  # (N, 17, 3)

                for person_keypoints in keypoints_data:
                    # person_keypoints 形状: (17, 3) - x, y, confidence
                    coords = person_keypoints[:, :2].cpu().numpy()
                    scores = person_keypoints[:, 2].cpu().numpy()

                    # 只保留置信度 > 0 的关键点
                    valid_mask = scores > 0
                    if np.any(valid_mask):
                        pose_keypoints.append(PoseKeypoints(coords, scores))

        return pose_keypoints

    def estimate_single(self, frame: np.ndarray) -> Optional[PoseKeypoints]:
        """
        估计单个人的姿态（置信度最高的）

        Args:
            frame: 输入图像

        Returns:
            关键点数据，如果没有检测到则返回 None
        """
        poses = self.estimate(frame)
        if poses:
            # 返回置信度最高的人
            best_pose = max(poses, key=lambda p: np.mean(p.scores))
            return best_pose
        return None


class PoseVisualizer:
    """姿态可视化工具"""

    @staticmethod
    def draw_pose(frame: np.ndarray,
                  keypoints: PoseKeypoints,
                  thickness: int = 2) -> np.ndarray:
        """
        在图像上绘制骨架

        Args:
            frame: 输入图像
            keypoints: 关键点数据
            thickness: 线条粗细

        Returns:
            绘制后的图像
        """
        import cv2

        output = frame.copy()
        h, w = frame.shape[:2]

        # 绘制骨架连接
        for start_idx, end_idx in COCO_SKELETON:
            if start_idx < keypoints.num_keypoints and end_idx < keypoints.num_keypoints:
                start_point = tuple(keypoints.keypoints[start_idx].astype(int))
                end_point = tuple(keypoints.keypoints[end_idx].astype(int))

                # 检查坐标是否在图像范围内
                if (0 <= start_point[0] < w and 0 <= start_point[1] < h and
                    0 <= end_point[0] < w and 0 <= end_point[1] < h):
                    # 根据置信度设置颜色
                    score = min(keypoints.scores[start_idx], keypoints.scores[end_idx])
                    color = (0, 255, 0) if score > 0.5 else (0, 165, 255)
                    cv2.line(output, start_point, end_point, color, thickness)

        # 绘制关键点
        for i, (point, score) in enumerate(zip(keypoints.keypoints, keypoints.scores)):
            if score > 0:
                pt = tuple(point.astype(int))
                if 0 <= pt[0] < w and 0 <= pt[1] < h:
                    # 关键点颜色根据置信度
                    color = (0, 255, 0) if score > 0.5 else (0, 165, 255)
                    cv2.circle(output, pt, 4, color, -1)
                    # 标注关键点编号
                    cv2.putText(output, str(i), (pt[0] + 5, pt[1] - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        return output

    @staticmethod
    def draw_action_label(frame: np.ndarray,
                          action: str,
                          confidence: float,
                          position: Tuple[int, int] = (10, 30)) -> np.ndarray:
        """
        在图像上绘制动作标签

        Args:
            frame: 输入图像
            action: 动作类别
            confidence: 置信度
            position: 文字位置

        Returns:
            绘制后的图像
        """
        import cv2

        output = frame.copy()
        label = f"{action}: {confidence:.2f}"

        # 绘制背景
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(output,
                     (position[0] - 5, position[1] - text_height - 5),
                     (position[0] + text_width + 5, position[1] + 5),
                     (0, 0, 0), -1)

        # 绘制文字
        cv2.putText(output, label, position,
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        return output
