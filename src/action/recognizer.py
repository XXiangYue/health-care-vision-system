"""
行为识别模块
使用 STGCN 时空图卷积网络进行动作分类
"""
import numpy as np
from typing import List, Optional, Dict, Any
from pathlib import Path
import pickle

# 图卷积邻接矩阵定义（COCO 17点骨架）
COCO_GRAPH_LAYOUT = {
    'num_nodes': 17,
    'edges': [
        (0, 1), (0, 2), (1, 3), (2, 4),  # 头
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # 上肢
        (5, 6), (5, 11), (6, 12), (11, 12),  # 躯干
        (11, 13), (13, 15), (12, 14), (14, 16)  # 下肢
    ],
    'self_loop': True
}


def build_adjacency_matrix(num_nodes: int = 17, edges: List[tuple] = None) -> np.ndarray:
    """
    构建邻接矩阵

    Args:
        num_nodes: 节点数量
        edges: 边列表

    Returns:
        邻接矩阵
    """
    adj = np.zeros((num_nodes, num_nodes), dtype=np.float32)

    if edges:
        for i, j in edges:
            adj[i, j] = 1
            adj[j, i] = 1

    # 添加自环
    adj += np.eye(num_nodes)

    # 归一化
    degree = np.sum(adj, axis=1)
    degree[degree == 0] = 1  # 避免除零
    degree_inv_sqrt = np.power(degree, -0.5)
    degree_inv_sqrt[np.isinf(degree_inv_sqrt)] = 0
    d_mat_inv_sqrt = np.diag(degree_inv_sqrt)
    adj_normalized = d_mat_inv_sqrt @ adj @ d_mat_inv_sqrt

    return adj_normalized


class STGCNModel(torch.nn.Module):
    """STGCN 时空图卷积网络模型"""

    def __init__(self,
                 in_channels: int = 2,  # x, y 坐标
                 num_classes: int = 8,
                 num_nodes: int = 17,
                 temporal_window: int = 30,
                 hidden_channels: int = 64,
                 num_st_gcn_layers: int = 4):
        """
        初始化 STGCN 模型

        Args:
            in_channels: 输入通道数
            num_classes: 动作类别数
            num_nodes: 骨架节点数
            temporal_window: 时间窗口帧数
            hidden_channels: 隐藏层通道数
            num_st_gcn_layers: ST-GCN 层数
        """
        super(STGCNModel, self).__init__()

        self.num_nodes = num_nodes
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.temporal_window = temporal_window

        # 构建邻接矩阵
        adj = build_adjacency_matrix(num_nodes, COCO_GRAPH_LAYOUT['edges'])
        self.register_buffer('adj', torch.from_numpy(adj))

        # ST-GCN 层
        self.st_gcn_layers = torch.nn.ModuleList()
        current_channels = in_channels

        for i in range(num_st_gcn_layers):
            out_channels = hidden_channels * (2 ** i) if i > 0 else hidden_channels
            self.st_gcn_layers.append(
                STGCNLayer(current_channels, out_channels, self.adj)
            )
            current_channels = out_channels

        # 时间卷积
        self.temporal_conv = torch.nn.Sequential(
            torch.nn.Conv2d(current_channels, current_channels, kernel_size=(9, 1), padding=(4, 0)),
            torch.nn.BatchNorm2d(current_channels),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5)
        )

        # 全局池化
        self.global_pool = torch.nn.AdaptiveAvgPool2d((1, 1))

        # 分类器
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(current_channels, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(256, num_classes)
        )

    def forward(self, x):
        """
        前向传播

        Args:
            x: 输入 (B, C, T, V) - B: batch, C: channels, T: temporal, V: nodes

        Returns:
            分类 logits
        """
        # ST-GCN 层
        for st_gcn_layer in self.st_gcn_layers:
            x = st_gcn_layer(x)

        # 时间卷积 (B, C, T, V) -> (B, C, T, 1)
        x = self.temporal_conv(x)

        # 全局池化 (B, C, 1, 1)
        x = self.global_pool(x)

        # 展平 (B, C)
        x = x.view(x.size(0), -1)

        # 分类
        x = self.classifier(x)

        return x


class STGCNLayer(torch.nn.Module):
    """单层 ST-GCN"""

    def __init__(self, in_channels: int, out_channels: int, adj: torch.Tensor):
        super(STGCNLayer, self).__init__()
        self.adj = adj

        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn = torch.nn.BatchNorm2d(out_channels)

    def forward(self, x):
        """
        Args:
            x: (B, C, T, V)
        """
        # 图卷积: 聚合邻居节点特征
        # x @ adj: (B, C, T, V) @ (V, V) -> (B, C, T, V)
        x = torch.einsum('nctv,vw->nctw', x, self.adj)

        # 空间卷积
        x = self.conv(x)
        x = self.bn(x)
        x = torch.nn.functional.relu(x)

        return x


class ActionRecognizer:
    """行为识别器"""

    def __init__(self,
                 model_path: str = None,
                 num_classes: int = 8,
                 sequence_length: int = 30,
                 action_classes: List[str] = None,
                 device: str = 'cpu'):
        """
        初始化行为识别器

        Args:
            model_path: 模型路径
            num_classes: 动作类别数
            sequence_length: 输入序列长度
            action_classes: 动作类别名称列表
            device: 设备 (cpu/cuda)
        """
        self.num_classes = num_classes
        self.sequence_length = sequence_length
        self.action_classes = action_classes or [
            'stand', 'sit', 'walk', 'run', 'fall', 'bend', 'raise_hands', 'unknown'
        ]
        self.device = device
        self.model = None

        # 关键点序列缓冲区
        self.keypoint_buffer = []
        self.graph = build_adjacency_matrix()

        # 加载模型
        if model_path and Path(model_path).exists():
            self._load_model(model_path)
        else:
            print("⚠ STGCN 模型文件不存在，将使用随机初始化的模型（需训练）")
            self._init_dummy_model()

    def _init_dummy_model(self):
        """初始化一个虚拟模型（用于测试）"""
        try:
            import torch
            self.model = STGCNModel(
                in_channels=2,
                num_classes=self.num_classes,
                num_nodes=17,
                temporal_window=self.sequence_length
            ).to(self.device)
            self.model.eval()
            print(f"✓ 行为识别器初始化完成（虚拟模型，待训练）")
        except ImportError:
            print("⚠ PyTorch 未安装，行为识别功能不可用")

    def _load_model(self, model_path: str):
        """加载训练好的模型"""
        try:
            import torch
            self.model = STGCNModel(
                in_channels=2,
                num_classes=self.num_classes,
                num_nodes=17,
                temporal_window=self.sequence_length
            ).to(self.device)

            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.eval()
            print(f"✓ 行为识别模型加载成功: {model_path}")
        except Exception as e:
            print(f"⚠ 模型加载失败: {e}，使用虚拟模型")
            self._init_dummy_model()

    def add_keypoints(self, keypoints: np.ndarray) -> bool:
        """
        添加关键点到序列缓冲区

        Args:
            keypoints: 关键点坐标 (17, 2)

        Returns:
            缓冲区是否已满
        """
        if keypoints is None or len(keypoints) == 0:
            return False

        # 确保关键点数量正确
        if len(keypoints) < 17:
            # 填充
            padded = np.zeros((17, 2), dtype=np.float32)
            padded[:len(keypoints)] = keypoints[:17]
            keypoints = padded
        elif len(keypoints) > 17:
            keypoints = keypoints[:17]

        self.keypoint_buffer.append(keypoints)

        # 保持缓冲区大小
        if len(self.keypoint_buffer) > self.sequence_length:
            self.keypoint_buffer.pop(0)

        return len(self.keypoint_buffer) >= self.sequence_length

    def recognize(self) -> Dict[str, Any]:
        """
        识别当前动作

        Returns:
            识别结果 {'action': str, 'confidence': float}
        """
        if len(self.keypoint_buffer) < self.sequence_length:
            return {'action': 'unknown', 'confidence': 0.0, 'raw_output': None}

        try:
            import torch

            # 准备输入数据 (T, V, C) -> (C, T, V)
            sequence = np.array(self.keypoint_buffer, dtype=np.float32)  # (T, V, C)

            # 归一化
            sequence = self._normalize_sequence(sequence)

            # 转换为 tensor
            x = torch.from_numpy(sequence).permute(2, 0, 1).unsqueeze(0).to(self.device)  # (1, C, T, V)

            # 推理
            with torch.no_grad():
                logits = self.model(x)
                probs = torch.softmax(logits, dim=1)
                confidence, predicted = torch.max(probs, dim=1)

            action_idx = predicted.item()
            action = self.action_classes[action_idx] if action_idx < len(self.action_classes) else 'unknown'
            conf = confidence.item()

            return {
                'action': action,
                'confidence': conf,
                'raw_output': logits.cpu().numpy()
            }

        except Exception as e:
            print(f"识别错误: {e}")
            return {'action': 'unknown', 'confidence': 0.0, 'raw_output': None}

    def _normalize_sequence(self, sequence: np.ndarray) -> np.ndarray:
        """
        归一化关键点序列

        Args:
            sequence: (T, V, C)

        Returns:
            归一化后的序列
        """
        # 基于第一帧进行归一化（根节点对齐）
        if len(sequence) > 0:
            root = sequence[0, 0, :]  # 鼻子节点作为根
            sequence = sequence - root

        # 缩放到 [-1, 1]
        max_val = np.max(np.abs(sequence))
        if max_val > 0:
            sequence = sequence / max_val

        return sequence.astype(np.float32)

    def reset_buffer(self):
        """重置关键点缓冲区"""
        self.keypoint_buffer = []


class ActionSmoother:
    """动作平滑器 - 减少抖动"""

    def __init__(self, window_size: int = 5):
        """
        初始化平滑器

        Args:
            window_size: 平滑窗口大小
        """
        self.window_size = window_size
        self.history = []

    def smooth(self, action: str, confidence: float) -> Dict[str, Any]:
        """
        平滑动作识别结果

        Args:
            action: 当前动作
            confidence: 当前置信度

        Returns:
            平滑后的结果
        """
        self.history.append({'action': action, 'confidence': confidence})

        if len(self.history) > self.window_size:
            self.history.pop(0)

        if len(self.history) < 3:
            return {'action': action, 'confidence': confidence}

        # 投票选择最常见的动作
        actions = [h['action'] for h in self.history]
        action_counts = {}
        for a in actions:
            action_counts[a] = action_counts.get(a, 0) + 1

        smoothed_action = max(action_counts, key=action_counts.get)

        # 加权平均置信度
        smoothed_confidence = np.mean([h['confidence'] for h in self.history])

        return {
            'action': smoothed_action,
            'confidence': smoothed_confidence
        }

    def reset(self):
        """重置历史"""
        self.history = []


# 延迟导入 torch（如果未安装）
try:
    import torch
except ImportError:
    torch = None
    STGCNModel = object
    STGCNLayer = object
