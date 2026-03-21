# 视觉行为动作检测系统

基于 YOLOv8-Pose 和 STGCN 的实时人体行为动作检测系统，是多模态智慧健康管家系统的视觉模块。

## 功能特性

- **姿态估计**：使用 YOLOv8-Pose 检测人体17个骨骼关键点（COCO格式）
- **行为识别**：基于 STGCN 时空图卷积网络识别动作类别
- **行为统计**：久坐检测、活动量统计、动作变化分析
- **摔倒检测**：自动检测摔倒事件并记录
- **数据存储**：SQLite 持久化健康日志
- **定时分析**：可配置定时任务触发 LLM 健康分析（待实现）

## 技术栈

| 模块 | 技术 |
|------|------|
| 姿态估计 | YOLOv8-Pose (Ultralytics) |
| 行为识别 | STGCN (PyTorch) |
| 视频处理 | OpenCV |
| 数据处理 | NumPy, SciPy |
| 日志存储 | SQLite |
| 定时任务 | APScheduler |

## 项目结构

```
.
├── src/
│   ├── pose/              # 姿态估计模块
│   │   └── estimator.py  # YOLOv8-Pose 关键点检测
│   ├── action/           # 行为识别模块
│   │   └── recognizer.py # STGCN 动作分类
│   ├── stats/            # 行为统计模块
│   │   └── analyzer.py   # 统计分析、久坐检测
│   ├── storage/          # 数据存储模块
│   │   └── database.py   # SQLite 存储
│   ├── core/             # 核心协调模块
│   │   └── detector.py   # 主检测器
│   └── utils/            # 工具模块
│       └── config.py     # 配置管理
├── config/
│   └── config.yaml       # 配置文件
├── tests/
│   └── test_modules.py   # 单元测试
├── models/               # 模型权重目录
├── data/                 # 数据目录
├── logs/                 # 日志目录
├── requirements.txt      # 依赖
├── config.yaml           # 配置文件（复制自config/）
├── train.py             # 训练脚本
├── pytest.ini           # 测试配置
└── README.md            # 项目说明
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 下载模型

YOLOv8-Pose 模型会自动下载。

STGCN 模型需要训练：
```bash
python train.py
```

或下载预训练模型放到 `models/` 目录。

### 3. 修改配置

编辑 `config/config.yaml`：

```yaml
video:
  source: 0  # 摄像头ID，或视频文件路径

model:
  pose:
    model_name: "yolov8n-pose.pt"
```

### 4. 运行

```bash
# 实时摄像头检测
python -m src.core.detector

# 指定视频文件
python -m src.core.detector --video path/to/video.mp4

# 单张图片
python -m src.core.detector --image path/to/image.jpg

# 保存视频
python -m src.core.detector --video 0 --save output.mp4
```

### 5. 运行测试

```bash
pytest
```

## 配置说明

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| `model.pose.model_name` | YOLOv8-Pose 模型 | yolov8n-pose.pt |
| `model.pose.confidence_threshold` | 置信度阈值 | 0.5 |
| `model.action.sequence_length` | 输入帧数 | 30 |
| `stats.sedentary_threshold` | 久坐阈值(秒) | 3600 |
| `database.path` | 数据库路径 | data/health_logs.db |
| `video.source` | 视频源 | 0 |
| `video.frame_skip` | 跳帧数 | 2 |

## 动作类别

- `stand` - 站立
- `sit` - 坐下
- `walk` - 行走
- `run` - 跑步
- `fall` - 摔倒
- `bend` - 弯腰
- `raise_hands` - 举手
- `unknown` - 未知

## 数据存储

SQLite 数据库包含以下表：

- `action_logs` - 动作日志
- `daily_stats` - 每日统计
- `fall_events` - 摔倒事件

## 模块接口

### PoseEstimator

```python
from src.pose.estimator import PoseEstimator, PoseKeypoints

estimator = PoseEstimator()
poses = estimator.estimate(frame)  # 返回 List[PoseKeypoints]
best_pose = estimator.estimate_single(frame)  # 返回 PoseKeypoints
```

### ActionRecognizer

```python
from src.action.recognizer import ActionRecognizer

recognizer = ActionRecognizer(sequence_length=30)
recognizer.add_keypoints(keypoints)  # 添加关键点
result = recognizer.recognize()       # {'action': str, 'confidence': float}
```

### StatsAnalyzer

```python
from src.stats.analyzer import StatsAnalyzer

analyzer = StatsAnalyzer(sedentary_threshold=3600)
analyzer.add_action('sit', 0.9)
stats = analyzer.get_daily_stats()
suggestions = analyzer.get_health_suggestions()
```

### Database

```python
from src.storage.database import Database

db = Database('data/health_logs.db')
db.insert_action_log('sit', 0.9, duration=10.0)
logs = db.get_action_logs(limit=100)
```

## 开发指南

### 添加新的动作类别

1. 修改 `config/config.yaml` 中的 `action_classes`
2. 修改 `config/config.yaml` 中的 `model.action.num_classes`
3. 重新训练 STGCN 模型

### 训练自己的模型

```python
# 使用真实数据集训练
python train.py
```

### 导出 ONNX 模型

（待实现）

## 问题记录

实际开发中的问题和解决方案记录在 [问题记录.md](问题记录.md)

## License

MIT License

## 参考

- [YOLOv8](https://github.com/ultralytics/ultralytics)
- [ST-GCN](https://github.com/yysijie/st-gcn)
- [COCO Keypoints](https://cocodataset.org/#keypoints-2020)
