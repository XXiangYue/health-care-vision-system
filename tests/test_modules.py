"""
单元测试
"""
import pytest
import numpy as np
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestPoseEstimator:
    """姿态估计器测试"""

    def test_keypoints_creation(self):
        """测试关键点数据类创建"""
        from src.pose.estimator import PoseKeypoints

        keypoints = np.random.rand(17, 2).astype(np.float32)
        scores = np.random.rand(17).astype(np.float32)

        pk = PoseKeypoints(keypoints, scores)

        assert pk.num_keypoints == 17
        assert pk.is_valid

    def test_keypoints_normalize(self):
        """测试关键点归一化"""
        from src.pose.estimator import PoseKeypoints

        keypoints = np.array([[100, 200], [150, 250]], dtype=np.float32)
        scores = np.array([0.9, 0.8], dtype=np.float32)

        pk = PoseKeypoints(keypoints, scores)
        normalized = pk.normalize((640, 480))

        assert normalized.keypoints[0, 0] == pytest.approx(0.15625, rel=0.01)
        assert normalized.keypoints[0, 1] == pytest.approx(0.41667, rel=0.01)


class TestActionRecognizer:
    """行为识别器测试"""

    def test_recognizer_init(self):
        """测试识别器初始化"""
        from src.action.recognizer import ActionRecognizer

        recognizer = ActionRecognizer(
            num_classes=8,
            sequence_length=30
        )

        assert recognizer.num_classes == 8
        assert recognizer.sequence_length == 30

    def test_keypoints_buffer(self):
        """测试关键点缓冲区"""
        from src.action.recognizer import ActionRecognizer

        recognizer = ActionRecognizer(sequence_length=10)

        # 添加关键点
        for _ in range(5):
            keypoints = np.random.rand(17, 2).astype(np.float32)
            recognizer.add_keypoints(keypoints)

        assert len(recognizer.keypoint_buffer) == 5

    def test_normalize_sequence(self):
        """测试序列归一化"""
        from src.action.recognizer import ActionRecognizer

        recognizer = ActionRecognizer()

        sequence = np.random.rand(30, 17, 2).astype(np.float32)
        normalized = recognizer._normalize_sequence(sequence)

        assert normalized.shape == (30, 17, 2)
        assert normalized.dtype == np.float32


class TestActionSmoother:
    """动作平滑器测试"""

    def test_smoother_init(self):
        """测试平滑器初始化"""
        from src.action.recognizer import ActionSmoother

        smoother = ActionSmoother(window_size=5)
        assert smoother.window_size == 5

    def test_smooth(self):
        """测试平滑功能"""
        from src.action.recognizer import ActionSmoother

        smoother = ActionSmoother(window_size=3)

        # 连续相同动作
        result = smoother.smooth('sit', 0.9)
        assert result['action'] == 'sit'

        result = smoother.smooth('sit', 0.8)
        assert result['action'] == 'sit'

    def test_smooth_voting(self):
        """测试投票功能"""
        from src.action.recognizer import ActionSmoother

        smoother = ActionSmoother(window_size=5)

        # 混合动作
        smoother.smooth('sit', 0.9)
        smoother.smooth('sit', 0.9)
        smoother.smooth('walk', 0.7)
        smoother.smooth('stand', 0.6)
        smoother.smooth('stand', 0.6)

        result = smoother.smooth('stand', 0.6)
        # 投票应该选择出现最多的
        assert result['action'] == 'stand'


class TestStatsAnalyzer:
    """统计分析器测试"""

    def test_analyzer_init(self):
        """测试分析器初始化"""
        from src.stats.analyzer import StatsAnalyzer

        analyzer = StatsAnalyzer(sedentary_threshold=3600)
        assert analyzer.sedentary_threshold == 3600

    def test_add_action(self):
        """测试添加动作"""
        from src.stats.analyzer import StatsAnalyzer

        analyzer = StatsAnalyzer()

        result = analyzer.add_action('sit', 0.9)
        assert result is not None
        assert analyzer.current_action == 'sit'

    def test_action_change_detection(self):
        """测试动作变化检测"""
        from src.stats.analyzer import StatsAnalyzer

        analyzer = StatsAnalyzer()

        analyzer.add_action('sit', 0.9)
        result = analyzer.add_action('walk', 0.8)

        assert result['type'] == 'action_change'
        assert result['from'] == 'sit'
        assert result['to'] == 'walk'

    def test_daily_stats(self):
        """测试每日统计"""
        from src.stats.analyzer import StatsAnalyzer

        analyzer = StatsAnalyzer()

        # 添加一些动作
        analyzer.add_action('sit', 0.9)
        analyzer.add_action('sit', 0.9)
        analyzer.add_action('walk', 0.8)

        stats = analyzer.get_daily_stats()

        assert stats['total_records'] >= 3
        assert 'action_durations' in stats

    def test_health_suggestions(self):
        """测试健康建议"""
        from src.stats.analyzer import StatsAnalyzer

        analyzer = StatsAnalyzer(sedentary_threshold=1)  # 1秒阈值

        # 长时间久坐
        for _ in range(10):
            analyzer.add_action('sit', 0.9)

        suggestions = analyzer.get_health_suggestions()
        assert len(suggestions) > 0


class TestDatabase:
    """数据库测试"""

    def test_database_init(self, tmp_path):
        """测试数据库初始化"""
        from src.storage.database import Database

        db_path = tmp_path / "test.db"
        db = Database(str(db_path))

        assert db_path.exists()

    def test_insert_action_log(self, tmp_path):
        """测试插入动作日志"""
        from src.storage.database import Database

        db_path = tmp_path / "test.db"
        db = Database(str(db_path))

        log_id = db.insert_action_log('sit', 0.9, duration=10.0)
        assert log_id > 0

    def test_query_logs(self, tmp_path):
        """测试查询日志"""
        from src.storage.database import Database

        db_path = tmp_path / "test.db"
        db = Database(str(db_path))

        db.insert_action_log('sit', 0.9)
        db.insert_action_log('walk', 0.8)

        logs = db.get_action_logs(limit=10)
        assert len(logs) == 2


class TestConfig:
    """配置测试"""

    def test_config_init(self):
        """测试配置初始化"""
        from src.utils.config import Config

        config = Config()

        assert config is not None
        assert config.action_classes is not None

    def test_config_get(self):
        """测试配置获取"""
        from src.utils.config import Config

        config = Config()

        # 测试嵌套键
        confidence = config.get('model.pose.confidence_threshold')
        assert confidence is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
