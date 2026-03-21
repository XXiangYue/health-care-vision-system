"""
配置加载工具
"""
import os
import yaml
from pathlib import Path
from typing import Any, Dict


class Config:
    """配置管理类"""

    def __init__(self, config_path: str = None):
        """
        初始化配置

        Args:
            config_path: 配置文件路径
        """
        if config_path is None:
            # 默认使用项目根目录下的 config/config.yaml
            base_dir = Path(__file__).parent.parent
            config_path = base_dir / "config" / "config.yaml"

        self.config_path = Path(config_path)
        self._config: Dict[str, Any] = {}
        self._load()

    def _load(self):
        """加载配置文件"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {self.config_path}")

        with open(self.config_path, 'r', encoding='utf-8') as f:
            self._config = yaml.safe_load(f)

    def get(self, key: str, default: Any = None) -> Any:
        """
        获取配置项

        Args:
            key: 配置键，支持点号分隔的嵌套键，如 "model.pose.model_name"
            default: 默认值

        Returns:
            配置值
        """
        keys = key.split('.')
        value = self._config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    @property
    def model(self) -> Dict[str, Any]:
        """模型配置"""
        return self._config.get('model', {})

    @property
    def pose_config(self) -> Dict[str, Any]:
        """姿态估计配置"""
        return self.model.get('pose', {})

    @property
    def action_config(self) -> Dict[str, Any]:
        """行为识别配置"""
        return self.model.get('action', {})

    @property
    def action_classes(self):
        """动作类别列表"""
        return self._config.get('action_classes', [])

    @property
    def stats_config(self) -> Dict[str, Any]:
        """统计配置"""
        return self._config.get('stats', {})

    @property
    def database_config(self) -> Dict[str, Any]:
        """数据库配置"""
        return self._config.get('database', {})

    @property
    def video_config(self) -> Dict[str, Any]:
        """视频配置"""
        return self._config.get('video', {})

    @property
    def performance_config(self) -> Dict[str, Any]:
        """性能配置"""
        return self._config.get('performance', {})

    def reload(self):
        """重新加载配置"""
        self._load()


# 全局配置实例
_config_instance = None


def get_config(config_path: str = None) -> Config:
    """
    获取配置单例

    Args:
        config_path: 配置文件路径

    Returns:
        Config 实例
    """
    global _config_instance
    if _config_instance is None:
        _config_instance = Config(config_path)
    return _config_instance
