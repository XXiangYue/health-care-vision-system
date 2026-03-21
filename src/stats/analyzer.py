"""
行为统计模块
统计分析用户行为，包括久坐检测、活动量统计等
"""
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from collections import defaultdict
import json


class ActionRecord:
    """单条动作记录"""

    def __init__(self,
                 action: str,
                 confidence: float,
                 timestamp: datetime = None):
        self.action = action
        self.confidence = confidence
        self.timestamp = timestamp or datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        return {
            'action': self.action,
            'confidence': self.confidence,
            'timestamp': self.timestamp.isoformat()
        }


class StatsAnalyzer:
    """统计分析器"""

    def __init__(self,
                 sedentary_threshold: int = 3600,
                 stat_window: int = 3600,
                 action_classes: List[str] = None):
        """
        初始化统计分析器

        Args:
            sedentary_threshold: 久坐阈值（秒）
            stat_window: 统计时间窗口（秒）
            action_classes: 动作类别列表
        """
        self.sedentary_threshold = sedentary_threshold
        self.stat_window = stat_window
        self.action_classes = action_classes or [
            'stand', 'sit', 'walk', 'run', 'fall', 'bend', 'raise_hands', 'unknown'
        ]

        # 动作记录缓冲区
        self.action_history: List[ActionRecord] = []

        # 当前状态
        self.current_action: Optional[str] = None
        self.current_action_start: Optional[datetime] = None
        self.current_action_duration: float = 0.0

        # 统计缓存
        self._daily_stats_cache: Optional[Dict[str, Any]] = None
        self._last_cache_time: Optional[datetime] = None

    def add_action(self, action: str, confidence: float = 1.0) -> Dict[str, Any]:
        """
        添加动作记录

        Args:
            action: 动作类别
            confidence: 置信度

        Returns:
            状态更新信息
        """
        now = datetime.now()
        record = ActionRecord(action, confidence, now)
        self.action_history.append(record)

        # 状态变化检测
        status_update = self._check_state_change(action, now)

        # 更新当前状态
        if action != self.current_action:
            self.current_action = action
            self.current_action_start = now

        self.current_action_duration = (now - self.current_action_start).total_seconds()

        # 清理过时的记录（保留24小时）
        cutoff = now - timedelta(hours=24)
        self.action_history = [r for r in self.action_history if r.timestamp > cutoff]

        # 清除缓存
        self._daily_stats_cache = None

        return status_update

    def _check_state_change(self, action: str, timestamp: datetime) -> Dict[str, Any]:
        """检查动作状态变化"""
        if not self.current_action or action != self.current_action:
            return {
                'type': 'action_change',
                'from': self.current_action,
                'to': action,
                'timestamp': timestamp.isoformat()
            }
        return {}

    def get_current_state(self) -> Dict[str, Any]:
        """
        获取当前状态

        Returns:
            当前状态信息
        """
        return {
            'current_action': self.current_action,
            'current_action_duration': self.current_action_duration,
            'is_sedentary': (
                self.current_action in ['sit', 'stand'] and
                self.current_action_duration > self.sedentary_threshold
            )
        }

    def get_daily_stats(self) -> Dict[str, Any]:
        """
        获取每日统计

        Returns:
            每日统计数据
        """
        now = datetime.now()
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)

        # 筛选今天的记录
        today_records = [r for r in self.action_history if r.timestamp >= today_start]

        if not today_records:
            return self._empty_stats()

        # 动作时长统计
        action_durations = self._calculate_action_durations(today_records)

        # 计算各项统计
        total_active_time = sum(
            d for action, d in action_durations.items()
            if action in ['walk', 'run', 'bend', 'raise_hands']
        )

        total_sedentary_time = sum(
            d for action, d in action_durations.items()
            if action in ['sit', 'stand']
        )

        # 动作变化次数
        action_changes = self._count_action_changes(today_records)

        # 最常见动作
        most_common_action = max(action_durations, key=action_durations.get) if action_durations else 'unknown'

        stats = {
            'date': today_start.strftime('%Y-%m-%d'),
            'total_records': len(today_records),
            'action_durations': action_durations,
            'total_active_time': total_active_time,
            'total_sedentary_time': total_sedentary_time,
            'action_changes': action_changes,
            'most_common_action': most_common_action,
            'sedentary_warnings': self._count_sedentary_warnings(today_records)
        }

        return stats

    def _calculate_action_durations(self, records: List[ActionRecord]) -> Dict[str, float]:
        """计算各动作持续时长"""
        if not records:
            return {}

        # 按动作分组，计算持续时间
        action_durations = defaultdict(float)
        current_action = records[0].action
        current_start = records[0].timestamp

        for i in range(1, len(records)):
            if records[i].action != current_action:
                duration = (records[i].timestamp - current_start).total_seconds()
                action_durations[current_action] += duration
                current_action = records[i].action
                current_start = records[i].timestamp

        # 最后一段
        if records:
            duration = (records[-1].timestamp - current_start).total_seconds()
            action_durations[current_action] += duration

        return dict(action_durations)

    def _count_action_changes(self, records: List[ActionRecord]) -> int:
        """统计动作变化次数"""
        if len(records) < 2:
            return 0

        changes = 0
        for i in range(1, len(records)):
            if records[i].action != records[i-1].action:
                changes += 1

        return changes

    def _count_sedentary_warnings(self, records: List[ActionRecord]) -> int:
        """统计久坐警告次数"""
        warnings = 0
        current_action = None
        action_start = None

        for record in records:
            if record.action != current_action:
                if current_action in ['sit', 'stand'] and action_start:
                    duration = (record.timestamp - action_start).total_seconds()
                    if duration >= self.sedentary_threshold:
                        warnings += 1
                current_action = record.action
                action_start = record.timestamp

        return warnings

    def _empty_stats(self) -> Dict[str, Any]:
        """返回空统计数据"""
        return {
            'date': datetime.now().strftime('%Y-%m-%d'),
            'total_records': 0,
            'action_durations': {},
            'total_active_time': 0,
            'total_sedentary_time': 0,
            'action_changes': 0,
            'most_common_action': 'unknown',
            'sedentary_warnings': 0
        }

    def check_fall(self) -> bool:
        """
        检测是否发生摔倒

        Returns:
            是否检测到摔倒
        """
        if len(self.action_history) < 3:
            return False

        # 检查最近的动作序列
        recent = self.action_history[-3:]
        actions = [r.action for r in recent]

        # 模式：站立/行走 -> 摔倒
        if len(actions) >= 2:
            if actions[-1] == 'fall' and actions[-2] in ['walk', 'stand', 'run']:
                return True

        return False

    def get_activity_level(self) -> str:
        """
        获取活动水平评估

        Returns:
            活动水平: 'low', 'moderate', 'high'
        """
        stats = self.get_daily_stats()

        active_time = stats.get('total_active_time', 0)
        sedentary_time = stats.get('total_sedentary_time', 0)
        total_time = active_time + sedentary_time

        if total_time == 0:
            return 'unknown'

        activity_ratio = active_time / total_time

        if activity_ratio < 0.1:
            return 'low'
        elif activity_ratio < 0.3:
            return 'moderate'
        else:
            return 'high'

    def get_health_suggestions(self) -> List[str]:
        """
        获取健康建议

        Returns:
            建议列表
        """
        stats = self.get_daily_stats()
        suggestions = []

        # 久坐建议
        sedentary_time = stats.get('total_sedentary_time', 0)
        sedentary_warnings = stats.get('sedentary_warnings', 0)

        if sedentary_time > 4 * 3600:
            suggestions.append("您今天的久坐时间较长，建议每小时起身活动5-10分钟")
        elif sedentary_time > 2 * 3600:
            suggestions.append("注意适当活动，避免长时间久坐")

        if sedentary_warnings >= 2:
            suggestions.append(f"今日已触发{sedentary_warnings}次久坐提醒，请注意休息")

        # 活动量建议
        activity_level = self.get_activity_level()
        if activity_level == 'low':
            suggestions.append("活动量偏低，建议增加户外活动或运动")
        elif activity_level == 'high':
            suggestions.append("活动量充足，继续保持！")

        # 摔倒风险（老年人）
        if stats.get('most_common_action') == 'fall':
            suggestions.append("检测到摔倒动作，建议检查是否需要帮助")

        return suggestions

    def reset(self):
        """重置统计"""
        self.action_history = []
        self.current_action = None
        self.current_action_start = None
        self.current_action_duration = 0.0
        self._daily_stats_cache = None

    def export_data(self) -> str:
        """
        导出统计数据为 JSON 字符串

        Returns:
            JSON 字符串
        """
        stats = self.get_daily_stats()
        stats['suggestions'] = self.get_health_suggestions()
        return json.dumps(stats, ensure_ascii=False, indent=2)
