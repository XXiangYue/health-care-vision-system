"""
数据存储模块
使用 SQLite 持久化存储健康日志
"""
import sqlite3
import json
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from pathlib import Path
import threading


class Database:
    """SQLite 数据库管理类"""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls, db_path: str = None):
        """单例模式"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, db_path: str = None):
        """
        初始化数据库

        Args:
            db_path: 数据库文件路径
        """
        if db_path is not None:
            self.db_path = Path(db_path)
            self._init_connection()
            self._create_tables()

    def _init_connection(self):
        """初始化数据库连接"""
        # 确保目录存在
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        # 启用外键约束
        self.conn.execute("PRAGMA foreign_keys = ON")

    def _create_tables(self):
        """创建数据表"""
        cursor = self.conn.cursor()

        # 动作日志表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS action_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                action TEXT NOT NULL,
                confidence REAL NOT NULL,
                duration REAL,
                metadata TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # 每日统计表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS daily_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL UNIQUE,
                total_records INTEGER DEFAULT 0,
                action_durations TEXT,
                total_active_time REAL DEFAULT 0,
                total_sedentary_time REAL DEFAULT 0,
                action_changes INTEGER DEFAULT 0,
                most_common_action TEXT,
                sedentary_warnings INTEGER DEFAULT 0,
                health_suggestions TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # 摔倒事件表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS fall_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                confidence REAL,
                alert_sent INTEGER DEFAULT 0,
                handled INTEGER DEFAULT 0,
                notes TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # 创建索引
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_action_logs_timestamp
            ON action_logs(timestamp)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_action_logs_action
            ON action_logs(action)
        """)

        self.conn.commit()

    def insert_action_log(self,
                          action: str,
                          confidence: float,
                          duration: float = None,
                          metadata: Dict[str, Any] = None) -> int:
        """
        插入动作日志

        Args:
            action: 动作类别
            confidence: 置信度
            duration: 持续时长（秒）
            metadata: 额外元数据

        Returns:
            记录 ID
        """
        cursor = self.conn.cursor()

        timestamp = datetime.now().isoformat()
        metadata_json = json.dumps(metadata) if metadata else None

        cursor.execute("""
            INSERT INTO action_logs (timestamp, action, confidence, duration, metadata)
            VALUES (?, ?, ?, ?, ?)
        """, (timestamp, action, confidence, duration, metadata_json))

        self.conn.commit()
        return cursor.lastrowid

    def get_action_logs(self,
                        start_time: datetime = None,
                        end_time: datetime = None,
                        action: str = None,
                        limit: int = 100) -> List[Dict[str, Any]]:
        """
        查询动作日志

        Args:
            start_time: 开始时间
            end_time: 结束时间
            action: 动作类别过滤
            limit: 返回数量限制

        Returns:
            日志列表
        """
        cursor = self.conn.cursor()

        query = "SELECT * FROM action_logs WHERE 1=1"
        params = []

        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time.isoformat())

        if end_time:
            query += " AND timestamp <= ?"
            params.append(end_time.isoformat())

        if action:
            query += " AND action = ?"
            params.append(action)

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        cursor.execute(query, params)
        rows = cursor.fetchall()

        return [dict(row) for row in rows]

    def insert_daily_stats(self,
                          stats: Dict[str, Any]) -> int:
        """
        插入或更新每日统计

        Args:
            stats: 统计数据

        Returns:
            记录 ID
        """
        cursor = self.conn.cursor()

        date = stats.get('date', datetime.now().strftime('%Y-%m-%d'))
        action_durations = json.dumps(stats.get('action_durations', {}))
        health_suggestions = json.dumps(stats.get('suggestions', []))

        # 使用 INSERT OR REPLACE
        cursor.execute("""
            INSERT INTO daily_stats (
                date, total_records, action_durations,
                total_active_time, total_sedentary_time,
                action_changes, most_common_action,
                sedentary_warnings, health_suggestions,
                updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(date) DO UPDATE SET
                total_records = excluded.total_records,
                action_durations = excluded.action_durations,
                total_active_time = excluded.total_active_time,
                total_sedentary_time = excluded.total_sedentary_time,
                action_changes = excluded.action_changes,
                most_common_action = excluded.most_common_action,
                sedentary_warnings = excluded.sedentary_warnings,
                health_suggestions = excluded.health_suggestions,
                updated_at = CURRENT_TIMESTAMP
        """, (
            date,
            stats.get('total_records', 0),
            action_durations,
            stats.get('total_active_time', 0),
            stats.get('total_sedentary_time', 0),
            stats.get('action_changes', 0),
            stats.get('most_common_action', 'unknown'),
            stats.get('sedentary_warnings', 0),
            health_suggestions
        ))

        self.conn.commit()
        return cursor.lastrowid

    def get_daily_stats(self, date: str = None) -> Optional[Dict[str, Any]]:
        """
        获取指定日期的统计

        Args:
            date: 日期字符串 (YYYY-MM-DD)，默认今天

        Returns:
            统计数据
        """
        cursor = self.conn.cursor()

        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')

        cursor.execute("""
            SELECT * FROM daily_stats WHERE date = ?
        """, (date,))

        row = cursor.fetchone()
        if row:
            result = dict(row)
            # 解析 JSON 字段
            if result.get('action_durations'):
                result['action_durations'] = json.loads(result['action_durations'])
            if result.get('health_suggestions'):
                result['health_suggestions'] = json.loads(result['health_suggestions'])
            return result

        return None

    def get_weekly_stats(self) -> List[Dict[str, Any]]:
        """
        获取本周统计数据

        Returns:
            统计数据列表
        """
        cursor = self.conn.cursor()

        # 获取本周开始日期
        today = datetime.now()
        week_start = today - timedelta(days=today.weekday())
        week_start_str = week_start.strftime('%Y-%m-%d')

        cursor.execute("""
            SELECT * FROM daily_stats
            WHERE date >= ?
            ORDER BY date ASC
        """, (week_start_str,))

        rows = cursor.fetchall()
        return [dict(row) for row in rows]

    def insert_fall_event(self,
                         confidence: float = None,
                         alert_sent: bool = False,
                         notes: str = None) -> int:
        """
        插入摔倒事件

        Args:
            confidence: 置信度
            alert_sent: 是否已发送警报
            notes: 备注

        Returns:
            事件 ID
        """
        cursor = self.conn.cursor()

        timestamp = datetime.now().isoformat()

        cursor.execute("""
            INSERT INTO fall_events (timestamp, confidence, alert_sent, notes)
            VALUES (?, ?, ?, ?)
        """, (timestamp, confidence, 1 if alert_sent else 0, notes))

        self.conn.commit()
        return cursor.lastrowid

    def get_recent_falls(self, hours: int = 24) -> List[Dict[str, Any]]:
        """
        获取最近的摔倒事件

        Args:
            hours: 小时数

        Returns:
            摔倒事件列表
        """
        cursor = self.conn.cursor()

        cutoff = (datetime.now() - timedelta(hours=hours)).isoformat()

        cursor.execute("""
            SELECT * FROM fall_events
            WHERE timestamp >= ?
            ORDER BY timestamp DESC
        """, (cutoff,))

        rows = cursor.fetchall()
        return [dict(row) for row in rows]

    def mark_fall_handled(self, event_id: int):
        """
        标记摔倒事件已处理

        Args:
            event_id: 事件 ID
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            UPDATE fall_events SET handled = 1 WHERE id = ?
        """, (event_id,))
        self.conn.commit()

    def get_action_summary(self,
                          start_date: str = None,
                          end_date: str = None) -> Dict[str, Any]:
        """
        获取动作统计摘要

        Args:
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)

        Returns:
            摘要统计
        """
        cursor = self.conn.cursor()

        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')

        cursor.execute("""
            SELECT
                action,
                COUNT(*) as count,
                AVG(confidence) as avg_confidence,
                SUM(duration) as total_duration
            FROM action_logs
            WHERE timestamp >= ? AND timestamp <= ?
            GROUP BY action
            ORDER BY count DESC
        """, (start_date, end_date + " 23:59:59"))

        rows = cursor.fetchall()

        return {
            'start_date': start_date,
            'end_date': end_date,
            'summary': [dict(row) for row in rows]
        }

    def close(self):
        """关闭数据库连接"""
        if hasattr(self, 'conn'):
            self.conn.close()

    def __del__(self):
        """析构函数"""
        self.close()
