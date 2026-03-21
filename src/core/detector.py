"""
核心检测器模块
整合姿态估计、行为识别、统计分析模块
"""
import cv2
import numpy as np
import argparse
from typing import Optional, Dict, Any, List
from pathlib import Path
from datetime import datetime
import logging

# 导入各模块
from ..pose.estimator import PoseEstimator, PoseVisualizer
from ..action.recognizer import ActionRecognizer, ActionSmoother
from ..stats.analyzer import StatsAnalyzer
from ..storage.database import Database
from ..utils.config import get_config


class ActionDetector:
    """行为动作检测器 - 整合所有模块"""

    def __init__(self, config_path: str = None, video_source: int = None):
        """
        初始化检测器

        Args:
            config_path: 配置文件路径
            video_source: 视频源 (摄像头ID或视频文件路径)
        """
        # 加载配置
        self.config = get_config(config_path)

        # 视频源配置
        if video_source is not None:
            self.video_source = video_source
        else:
            self.video_source = self.config.get('video.source', 0)

        # 初始化各模块
        self._init_pose_estimator()
        self._init_action_recognizer()
        self._init_stats_analyzer()
        self._init_database()
        self._init_visualizer()

        # 动作平滑器
        self.action_smoother = ActionSmoother(
            window_size=self.config.get('stats.smoothing_window', 5)
        )

        # 日志
        self._setup_logging()

        # 状态
        self.is_running = False
        self.frame_count = 0

    def _init_pose_estimator(self):
        """初始化姿态估计器"""
        pose_config = self.config.pose_config
        self.pose_estimator = PoseEstimator(
            model_name=pose_config.get('model_name', 'yolov8n-pose.pt'),
            confidence_threshold=pose_config.get('confidence_threshold', 0.5),
            nms_threshold=pose_config.get('nms_threshold', 0.4)
        )

    def _init_action_recognizer(self):
        """初始化行为识别器"""
        action_config = self.config.action_config
        self.action_recognizer = ActionRecognizer(
            model_path=action_config.get('model_path'),
            num_classes=action_config.get('num_classes', 8),
            sequence_length=action_config.get('sequence_length', 30),
            action_classes=self.config.action_classes
        )

    def _init_stats_analyzer(self):
        """初始化统计分析器"""
        stats_config = self.config.stats_config
        self.stats_analyzer = StatsAnalyzer(
            sedentary_threshold=stats_config.get('sedentary_threshold', 3600),
            stat_window=stats_config.get('stat_window', 3600),
            action_classes=self.config.action_classes
        )

    def _init_database(self):
        """初始化数据库"""
        db_config = self.config.database_config
        db_path = db_config.get('path', 'data/health_logs.db')

        # 转换为绝对路径
        if not Path(db_path).is_absolute():
            base_dir = Path(__file__).parent.parent.parent
            db_path = base_dir / db_path

        self.database = Database(str(db_path))
        print(f"✓ 数据库初始化成功: {db_path}")

    def _init_visualizer(self):
        """初始化可视化工具"""
        self.visualizer = PoseVisualizer()

    def _setup_logging(self):
        """设置日志"""
        log_config = self.config.get('logging', {})
        log_level = log_config.get('level', 'INFO')

        logging.basicConfig(
            level=getattr(logging, log_level),
            format=log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
        self.logger = logging.getLogger(__name__)

    def process_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        处理单帧图像

        Args:
            frame: 输入图像

        Returns:
            处理结果
        """
        self.frame_count += 1

        result = {
            'frame': frame,
            'pose_detected': False,
            'keypoints': None,
            'action': 'unknown',
            'action_confidence': 0.0,
            'smoothed_action': None,
            'smoothed_confidence': 0.0,
            'stats': {},
            'fall_detected': False
        }

        # 1. 姿态估计
        poses = self.pose_estimator.estimate(frame)

        if poses:
            result['pose_detected'] = True

            # 选择置信度最高的人
            best_pose = max(poses, key=lambda p: np.mean(p.scores))
            result['keypoints'] = best_pose

            # 2. 添加关键点到行为识别器
            is_ready = self.action_recognizer.add_keypoints(best_pose.keypoints)

            if is_ready:
                # 3. 行为识别
                recognition = self.action_recognizer.recognize()
                result['action'] = recognition.get('action', 'unknown')
                result['action_confidence'] = recognition.get('confidence', 0.0)

                # 4. 平滑处理
                smoothed = self.action_smoother.smooth(
                    result['action'],
                    result['action_confidence']
                )
                result['smoothed_action'] = smoothed['action']
                result['smoothed_confidence'] = smoothed['confidence']

                # 5. 统计分析
                self.stats_analyzer.add_action(
                    result['smoothed_action'],
                    result['smoothed_confidence']
                )

                # 6. 摔倒检测
                if result['smoothed_action'] == 'fall':
                    if self.stats_analyzer.check_fall():
                        result['fall_detected'] = True
                        self._handle_fall_event(result)

                # 7. 存储到数据库
                self._save_to_database(result)

        # 获取当前统计
        result['stats'] = self.stats_analyzer.get_current_state()

        return result

    def _save_to_database(self, result: Dict[str, Any]):
        """保存结果到数据库"""
        try:
            # 保存动作日志
            self.database.insert_action_log(
                action=result.get('smoothed_action', 'unknown'),
                confidence=result.get('smoothed_confidence', 0.0),
                duration=result['stats'].get('current_action_duration', 0.0),
                metadata={'frame_count': self.frame_count}
            )
        except Exception as e:
            self.logger.error(f"数据库写入失败: {e}")

    def _handle_fall_event(self, result: Dict[str, Any]):
        """处理摔倒事件"""
        self.logger.warning("⚠ 检测到摔倒！")

        # 保存摔倒事件
        try:
            self.database.insert_fall_event(
                confidence=result.get('action_confidence', 0.0),
                alert_sent=False,
                notes=f"检测到摔倒动作，帧号: {self.frame_count}"
            )
        except Exception as e:
            self.logger.error(f"摔倒事件记录失败: {e}")

        # TODO: 发送警报通知

    def draw_result(self, frame: np.ndarray, result: Dict[str, Any]) -> np.ndarray:
        """
        在帧上绘制结果

        Args:
            frame: 输入帧
            result: 处理结果

        Returns:
            绘制后的帧
        """
        output = frame.copy()

        # 绘制骨架（如果检测到姿态）
        if result.get('keypoints'):
            output = self.visualizer.draw_pose(output, result['keypoints'])

        # 绘制动作标签
        if result.get('smoothed_action'):
            output = self.visualizer.draw_action_label(
                output,
                result['smoothed_action'],
                result['smoothed_confidence']
            )

        # 绘制统计信息
        stats = result.get('stats', {})
        if stats:
            info_lines = [
                f"当前动作: {stats.get('current_action', 'unknown')}",
                f"持续时间: {stats.get('current_action_duration', 0):.1f}秒"
            ]

            if stats.get('is_sedentary'):
                info_lines.append("⚠ 久坐提醒!")

            y_offset = 60
            for line in info_lines:
                cv2.putText(output, line, (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                y_offset += 25

        # 绘制摔倒警告
        if result.get('fall_detected'):
            cv2.putText(output, "⚠ 摔 倒 警 告!", (frame.shape[1]//2 - 100, frame.shape[0]//2),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

        return output

    def run(self, display: bool = True, save_path: str = None):
        """
        运行检测

        Args:
            display: 是否显示视频
            save_path: 保存视频路径
        """
        # 打开视频源
        if isinstance(self.video_source, int):
            cap = cv2.VideoCapture(self.video_source)
            print(f"✓ 打开摄像头: {self.video_source}")
        else:
            cap = cv2.VideoCapture(str(self.video_source))
            print(f"✓ 打开视频: {self.video_source}")

        if not cap.isOpened():
            raise RuntimeError(f"无法打开视频源: {self.video_source}")

        # 获取视频属性
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # 视频写入器
        writer = None
        if save_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(save_path, fourcc, fps, (width, height))
            print(f"✓ 视频保存: {save_path}")

        # 跳帧配置
        frame_skip = self.config.get('video.frame_skip', 2)
        frame_idx = 0

        self.is_running = True
        print("=" * 50)
        print("开始行为检测 (按 'q' 退出)")
        print("=" * 50)

        try:
            while self.is_running:
                ret, frame = cap.read()
                if not ret:
                    print("视频结束或读取失败")
                    break

                frame_idx += 1

                # 跳帧处理
                if frame_idx % (frame_skip + 1) != 0:
                    if display:
                        cv2.imshow('Action Detection', frame)
                    if writer:
                        writer.write(frame)
                    continue

                # 处理帧
                result = self.process_frame(frame)

                # 绘制结果
                output = self.draw_result(result, frame)

                # 显示
                if display:
                    cv2.imshow('Action Detection', output)

                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("用户退出")
                        break

                # 保存
                if writer:
                    writer.write(output)

        except KeyboardInterrupt:
            print("\n用户中断")
        finally:
            self.is_running = False
            cap.release()
            if writer:
                writer.release()
            if display:
                cv2.destroyAllWindows()

            # 打印统计摘要
            self._print_summary()

    def run_on_image(self, image_path: str) -> Dict[str, Any]:
        """
        在单张图片上运行检测

        Args:
            image_path: 图片路径

        Returns:
            检测结果
        """
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))

        frame = cv2.imread(image_path)
        if frame is None:
            raise ValueError(f"无法读取图片: {image_path}")

        result = self.process_frame(frame)
        output = self.draw_result(result, frame)

        # 保存结果
        output_path = Path(image_path).stem + '_result.jpg'
        cv2.imwrite(output_path, output)
        print(f"✓ 结果已保存: {output_path}")

        return result

    def _print_summary(self):
        """打印统计摘要"""
        print("\n" + "=" * 50)
        print("检测统计摘要")
        print("=" * 50)

        stats = self.stats_analyzer.get_daily_stats()
        print(f"处理帧数: {self.frame_count}")
        print(f"今日动作记录: {stats.get('total_records', 0)}")
        print(f"最常见动作: {stats.get('most_common_action', 'unknown')}")
        print(f"总活动时间: {stats.get('total_active_time', 0)/3600:.2f} 小时")
        print(f"总久坐时间: {stats.get('total_sedentary_time', 0)/3600:.2f} 小时")

        suggestions = self.stats_analyzer.get_health_suggestions()
        if suggestions:
            print("\n健康建议:")
            for s in suggestions:
                print(f"  - {s}")

    def get_stats(self) -> Dict[str, Any]:
        """获取当前统计"""
        return {
            'daily': self.stats_analyzer.get_daily_stats(),
            'current': self.stats_analyzer.get_current_state(),
            'activity_level': self.stats_analyzer.get_activity_level(),
            'suggestions': self.stats_analyzer.get_health_suggestions()
        }

    def stop(self):
        """停止检测"""
        self.is_running = False


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='视觉行为动作检测系统')
    parser.add_argument('--config', type=str, help='配置文件路径')
    parser.add_argument('--video', type=str, help='视频源 (摄像头ID或视频文件)')
    parser.add_argument('--image', type=str, help='单张图片路径')
    parser.add_argument('--no-display', action='store_true', help='不显示视频')
    parser.add_argument('--save', type=str, help='保存视频路径')

    args = parser.parse_args()

    # 确定视频源
    video_source = args.video
    if video_source is not None:
        try:
            video_source = int(video_source)
        except ValueError:
            pass  # 保持字符串（文件路径）

    # 创建检测器
    detector = ActionDetector(
        config_path=args.config,
        video_source=video_source
    )

    # 运行
    if args.image:
        result = detector.run_on_image(args.image)
        print(f"\n检测结果:")
        print(f"  动作: {result.get('smoothed_action', 'unknown')}")
        print(f"  置信度: {result.get('smoothed_confidence', 0):.2f}")
    else:
        detector.run(
            display=not args.no_display,
            save_path=args.save
        )


if __name__ == '__main__':
    main()
