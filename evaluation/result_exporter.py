"""
追踪结果导出模块
支持将异常检测结果导出为JSON格式，用于准确率评估
"""

import json
from datetime import datetime
from pathlib import Path
from collections import defaultdict


class TrackingResultExporter:
    """追踪和异常检测结果导出器"""
    
    def __init__(self, output_dir='evaluation'):
        """
        初始化导出器
        
        Args:
            output_dir: 输出目录
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # 按帧存储结果
        self.frame_results = defaultdict(list)
        self.video_info = {}
        self.frame_count = 0
    
    def add_frame_result(self, frame_id, track_id, bbox, class_name, 
                        confidence, is_anomalous, reason=''):
        """
        添加单帧的追踪结果
        
        Args:
            frame_id: 帧编号
            track_id: 追踪ID
            bbox: [x1, y1, x2, y2]
            class_name: 类别名称
            confidence: 检测置信度
            is_anomalous: 是否异常
            reason: 异常原因
        """
        result = {
            'track_id': int(track_id),
            'class': class_name,
            'confidence': float(confidence),
            'bbox': [float(x) for x in bbox],
            'is_anomalous': bool(is_anomalous),
            'reason': reason
        }
        self.frame_results[frame_id].append(result)
        self.frame_count = max(self.frame_count, frame_id)
    
    def set_video_info(self, video_path, fps, resolution=None):
        """
        设置视频信息
        
        Args:
            video_path: 视频路径
            fps: 帧率
            resolution: 分辨率 (width, height)
        """
        self.video_info = {
            'video': str(video_path),
            'fps': fps,
            'total_frames': self.frame_count,
            'resolution': resolution,
            'export_time': datetime.now().isoformat()
        }
    
    def export_json(self, filename='predictions.json'):
        """导出为JSON格式"""
        output_data = {
            'metadata': self.video_info,
            'frames': []
        }
        
        # 按帧编号排序
        for frame_id in sorted(self.frame_results.keys()):
            frame_data = {
                'frame_id': frame_id,
                'num_objects': len(self.frame_results[frame_id]),
                'num_anomalies': sum(
                    1 for obj in self.frame_results[frame_id] 
                    if obj['is_anomalous']
                ),
                'predictions': self.frame_results[frame_id]
            }
            output_data['frames'].append(frame_data)
        
        output_path = self.output_dir / filename
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        print(f"✓ 预测结果已导出到: {output_path}")
        return output_path
    
    def export_statistics(self, filename='statistics.json'):
        """导出统计信息"""
        stats = {
            '总帧数': self.frame_count,
            '检测对象总数': sum(len(objs) for objs in self.frame_results.values()),
            '异常对象总数': sum(
                sum(1 for obj in objs if obj['is_anomalous'])
                for objs in self.frame_results.values()
            ),
            '平均每帧对象数': sum(len(objs) for objs in self.frame_results.values()) / max(1, self.frame_count),
            '类别统计': self._get_class_stats(),
            '异常类型统计': self._get_anomaly_stats()
        }
        
        output_path = self.output_dir / filename
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        
        print(f"✓ 统计信息已导出到: {output_path}")
        return output_path
    
    def _get_class_stats(self):
        """获取类别统计"""
        class_counts = defaultdict(int)
        class_anomalies = defaultdict(int)
        
        for objs in self.frame_results.values():
            for obj in objs:
                class_counts[obj['class']] += 1
                if obj['is_anomalous']:
                    class_anomalies[obj['class']] += 1
        
        return {
            cls: {
                '总数': count,
                '异常数': class_anomalies[cls],
                '异常率': f"{100*class_anomalies[cls]/count:.2f}%" if count > 0 else "0%"
            }
            for cls, count in class_counts.items()
        }
    
    def _get_anomaly_stats(self):
        """获取异常类型统计"""
        reason_counts = defaultdict(int)
        
        for objs in self.frame_results.values():
            for obj in objs:
                if obj['is_anomalous']:
                    reason = obj.get('reason', '未知原因')
                    reason_counts[reason] += 1
        
        return dict(reason_counts)
    
    def export_summary(self, filename='summary.txt'):
        """导出文本格式的总结"""
        summary = []
        summary.append("=" * 60)
        summary.append("追踪和异常检测结果总结")
        summary.append("=" * 60)
        summary.append(f"\n视频信息:")
        summary.append(f"  视频: {self.video_info.get('video', 'N/A')}")
        summary.append(f"  帧率: {self.video_info.get('fps', 'N/A')} fps")
        summary.append(f"  总帧数: {self.frame_count}")
        summary.append(f"  分辨率: {self.video_info.get('resolution', 'N/A')}")
        
        total_objects = sum(len(objs) for objs in self.frame_results.values())
        total_anomalies = sum(
            sum(1 for obj in objs if obj['is_anomalous'])
            for objs in self.frame_results.values()
        )
        
        summary.append(f"\n检测统计:")
        summary.append(f"  总检测对象数: {total_objects}")
        summary.append(f"  异常对象数: {total_anomalies}")
        summary.append(f"  异常率: {100*total_anomalies/max(1, total_objects):.2f}%")
        
        summary.append(f"\n类别分布:")
        class_stats = self._get_class_stats()
        for cls, stats in class_stats.items():
            summary.append(f"  {cls}: {stats['总数']}个 (异常: {stats['异常数']}个, 异常率: {stats['异常率']})")
        
        summary.append(f"\n异常类型:")
        anomaly_stats = self._get_anomaly_stats()
        for reason, count in anomaly_stats.items():
            summary.append(f"  {reason}: {count}")
        
        summary.append("\n" + "=" * 60)
        
        output_path = self.output_dir / filename
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(summary))
        
        print(f"✓ 总结已导出到: {output_path}")
        
        # 同时打印到控制台
        print('\n'.join(summary))
        
        return output_path


# 使用示例
if __name__ == "__main__":
    # 创建导出器
    exporter = TrackingResultExporter()
    
    # 设置视频信息
    exporter.set_video_info(
        video_path='test.mp4',
        fps=30,
        resolution=(1920, 1080)
    )
    
    # 添加示例结果（模拟）
    for frame_id in range(1, 31):
        if frame_id <= 15:
            # 前15帧
            exporter.add_frame_result(
                frame_id=frame_id,
                track_id=1,
                bbox=[100, 100, 200, 200],
                class_name='car',
                confidence=0.95,
                is_anomalous=False,
                reason=''
            )
        else:
            # 后15帧（异常）
            exporter.add_frame_result(
                frame_id=frame_id,
                track_id=1,
                bbox=[300, 100, 400, 200],
                class_name='car',
                confidence=0.95,
                is_anomalous=True,
                reason='速度过快 (突跳像素数>100)'
            )
    
    # 导出结果
    exporter.export_json()
    exporter.export_statistics()
    exporter.export_summary()
