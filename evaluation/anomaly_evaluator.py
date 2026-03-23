"""
异常检测准确率评估工具
评估指标：准确率、精确度、召回率、F1分数、混淆矩阵
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
import pandas as pd


class AnomalyEvaluator:
    """异常检测评估类"""
    
    def __init__(self, annotations_file, predictions_file):
        """
        初始化评估器
        
        Args:
            annotations_file: 人工标注的JSON文件路径
            predictions_file: 程序输出的JSON文件路径
        """
        self.annotations_file = annotations_file
        self.predictions_file = predictions_file
        
        # 加载数据
        with open(annotations_file, 'r', encoding='utf-8') as f:
            self.ground_truth = json.load(f)
        
        with open(predictions_file, 'r', encoding='utf-8') as f:
            self.predictions = json.load(f)
        
        # 评估结果
        self.tp = 0  # 真正例：正确检测为异常
        self.fp = 0  # 假正例：错误检测为异常
        self.fn = 0  # 假负例：漏检异常
        self.tn = 0  # 真负例：正确检测为正常
        
        self.results = {}
    
    def evaluate(self):
        """执行评估"""
        # 收集所有真值和预测
        true_labels = []
        pred_labels = []
        
        # 转换标注为统一格式
        gt_dict = self._build_gt_dict()
        
        # 逐帧比较
        for frame_data in self.predictions.get('frames', []):
            frame_id = frame_data['frame_id']
            
            for pred_obj in frame_data.get('predictions', []):
                track_id = pred_obj['track_id']
                pred_anomaly = pred_obj['is_anomalous']
                
                # 查找对应的真值
                true_anomaly = self._get_ground_truth(gt_dict, frame_id, track_id)
                
                true_labels.append(true_anomaly)
                pred_labels.append(pred_anomaly)
                
                # 计算混淆矩阵成分
                if true_anomaly == 1 and pred_anomaly == 1:
                    self.tp += 1
                elif true_anomaly == 0 and pred_anomaly == 1:
                    self.fp += 1
                elif true_anomaly == 1 and pred_anomaly == 0:
                    self.fn += 1
                else:
                    self.tn += 1
        
        # 转换为numpy数组
        true_labels = np.array(true_labels)
        pred_labels = np.array(pred_labels)
        
        # 计算评估指标
        self._calculate_metrics(true_labels, pred_labels)
        
        return self.results
    
    def _build_gt_dict(self):
        """构建真值字典，便于快速查询"""
        gt_dict = defaultdict(lambda: defaultdict(lambda: None))
        
        for frame_data in self.ground_truth.get('frames', []):
            frame_id = frame_data['frame_id']
            
            for anomaly in frame_data.get('anomalies', []):
                track_id = anomaly['track_id']
                is_anomalous = 1 if anomaly.get('is_anomalous', False) else 0
                gt_dict[frame_id][track_id] = is_anomalous
        
        return gt_dict
    
    def _get_ground_truth(self, gt_dict, frame_id, track_id):
        """
        获取真值标签
        
        Args:
            gt_dict: 真值字典
            frame_id: 帧编号
            track_id: 追踪ID
            
        Returns:
            1 表示异常，0 表示正常，-1 表示未标注
        """
        label = gt_dict[frame_id].get(track_id, -1)
        return label
    
    def _calculate_metrics(self, true_labels, pred_labels):
        """计算评估指标"""
        total = len(true_labels)
        
        # 基础指标
        accuracy = (self.tp + self.tn) / total if total > 0 else 0
        precision = self.tp / (self.tp + self.fp) if (self.tp + self.fp) > 0 else 0
        recall = self.tp / (self.tp + self.fn) if (self.tp + self.fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # 存储结果
        self.results = {
            '混淆矩阵': {
                'TP (真正例)': self.tp,
                'FP (假正例)': self.fp,
                'FN (假负例)': self.fn,
                'TN (真负例)': self.tn,
            },
            '评估指标': {
                '准确率 (Accuracy)': f"{accuracy:.4f}",
                '精确度 (Precision)': f"{precision:.4f}",
                '召回率 (Recall)': f"{recall:.4f}",
                'F1分数': f"{f1_score:.4f}",
            },
            '数据统计': {
                '总样本数': total,
                '异常样本数': np.sum(true_labels == 1),
                '正常样本数': np.sum(true_labels == 0),
            }
        }
        
        # 计算更详细的混淆矩阵
        cm = confusion_matrix(true_labels, pred_labels)
        self.results['详细混淆矩阵'] = cm.tolist()
        
        # 使用sklearn的分类报告
        report = classification_report(true_labels, pred_labels, 
                                      target_names=['正常', '异常'],
                                      output_dict=False)
        self.results['分类报告'] = report
    
    def print_results(self):
        """打印评估结果"""
        print("\n" + "="*60)
        print("异常检测准确率评估结果")
        print("="*60)
        
        # 混淆矩阵
        print("\n【混淆矩阵】")
        for key, value in self.results['混淆矩阵'].items():
            print(f"  {key}: {value}")
        
        # 评估指标
        print("\n【评估指标】")
        for key, value in self.results['评估指标'].items():
            print(f"  {key}: {value}")
        
        # 数据统计
        print("\n【数据统计】")
        for key, value in self.results['数据统计'].items():
            print(f"  {key}: {value}")
        
        # 分类报告
        print("\n【分类报告】")
        print(self.results['分类报告'])
        
        print("="*60 + "\n")
    
    def save_results(self, output_file):
        """保存评估结果到JSON文件"""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)
        print(f"✓ 评估结果已保存到: {output_file}")
    
    def export_csv(self, output_file):
        """导出为CSV表格"""
        data = []
        
        # 基础指标行
        data.append({
            '指标': '准确率',
            '值': self.results['评估指标']['准确率 (Accuracy)']
        })
        data.append({
            '指标': '精确度',
            '值': self.results['评估指标']['精确度 (Precision)']
        })
        data.append({
            '指标': '召回率',
            '值': self.results['评估指标']['召回率 (Recall)']
        })
        data.append({
            '指标': 'F1分数',
            '值': self.results['评估指标']['F1分数']
        })
        
        df = pd.DataFrame(data)
        df.to_csv(output_file, index=False, encoding='utf-8')
        print(f"✓ CSV结果已导出到: {output_file}")


class FrameAnnotatorHelper:
    """帮助器类：生成程序输出JSON格式"""
    
    @staticmethod
    def create_prediction_json(video_path, tracker_results):
        """
        从追踪器结果创建预测JSON
        
        Args:
            video_path: 视频文件路径
            tracker_results: 追踪结果格式
                {frame_id: [(track_id, x1, y1, x2, y2, conf, class, is_anomalous), ...]}
        
        Returns:
            预测JSON数据
        """
        predictions_json = {
            'video': str(video_path),
            'frames': []
        }
        
        for frame_id, objects in sorted(tracker_results.items()):
            frame_data = {
                'frame_id': frame_id,
                'predictions': []
            }
            
            for obj in objects:
                track_id, x1, y1, x2, y2, conf, class_id, is_anomalous = obj
                
                frame_data['predictions'].append({
                    'track_id': track_id,
                    'class': class_id,
                    'confidence': conf,
                    'bbox': [x1, y1, x2, y2],
                    'is_anomalous': is_anomalous
                })
            
            predictions_json['frames'].append(frame_data)
        
        return predictions_json


# 使用示例
if __name__ == "__main__":
    # 假设已有标注文件和预测文件
    evaluator = AnomalyEvaluator(
        annotations_file='annotations.json',
        predictions_file='predictions.json'
    )
    
    results = evaluator.evaluate()
    evaluator.print_results()
    evaluator.save_results('evaluation_results.json')
    evaluator.export_csv('evaluation_results.csv')
