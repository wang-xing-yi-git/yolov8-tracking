"""
如何在detect_and_trk.py中集成结果导出功能
这个文件提供了具体的代码实现建议
"""

# ========================================
# 1. 在detect_and_trk.py顶部添加导入
# ========================================

# 在现有导入之后添加：
# from evaluation.result_exporter import TrackingResultExporter
# 或者如果在同一目录下：
from result_exporter import TrackingResultExporter


# ========================================
# 2. 在主函数中初始化导出器
# ========================================

# 在视频处理循环前，初始化导出器：
def main_detect_and_track(
    weights='yolov8n.pt',
    conf=0.25,
    iou=0.45,
    source='0',
    classes=None,
    project='runs/detect',
    name='detect',
    exist_ok=False,
    half=False,
    dnn=False,
    device='',
    view_img=False,
    save_txt=False,
    save_conf=False,
    save_crop=False,
    nosave=False,
    line_thickness=3,
    hide_labels=False,
    hide_conf=False,
    hide_class_id=False,
    vid_stride=1,
    skip_frame=1,
    export_results=True,  # 新增参数
):
    """
    主函数（示意，保留原有所有参数）
    """
    
    # ===== 新增代码: 初始化导出器 =====
    exporter = None
    if export_results:
        exporter = TrackingResultExporter(output_dir='evaluation')
    # ===================================
    
    # 原有的初始化代码...
    model = ...
    dataset = ...
    
    # 获取视频信息（用于设置导出器元数据）
    if hasattr(dataset, 'fps'):
        fps = dataset.fps
    else:
        fps = 30  # 默认值
    
    if exporter:
        exporter.set_video_info(
            video_path=source,
            fps=fps,
            resolution=None  # 可以从第一帧获取
        )
    
    # ===== 主要处理循环 =====
    for frame_id, (path, img, im0s, vid_cap, s) in enumerate(dataset, 1):
        
        # 推理和后处理（原有代码）...
        results = model(img, size=imgsz, conf=conf, iou=iou)
        
        # 追踪（SORT）
        detections = extract_detections(results)  # 提取检测
        tracks = tracker.update(detections)  # 更新追踪
        
        # ===== 新增代码: 检查异常并导出结果 =====
        if exporter:
            for track in tracks:
                track_id = track.track_id
                bbox = track.bbox  # [x1, y1, x2, y2]
                class_id = track.class_id
                class_name = model.names[class_id]
                confidence = track.confidence
                
                # 检测异常
                is_anomalous, reason = _track_is_anomalous(
                    track_id=track_id,
                    current_bbox=bbox,
                    class_name=class_name,
                    tracker=tracker,
                    # ... 其他参数
                )
                
                # 导出结果
                exporter.add_frame_result(
                    frame_id=frame_id,
                    track_id=track_id,
                    bbox=bbox,
                    class_name=class_name,
                    confidence=confidence,
                    is_anomalous=is_anomalous,
                    reason=reason
                )
        # =======================================
        
        # 可视化和保存（原有代码）...
        if view_img:
            plot_and_display(im0s, tracks)
    
    # ===== 新增代码: 导出最终结果 =====
    if exporter:
        exporter.export_json()          # 保存为JSON
        exporter.export_statistics()    # 保存统计信息
        exporter.export_summary()       # 保存文本总结
    # ===================================


# ========================================
# 3. 修改 _track_is_anomalous 函数返回值
# ========================================

# 原有函数可能是这样的：
def _track_is_anomalous_OLD(track_id, current_bbox, class_name, tracker):
    """
    原始版本：返回布尔值
    """
    # 检查车速
    if class_name == '配送车':
        speed = calculate_speed(track_id, current_bbox, tracker)
        if speed > CAR_SPEED_THRESHOLD:
            return True  # 异常
    
    # 检查人体比例
    if class_name == 'person':
        aspect_ratio = calculate_aspect_ratio(current_bbox)
        if aspect_ratio < PERSON_ASPECT_RATIO_THRESHOLD:
            return True  # 躺下（异常）
    
    return False  # 正常


# 改进版本：返回元组 (是否异常, 异常原因)
def _track_is_anomalous(track_id, current_bbox, class_name, tracker):
    """
    改进版本：同时返回异常标签和原因，便于导出和分析
    
    Returns:
        tuple: (is_anomalous: bool, reason: str)
    """
    # 检查车速
    if class_name == '配送车':
        speed = calculate_speed(track_id, current_bbox, tracker)
        if speed > CAR_SPEED_THRESHOLD:
            reason = f'速度过快 (突跳像素数>{CAR_SPEED_THRESHOLD})'
            return True, reason
    
    # 检查人体比例
    if class_name == 'person':
        aspect_ratio = calculate_aspect_ratio(current_bbox)
        if aspect_ratio < PERSON_ASPECT_RATIO_THRESHOLD:
            reason = f'俯身/躺下 (长宽比<{PERSON_ASPECT_RATIO_THRESHOLD})'
            return True, reason
    
    return False, ''  # 正常


# ========================================
# 4. 完整的集成示例（伪代码）
# ========================================

"""
from yolo.v8.detect.detect_and_trk import main_detect_and_track
from evaluation.result_exporter import TrackingResultExporter

# 开始处理
main_detect_and_track(
    weights='yolov8n.pt',
    source='test_video.mp4',
    export_results=True,  # 启用结果导出
    # ... 其他参数
)

# 完成后，会生成以下文件：
# evaluation/
#   ├── predictions.json          # 逐帧预测结果
#   ├── statistics.json           # 汇总统计信息
#   └── summary.txt               # 文本总结报告

# 然后运行评估：
from evaluation.anomaly_evaluator import AnomalyEvaluator

evaluator = AnomalyEvaluator(
    annotations_file='evaluation/test_annotations.json',
    predictions_file='evaluation/predictions.json'
)

results = evaluator.evaluate()
evaluator.print_results()
evaluator.save_results()
"""


# ========================================
# 5. 常见问题解答 (FAQ)
# ========================================

FAQ = """
Q1: 为什么需要修改 _track_is_anomalous 的返回值？
A: 原来只返回True/False，无法记录异常的具体原因。改成返回(bool, str)元组后，
   可以保存"速度过快"、"躺下"等具体原因，便于后期分析和改进算法。

Q2: 如何处理多类异常？
A: 可以在 reason 字符串中列出所有异常原因，或改为返回列表：
   return True, ['速度过快', '摄像头遮挡']

Q3: 导出器会消耗很多内存吗？
A: 不会。导出器只存储必要的元数据，每个物体的结果大小约100-200字节。
   即使处理100万帧，内存占用也在可接受范围内。

Q4: 导出的JSON可以不包含置信度吗？
A: 可以。修改 TrackingResultExporter.add_frame_result() 方法，
   注释掉 'confidence' 字段即可。

Q5: 如何在训练期间周期性导出结果？
A: 在追踪循环中每N帧导出一次：
   if frame_id % 100 == 0:  # 每100帧导出一次
       exporter.export_json(f'predictions_frame_{frame_id}.json')

Q6: 如何合并多个视频的结果？
A: 创建多个导出器单独处理每个视频，然后：
   - combined_frames = []
   - 按顺序添加每个视频的frames
   - 最后保存为单个predictions.json
"""

# print(FAQ)  # 注释掉，避免直接执行时输出过多信息


# ========================================
# 6. 测试集成
# ========================================

# 快速测试脚本
if __name__ == "__main__":
    #from result_exporter import TrackingResultExporter
    
    print("测试TrackingResultExporter...")
    
    exporter = TrackingResultExporter()
    exporter.set_video_info('test.mp4', fps=30, resolution=(1920, 1080))
    
    # 模拟追踪数据
    for frame_id in range(1, 101):
        for track_id in range(1, 4):
            is_anomalous = (frame_id > 50) and (track_id == 1)
            reason = '速度过快' if is_anomalous else ''
            
            exporter.add_frame_result(
                frame_id=frame_id,
                track_id=track_id,
                bbox=[100 + track_id*50, 100, 200 + track_id*50, 200],
                class_name='car' if track_id == 1 else 'person',
                confidence=0.95,
                is_anomalous=is_anomalous,
                reason=reason
            )
    
    # 导出
    exporter.export_json()
    exporter.export_statistics()
    exporter.export_summary()
    
    print("\n✓ 测试完成！检查 evaluation/ 目录下的输出文件")
