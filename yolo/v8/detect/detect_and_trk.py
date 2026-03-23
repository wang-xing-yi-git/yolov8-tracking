# 引用yolo/engine/predictor.py中的BasePredictor类，并实现一个新的DetectionPredictor类来处理检测和追踪的逻辑。
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3])) # 将项目根目录添加到系统路径中，以便导入模块

import hydra
import torch
import cv2
from random import randint
from sort import *
# from ultralytics.yolo.engine.predictor import BasePredictor
# 上面的导入方式也可以，这里我采用的导入方式更能清晰地展示我们正在使用的BasePredictor类
from yolo.engine.predictor import BasePredictor
from yolo.utils import DEFAULT_CONFIG, ROOT, ops
from yolo.utils.checks import check_imgsz
from yolo.utils.plotting import Annotator, colors, save_one_box

# 新增：评估指标 - 导入结果导出器
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

try:
    from evaluation.result_exporter import TrackingResultExporter
    EXPORT_AVAILABLE = True
except ImportError:
    EXPORT_AVAILABLE = False
    print("警告: evaluation模块不可用，禁用评估指标导出")

tracker = None

# 简单异常启发式的阈值(每帧像素数和宽高比)
CAR_SPEED_THRESHOLD = 100  # 帧间大跳跃被认为是异常
PERSON_ASPECT_RATIO_THRESHOLD = 0.5  # 宽高比超过此值表示躺倒


def init_tracker():
    global tracker

    sort_max_age = 5
    sort_min_hits = 2
    sort_iou_thresh = 0.2
    tracker = Sort(max_age=sort_max_age, min_hits=sort_min_hits, iou_threshold=sort_iou_thresh)


def _track_is_anomalous(track, names):
    """如果给定的追踪满足我们的异常判断标准，返回(True, reason)。否则返回(False, '')。
    
    新增：评估指标 - 返回异常原因便于评估记录
    """
    cls_idx = int(track.detclass)
    cls_name = names[cls_idx] if names is not None and cls_idx < len(names) else str(cls_idx)
    # 计算最后两个质心之间的位移
    if len(track.centroidarr) >= 2:
        dx = track.centroidarr[-1][0] - track.centroidarr[-2][0]
        dy = track.centroidarr[-1][1] - track.centroidarr[-2][1]
        speed = (dx * dx + dy * dy) ** 0.5
    else:
        speed = 0
    
    # 车辆异常：突然的大位移
    if cls_name in ("car", "truck", "bus", "motorbike", "bicycle") and speed > CAR_SPEED_THRESHOLD:
        # 新增：评估指标 - 返回异常原因
        reason = f'速度过快 (突跳像素数>{CAR_SPEED_THRESHOLD:.1f})'
        return True, reason
    
    # 人物异常：宽高比表示躺倒
    if cls_name == "person" and len(track.bbox_history) > 0:
        bbox = track.bbox_history[-1]
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        if w > 0 and (h / w) < PERSON_ASPECT_RATIO_THRESHOLD:
            # 新增：评估指标 - 返回异常原因
            reason = f'俯身/躺下 (长宽比<{PERSON_ASPECT_RATIO_THRESHOLD})'
            return True, reason
    
    # 新增：评估指标 - 返回空原因表示正常
    return False, ''


rand_color_list = []


def draw_boxes(img, bbox, identities=None, categories=None, names=None, anomalies=None, offset=(0, 0)):
    """在图像上绘制被追踪的对象框。

    追踪ID出现在*anomalies*集合中的框将被绘制为红色
    并标记为"异常"。
    """
    anomalies = anomalies or set()
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        id = int(identities[i]) if identities is not None else 0
        is_anom = id in anomalies
        label = str(id) + (" ANOMALY" if is_anom else "")
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        color = (0, 0, 255) if is_anom else (0, 255, 253)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.rectangle(img, (x1, y1 - 20), (x1 + w, y1), (255, 144, 30), -1)
        cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, [255, 255, 255], 1)

    return img


def random_color_list():
    global rand_color_list
    rand_color_list = []
    for i in range(0, 5005):
        r = randint(0, 255)
        g = randint(0, 255)
        b = randint(0, 255)
        rand_color = (r, g, b)
        rand_color_list.append(rand_color)
    # ......................................


class DetectionPredictor(BasePredictor):

    def __init__(self, cfg):
        """初始化检测预测器
        
        新增：评估指标 - 初始化导出器
        """
        super().__init__(cfg)
        
        # 新增：评估指标 - 初始化结果导出器
        self.result_exporter = None
        if EXPORT_AVAILABLE:
            try:
                self.result_exporter = TrackingResultExporter(output_dir='evaluation')
            except Exception as e:
                print(f"警告: 初始化结果导出器失败: {e}")

    def get_annotator(self, img):
        return Annotator(img, line_width=self.args.line_thickness, example=str(self.model.names))

    def preprocess(self, img):
        img = torch.from_numpy(img).to(self.model.device)
        img = img.half() if self.model.fp16 else img.float()  # uint8转换为fp16/32
        img /= 255  # 0-255转换为0.0-1.0
        return img

    def postprocess(self, preds, img, orig_img):
        preds = ops.non_max_suppression(preds,
                                        self.args.conf,
                                        self.args.iou,
                                        agnostic=self.args.agnostic_nms,
                                        max_det=self.args.max_det)

        for i, pred in enumerate(preds):
            shape = orig_img[i].shape if self.webcam else orig_img.shape
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], shape).round()

        return preds

    def write_results(self, idx, preds, batch):

        p, im, im0 = batch
        log_string = ""
        if len(im.shape) == 3:
            im = im[None]  # 展开为批次维度
        self.seen += 1
        im0 = im0.copy()
        if self.webcam:  # 批量大小 >= 1
            log_string += f'{idx}: '
            frame = self.dataset.count
        else:
            frame = getattr(self.dataset, 'frame', 0)
        # 追踪器
        self.data_path = p

        save_path = str(self.save_dir / p.name)  # 图像文件
        self.txt_path = str(self.save_dir / 'labels' / p.stem) + ('' if self.dataset.mode == 'image' else f'_{frame}')
        log_string += '%gx%g ' % im.shape[2:]  # 打印字符串
        self.annotator = self.get_annotator(im0)

        det = preds[idx]
        self.all_outputs.append(det)
        if len(det) == 0:
            return log_string
        for c in det[:, 5].unique():
            n = (det[:, 5] == c).sum()  # 每个类别的检测数
            log_string += f"{n} {self.model.names[int(c)]}{'s' * (n > 1)}, "

        # #..................使用追踪函数....................
        dets_to_sort = np.empty((0, 6))

        for x1, y1, x2, y2, conf, detclass in det.cpu().detach().numpy():
            dets_to_sort = np.vstack((dets_to_sort,
                                      np.array([x1, y1, x2, y2, conf, detclass])))

        tracked_dets = tracker.update(dets_to_sort)
        tracks = tracker.getTrackers()

        # 根据简单启发式确定哪些追踪ID被认为是异常的
        # 新增：评估指标 - 收集异常原因用于导出
        anomalies = set()
        anomaly_reasons = {}  # 新增：评估指标 - 存储异常原因
        for track in tracks:
            is_anomalous, reason = _track_is_anomalous(track, self.model.names)  # 新增：评估指标 - 获取异常原因
            if is_anomalous:
                anomalies.add(track.id)
                anomaly_reasons[track.id] = reason  # 新增：评估指标 - 记录异常原因

        # 新增：评估指标 - 导出追踪结果
        if self.result_exporter is not None and len(tracked_dets) > 0:
            try:
                if frame == 1:  # 首次导出时设置视频信息
                    fps = self.dataset.fps if hasattr(self.dataset, 'fps') else 30
                    resolution = (im0.shape[1], im0.shape[0])
                    self.result_exporter.set_video_info(
                        video_path=str(p),
                        fps=fps,
                        resolution=resolution
                    )
                
                # 导出每个追踪对象的结果
                for track in tracks:
                    if len(track.bbox_history) > 0:
                        bbox = track.bbox_history[-1]
                        class_idx = int(track.detclass)
                        class_name = self.model.names[class_idx] if class_idx < len(self.model.names) else str(class_idx)
                        confidence = track.confidence if hasattr(track, 'confidence') else 0.95
                        is_anomalous = track.id in anomalies
                        reason = anomaly_reasons.get(track.id, '')
                        
                        self.result_exporter.add_frame_result(
                            frame_id=frame,
                            track_id=track.id,
                            bbox=bbox,
                            class_name=class_name,
                            confidence=confidence,
                            is_anomalous=is_anomalous,
                            reason=reason
                        )
            except Exception as e:
                print(f"警告: 导出追踪结果失败: {e}")

        for track in tracks:
            color = (0, 0, 255) if track.id in anomalies else rand_color_list[track.id]
            [cv2.line(im0, (int(track.centroidarr[i][0]),
                            int(track.centroidarr[i][1])),
                      (int(track.centroidarr[i + 1][0]),
                       int(track.centroidarr[i + 1][1])),
                      color, thickness=3)
             for i, _ in enumerate(track.centroidarr)
             if i < len(track.centroidarr) - 1]

        if len(tracked_dets) > 0:
            bbox_xyxy = tracked_dets[:, :4]
            identities = tracked_dets[:, 8]
            categories = tracked_dets[:, 4]
            draw_boxes(im0, bbox_xyxy, identities, categories, self.model.names, anomalies=anomalies)

        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # 归一化增益 whwh

        return log_string


@hydra.main(version_base=None, config_path=str(DEFAULT_CONFIG.parent), config_name=DEFAULT_CONFIG.name)
def predict(cfg):
    init_tracker()
    random_color_list()

    # 优先使用本地权重文件，如果存在的话，以避免自动下载
    if cfg.model is None:
        # 根据工作空间，项目根目录包含yolov8s.pt
        local = ROOT / "yolov8s.pt"
        if local.exists():
            cfg.model = str(local)
        else:
            cfg.model = "yolov8n.pt"  # 回退到规范名称，这会触发下载
    cfg.imgsz = check_imgsz(cfg.imgsz, min_dim=2)  # 检查图像大小
    cfg.source = cfg.source if cfg.source is not None else ROOT / "assets"
    predictor = DetectionPredictor(cfg)
    predictor()
    
    # 新增：评估指标 - 导出最终结果
    if predictor.result_exporter is not None:
        try:
            print("\n新增：评估指标 - 正在导出评估结果...")
            predictor.result_exporter.export_json()
            predictor.result_exporter.export_statistics()
            predictor.result_exporter.export_summary()
            print("新增：评估指标 - 导出完成！")
        except Exception as e:
            print(f"新增：评估指标 - 导出失败: {e}")


if __name__ == "__main__":
    predict()
    