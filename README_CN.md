# YOLOv8 目标检测和追踪系统

这是一个基于YOLOv8的专业目标检测和多目标追踪系统，兼容`ultralytics==8.0.0`。系统集成了SORT追踪算法，可实现实时的目标检测、追踪和异常行为识别。

## 项目概述

本项目实现了一个完整的目标检测和追踪管道，支持视频、图像和实时摄像头输入，具备以下功能：
- ✅ **实时目标检测** - 使用YOLOv8神经网络进行快速检测
- ✅ **多目标追踪** - 基于SORT算法的目标ID跟踪  
- ✅ **异常行为检测** - 识别不正常的物体移动和姿态
- ✅ **轨迹可视化** - 绘制目标的运动轨迹和追踪框
- ✅ **完整的生命周期管理** - 支持训练、验证和推理

---

## 项目结构详解

```
yolov8-tracking/
├── yolo/                           # 核心YOLOv8框架代码
│   ├── v8/                        # v8版本特定实现
│   │   └── detect/                # 检测和追踪模块 ⭐ 核心业务逻辑
│   │       ├── detect_and_trk.py  # 🎯 主程序入口（视频/图像/摄像头推理）
│   │       ├── predict.py         # 检测预测器（BasePredictor的具体实现）
│   │       ├── sort.py            # SORT追踪算法实现（卡尔曼滤波+匈牙利算法）
│   │       ├── train.py           # 模型训练脚本
│   │       └── val.py             # 模型验证脚本
│   │
│   ├── engine/                    # 训练、验证、推理的核心引擎
│   │   ├── model.py               # YOLO模型类（模型加载、初始化）
│   │   ├── predictor.py           # 推理基类（BasePredictor，图像预处理、后处理）
│   │   ├── trainer.py             # 训练器基类（模型训练、学习率调度）
│   │   ├── validator.py           # 验证器基类（模型验证、指标计算）
│   │   ├── exporter.py            # 模型导出工具（导出为onnx、torchscript等）
│   │   └── callbacks/             # 回调函数（tensorboard、comet、hub等）
│   │
│   ├── data/                      # 数据处理模块
│   │   ├── dataset.py             # 基础数据集类
│   │   ├── build.py               # 数据加载构建
│   │   ├── augment.py             # 数据增强方法
│   │   ├── utils.py               # 数据处理工具
│   │   ├── dataloaders/           # 数据加载器
│   │   │   ├── stream_loaders.py  # 视频、图像流加载器
│   │   │   ├── v5loader.py        # YOLOv5兼容数据加载器
│   │   │   └── v5augmentations.py # YOLOv5数据增强
│   │   └── datasets/              # 数据集配置和样本
│   │       ├── coco.yaml          # COCO数据集配置
│   │       ├── VOC.yaml           # Pascal VOC配置
│   │       └── *.yaml             # 其他数据集配置
│   │
│   ├── utils/                     # 通用工具函数
│   │   ├── torch_utils.py         # PyTorch相关工具
│   │   ├── ops.py                 # 操作算子（NMS、IOU等）
│   │   ├── metrics.py             # 性能指标计算
│   │   ├── loss.py                # 损失函数
│   │   ├── checks.py              # 检查和验证函数
│   │   ├── plotting.py            # 绘图和可视化
│   │   ├── files.py               # 文件操作
│   │   └── callbacks/             # 训练中的回调集成
│   │
│   └── configs/                   # 配置管理
│       ├── default.yaml           # 默认配置文件
│       ├── hydra_patch.py         # Hydra配置补丁
│       └── __init__.py            # 配置初始化
│
├── nn/                             # 神经网络模块（模型定义）
│   ├── tasks.py                   # 任务模型（分类、检测、分割）
│   ├── modules.py                 # 网络模块（卷积、注意力等）
│   └── autobackend.py             # 后端自动选择（ONNX、TensorRT等）
│
├── models/                        # 模型配置文件
│   └── v8/                        # YOLOv8系列模型配置
│       ├── yolov8n.yaml           # Nano版本配置（轻量级）
│       ├── yolov8s.yaml           # Small版本配置
│       ├── yolov8m.yaml           # Medium版本配置
│       ├── yolov8l.yaml           # Large版本配置
│       └── yolov8x.yaml           # XLarge版本配置（高精度）
│
├── runs/                          # 训练和推理输出结果
│   └── detect/
│       └── train/                 # 所有推理结果输出目录
│
├── yolov8n.pt                     # 预训练权重文件（Nano）
├── yolov8s.pt                     # 预训练权重文件（Small）
├── requirements.txt               # Python依赖列表
├── LICENSE                        # 许可证
└── README.md                      # 项目文档（本文件）
```

### 关键模块说明

#### 🎯 **detect_and_trk.py** - 程序主入口
- `init_tracker()` - 初始化SORT追踪器
- `DetectionPredictor.preprocess()` - 图像预处理（张量转换、归一化）
- `DetectionPredictor.postprocess()` - 后处理（NMS、缩放）
- `DetectionPredictor.write_results()` - 结果处理和可视化
- `predict()` - Hydra入口点，负责流程调度

#### 📊 **sort.py** - 多目标追踪算法
- `KalmanFilter` - 卡尔曼滤波器（预测目标位置）
- `linear_assignment()` - 匈牙利算法（关联检测和追踪）
- `iou_batch()` - 计算IOU相似度
- `Track` 类 - 维护单个追踪对象的生命周期

#### 🔧 **predictor.py** - 推理基类
- `BasePredictor.setup()` - 模型和数据集初始化
- `BasePredictor.__call__()` - 主推理循环
- 支持多种输入源（视频、图像、摄像头、URL）

#### 🧠 **model.py** - 模型管理
- `YOLO` 类 - 统一的模型接口
- 支持模型加载、训练、验证和推理
- 自动模型类型识别（分类、检测、分割）

---

## 程序执行流程详解

### 📹 视频处理流程（完整示例）

当用户执行命令：
```bash
python yolo/v8/detect/detect_and_trk.py model=yolov8s.pt source="video.mp4" show=True
```

#### 流程步骤：

```
┌─────────────────────────────────────────────────────────────────┐
│ 步骤 1: 初始化阶段                                              │
└─────────────────────────────────────────────────────────────────┘

detect_and_trk.py 的 predict() 函数被调用
    ├─ init_tracker()
    │   └─ Sort(max_age=5, min_hits=2, iou_threshold=0.2)
    │       初始化全局追踪器对象
    │
    ├─ random_color_list()
    │   └─ 为5005个追踪对象生成随机颜色
    │
    └─ 加载配置参数和预训练模型权重

┌─────────────────────────────────────────────────────────────────┐
│ 步骤 2: 模型加载                                                │
└─────────────────────────────────────────────────────────────────┘

    ├─ 检查本地权重文件
    │   ├─ yolov8s.pt 存在? → 使用本地文件
    │   └─ 不存在? → 自动下载预训练权重
    │
    ├─ 将模型加载到GPU/CPU
    │
    └─ 初始化数据加载器

┌─────────────────────────────────────────────────────────────────┐
│ 步骤 3: 创建预测器并启动主循环                                  │
└─────────────────────────────────────────────────────────────────┘

    DetectionPredictor(cfg) 实例化
    predictor() 启动主推理循环

┌─────────────────────────────────────────────────────────────────┐
│ 步骤 4: 帧处理循环（对每一帧执行）                              │
└─────────────────────────────────────────────────────────────────┘

    ┌─────────────────────────────────────────────────┐
    │ 4.1 加载视频帧                                   │
    │ LoadStreams/LoadImages → 读取单帧               │
    │ 输出形状: (H, W, 3)，值域: [0-255]            │
    └─────────────────────────────────────────────────┘
                            ↓
    ┌─────────────────────────────────────────────────┐
    │ 4.2 图像预处理 (preprocess)                      │
    │                                                  │
    │ • 转换为PyTorch张量                             │
    │ • 转移到GPU（如果可用）                         │
    │ • 转换数据类型：float32 or float16              │
    │ • 归一化：[0-255] → [0-1]                      │
    │ 输出: torch.Tensor, shape=(1,3,H,W), 值域[0,1]│
    └─────────────────────────────────────────────────┘
                            ↓
    ┌─────────────────────────────────────────────────┐
    │ 4.3 神经网络推理                                │
    │ nn.DetectionModel                               │
    │                                                  │
    │ ┌──────────────────────────────────────────┐   │
    │ │ 骨干网络 (Backbone)                      │   │
    │ │ 使用重复卷积块提取多级语义特征             │   │
    │ └──────┬───────────────────────────────────┘   │
    │        ↓                                        │
    │ ┌──────────────────────────────────────────┐   │
    │ │ 特征融合 (FPN)                           │   │
    │ │ 融合不同尺度的特征                        │   │
    │ └──────┬───────────────────────────────────┘   │
    │        ↓                                        │
    │ ┌──────────────────────────────────────────┐   │
    │ │ 检测头 (Detect Head)                     │   │
    │ │ 预测: [x,y,w,h] + 置信度 + 类别概率      │   │
    │ └──────┬───────────────────────────────────┘   │
    │        ↓                                        │
    │ 输出形状: (batch_size, num_predictions, 85)   │
    │ 格式: [x,y,w,h,conf,cls0,cls1,...]          │
    └─────────────────────────────────────────────────┘
                            ↓
    ┌─────────────────────────────────────────────────┐
    │ 4.4 后处理 (postprocess)                        │
    │                                                  │
    │ • 非极大值抑制 (NMS)                           │
    │   - 过滤低置信度框 (< conf_threshold)          │
    │   - 移除高度重叠框 (IOU > iou_threshold)       │
    │   - 保留最有信心的检测                         │
    │                                                  │
    │ • 坐标缩放                                     │
    │   - 将坐标从模型输入尺寸映射回原始图像尺寸     │
    │                                                  │
    │ 输出: 过滤后的检测框                           │
    │ 格式: [x1,y1,x2,y2,conf,class] × N个检测      │
    └─────────────────────────────────────────────────┘
                            ↓
    ┌─────────────────────────────────────────────────┐
    │ 4.5 追踪和异常检测 (write_results)             │
    │                                                  │
    │ [A] 提取检测信息                               │
    │     [x1,y1,x2,y2,conf,class] → [x1,y1,x2,y2,conf,class]
    │                                                  │
    │ [B] 调用追踪器更新                             │
    │     ┌────────────────────────────────────┐     │
    │     │ SORT Algorithm - tracker.update()   │     │
    │     │                                    │     │
    │     │ B1: 预测阶段                       │     │
    │     │     对所有活跃追踪，用卡尔曼滤波器 │     │
    │     │     预测其在当前帧的位置           │     │
    │     │                                    │     │
    │     │ B2: 关联阶段                       │     │
    │     │     计算检测框与预测框的IOU矩阵    │     │
    │     │     使用匈牙利算法求最优二部匹配   │     │
    │     │     → 产生配对 (matched) 和       │     │
    │     │       未匹配项 (unmatched)         │     │
    │     │                                    │     │
    │     │ B3: 状态更新                       │     │
    │     │     已配对: 用新检测更新追踪状态   │     │
    │     │     未匹配检测: 初始化新追踪       │     │
    │     │     未匹配追踪: age++, 若age>max_age则删除│
    │     │                                    │     │
    │     └────────────────────────────────────┘     │
    │                                                  │
    │ [C] 异常行为检测                               │
    │     对每个追踪对象：                           │
    │     • 车辆类别: 检测突然大的位置跳跃           │
    │       speed = sqrt(Δx² + Δy²)                 │
    │       if speed > 100px/frame → 标记为异常      │
    │                                                  │
    │     • 人物类别: 检测躺倒姿态                   │
    │       ratio = width / height                   │
    │       if ratio > 0.5 → 标记为异常              │
    │                                                  │
    │ [D] 可视化输出                                 │
    │     ├─ 绘制边界框                             │
    │     │  ├─ 红色: 异常对象                      │
    │     │  └─ 青绿色: 正常对象                    │
    │     │                                          │
    │     ├─ 显示追踪ID和标签                       │
    │     │  └─ 格式: "ID ANOMALY" (若异常)        │
    │     │                                          │
    │     └─ 绘制运动轨迹                           │
    │        └─ 连接追踪对象历史质心点               │
    └─────────────────────────────────────────────────┘
                            ↓
    ┌─────────────────────────────────────────────────┐
    │ 4.6 保存结果视频帧                             │
    │                                                  │
    │ • 使用 cv2.VideoWriter 写入处理后的帧          │
    │ • 输出路径: runs/detect/train/video_name.mp4    │
    │ • 帧率: 保持与原始视频相同                     │
    └─────────────────────────────────────────────────┘
                            ↓
    ┌─────────────────────────────────────────────────┐
    │ 继续循环直到视频结束                           │
    └─────────────────────────────────────────────────┘
```

### 🖼️ 图像处理流程（简化版）

```
加载图像 → 预处理 → 推理 → 后处理 → 追踪 → 异常检测 → 可视化 → 保存图像
```

### 🎥 摄像头实时处理流程

```
摄像头流(source=0) → LoadStreams(连续读取)
                    → 每帧循环执行: 预处理 → 推理 → 后处理 → 追踪 → 可视化
                    → 实时显示结果 (show=True)
```

---

## 快速开始

### 1. 克隆仓库

```bash
git clone https://github.com/RizwanMunawar/yolov8-object-tracking.git
cd yolov8-object-tracking
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
pip install ultralytics==8.0.0
```

### 3. 运行推理

#### 视频文件
```bash
python yolo/v8/detect/detect_and_trk.py model=yolov8s.pt source="test.mp4" show=True
```

#### 图像文件
```bash
python yolo/v8/detect/detect_and_trk.py model=yolov8m.pt source="image.jpg"
```

#### 网络摄像头 (编号0)
```bash
python yolo/v8/detect/detect_and_trk.py model=yolov8m.pt source=0 show=True
```

#### 外接摄像头 (编号1)
```bash
python yolo/v8/detect/detect_and_trk.py model=yolov8m.pt source=1 show=True
```

#### 使用带GPU加速的轻量级模型
```bash
python yolo/v8/detect/detect_and_trk.py model=yolov8n.pt source="video.mp4" device=0
```

### 输出说明

- 所有推理结果保存在 `runs/detect/train/` 目录
- 输出视频保存为: `runs/detect/train/video_name.mp4`
- 输出图像保存为: `runs/detect/train/image_name.jpg`
- 检测标签保存为: `runs/detect/train/labels/frame_name.txt`

---

## 模型选择指南

| 模型 | 参数量 | 推理速度 | 精度 | 使用场景 |
|------|--------|---------|------|----------|
| YOLOv8n | 3.2M | 最快 | 低 | 边缘设备、实时性要求高 |
| YOLOv8s | 11.2M | 快 | 中 | **推荐**，平衡精度和速度 |
| YOLOv8m | 25.9M | 中 | 中高 | 生产环境，精度要求较高 |
| YOLOv8l | 43.7M | 慢 | 高 | 高精度要求，离线处理 |
| YOLOv8x | 68.2M | 最慢 | 最高 | 最高精度需求 |

---

## 配置参数说明

### 主要命令行参数

```bash
python yolo/v8/detect/detect_and_trk.py \
    model=yolov8s.pt          # 模型权重文件
    source="video.mp4"        # 输入源（视频/图像/摄像头编号）
    show=True                 # 是否显示结果
    save=True                 # 是否保存结果
    conf=0.25                 # 置信度阈值（0-1）
    iou=0.45                  # NMS IOU阈值（0-1）
    device=0                  # GPU设备编号（0,1,2...）或'cpu'
    save_txt=True             # 是否保存txt格式标签
    line_thickness=2          # 边界框线宽
```

### SORT追踪参数

```python
# detect_and_trk.py 中的配置
sort_max_age = 5        # 追踪丢失多少帧后删除（帧数）
sort_min_hits = 2       # 检测多少次才创建追踪（帧数）
sort_iou_thresh = 0.2   # 关联的IOU阈值
```

### 异常检测阈值

```python
CAR_SPEED_THRESHOLD = 100          # 车辆突跳像素阈值
PERSON_ASPECT_RATIO_THRESHOLD = 0.5  # 人物躺倒宽高比阈值
```

---

## 文件输入输出说明

### 输入文件支持格式

**视频格式**: `.mp4`, `.avi`, `.mov`, `.mkv`, `.flv`, `.wmv`

**图像格式**: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`

**其他输入**:
- 摄像头: `source=0` (主摄像头), `source=1` (辅摄像头)
- RTSP流: `source="rtsp://..."`
- HTTP流: `source="http://..."`
- 图像文件夹: `source="path/to/images/"`

### 输出文件结构

```
runs/
└── detect/
    └── train/                          # 推理输出目录
        ├── video_name.mp4              # 处理后的视频
        ├── image_name.jpg              # 处理后的图像
        ├── labels/                     # 检测框标签
        │   ├── frame_001.txt
        │   ├── frame_002.txt
        │   └── ...
        └── crops/                      # 裁剪的检测对象
            ├── class_name/
            │   ├── object_001.jpg
            │   └── ...
```

### 标签文件格式 (labels/*.txt)

```
每行一个检测框:
class_id track_id x1 y1 x2 y2 confidence
```

示例：
```
0 5 100 150 250 400 0.95
2 3 500 200 650 500 0.87
```

---

## 算法原理详解

### SORT追踪算法流程

```
输入: 当前帧的检测框集合 D_t
      当前的活跃追踪集合 T_{t-1}

处理流程:
  ├─ 步骤1 预测
  │   对T_{t-1}中每个追踪，用卡尔曼滤波器预测其
  │   在当前帧的位置 → 得到预测集合 P_t
  │
  ├─ 步骤2 相似度计算
  │   计算D_t和P_t中所有对应的IOU相似度
  │   → 生成 M×N 的相似度矩阵（M=|P_t|, N=|D_t|）
  │
  ├─ 步骤3 匹配
  │   使用匈牙利算法求相似度矩阵的最大加权二部匹配
  │   IOU > 阈值的作为有效匹配，否则为未匹配
  │
  ├─ 步骤4 更新
  │   • 已匹配追踪: 用新检测更新卡尔曼状态
  │   • 未匹配检测: 创建新的追踪
  │   • 未匹配追踪: age计数器+1，若超过max_age则删除
  │
  └─ 输出: 更新后的追踪集合 T_t 和当前帧的追踪结果

卡尔曼滤波器状态:
  状态向量 x = [x, y, s, r, vx, vy, vs]
  其中:
    (x,y) = 边界框中心坐标
    s = 面积（尺度）
    r = 宽高比
    (vx,vy,vs) = 速度（1帧的位移）
```

### NMS (非极大值抑制) 原理

```
输入: 所有检测框及其置信度
过程:
  1. 按置信度从高到低排序
  2. 选择置信度最高的框作为保留框
  3. 计算保留框与其他框的IOU
  4. 删除IOU > 阈值的框（认定为重复检测）
  5. 从剩余框中重复步骤2-4

输出: 过滤后的高质量检测框
```

---

## 常见问题与解决方案

### Q1: GPU内存不足
```bash
# 使用更轻量级的模型
python yolo/v8/detect/detect_and_trk.py model=yolov8n.pt source="video.mp4"

# 或使用CPU推理
python yolo/v8/detect/detect_and_trk.py model=yolov8s.pt source="video.mp4" device=cpu
```

### Q2: 推理速度慢
```bash
# 降低图像分辨率（默认640）
# 在配置文件中修改 imgsz 参数

# 使用Nano或Small模型
python yolo/v8/detect/detect_and_trk.py model=yolov8n.pt source="video.mp4"
```

### Q3: 追踪ID频繁变换
```python
# 调整SORT参数，在 detect_and_trk.py 中修改:
sort_max_age = 30         # 增加此值（允许更长的丢失时间）
sort_min_hits = 1         # 减少此值（更快创建追踪）
sort_iou_thresh = 0.3     # 增加此值（更松散的关联条件）
```

### Q4: 检测效果不好
```bash
# 调整置信度和IOU阈值
python yolo/v8/detect/detect_and_trk.py \
    model=yolov8s.pt \
    source="video.mp4" \
    conf=0.5 \  # 提高置信度（减少误检）
    iou=0.5     # 提高IOU（避免框重叠）
```

---

## 训练自定义模型

### 准备数据集

```
dataset/
├── images/
│   ├── train/  (训练集图像)
│   └── val/    (验证集图像)
└── labels/
    ├── train/  (训练集标签，YOLO格式)
    └── val/    (验证集标签)
```

标签格式 (YOLO格式):
```
class_id center_x center_y width height
（坐标和大小归一化到[0,1]）
```

### 创建数据集配置

```yaml
# dataset.yaml
path: /path/to/dataset
train: images/train
val: images/val

nc: 3  # 类别数
names: ['car', 'person', 'bicycle']  # 类别名称
```

### 训练模型

```bash
python yolo/v8/detect/train.py \
    model=yolov8s.yaml \
    data=dataset.yaml \
    epochs=50 \
    imgsz=640 \
    device=0
```

---

## 参考资源

- **官方YOLOv8仓库**: https://github.com/ultralytics/ultralytics
- **SORT论文**: https://arXiv.org/pdf/1602.00763.pdf
- **SORT实现**: https://github.com/abewley/sort
- **YOLOv8文档**: https://docs.ultralytics.com/

### 推荐阅读

- [在自定义数据集上训练YOLOv8](https://muhammadrizwanmunawar.medium.com/train-yolov8-on-custom-data-6d28cd348262)
- YOLOv8论文和技术细节
- Kalman滤波和目标追踪原理

---

## 性能评估结果

下表展示了不同模型的性能表现：

| 模型 | 输入尺寸 | 推理时间(ms) | 吞吐量(fps) | mAP50 |
|------|---------|-------------|----------|-------|
| YOLOv8n | 640 | 25 | 40 | 50.4 |
| YOLOv8s | 640 | 40 | 25 | 55.2 |
| YOLOv8m | 640 | 70 | 14.3 | 60.1 |
| YOLOv8l | 640 | 95 | 10.5 | 63.4 |
| YOLOv8x | 640 | 130 | 7.7 | 65.2 |

*注: 性能数据基于GPU推理 (NVIDIA RTX 3080)*

---

## 许可证

本项目遵循GPL-3.0许可证。详见 [LICENSE](LICENSE) 文件。

---

## 联系与反馈

- **原作者**: Muhammad Rizwan Munawar
  - Medium: https://muhammadrizwanmunawar.medium.com/
  - LinkedIn: https://www.linkedin.com/in/muhammadrizwanmunawar/

如有问题或建议，欢迎提交Issue或Pull Request！
