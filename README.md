# YOLOv8 目标检测和追踪系统

这是一个基于YOLOv8的专业目标检测和多目标追踪系统，兼容`ultralytics==8.0.0`。系统集成了SORT追踪算法，可实现实时的目标检测、追踪和异常行为识别。

## 项目概述

本项目实现了一个完整的目标检测和追踪管道，支持视频、图像和实时摄像头输入，具备以下功能：
- ✅ **实时目标检测** - 使用YOLOv8神经网络进行快速检测
- ✅ **多目标追踪** - 基于SORT算法的目标ID跟踪  
- ✅ **异常行为检测** - 识别不正常的物体移动和姿态
- ✅ **轨迹可视化** - 绘制目标的运动轨迹和追踪框
- ✅ **完整的生命周期管理** - 支持训练、验证和推理

> **注**: 本项目兼容 `ultralytics==8.0.0`，建议使用最新版本并参考官方仓库：[GitHub - Ultralytics](https://github.com/ultralytics/ultralytics/)

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

#### 网络摄像头
```bash
python yolo/v8/detect/detect_and_trk.py model=yolov8m.pt source=0 show=True
```

#### 外接摄像头
```bash
python yolo/v8/detect/detect_and_trk.py model=yolov8m.pt source=1 show=True
```

所有推理结果保存在 `runs/detect/train/` 目录。

---

## 项目结构详细说明

```
yolov8-tracking/
├── yolo/                           # 核心YOLOv8框架代码
│   ├── v8/                        # v8版本特定实现
│   │   └── detect/                # 检测和追踪模块 ⭐ 核心业务逻辑
│   │       ├── detect_and_trk.py  # 🎯 主程序入口
│   │       ├── predict.py         # 检测预测器
│   │       ├── sort.py            # SORT追踪算法
│   │       ├── train.py           # 训练脚本
│   │       └── val.py             # 验证脚本
│   │
│   ├── engine/                    # 核心引擎
│   │   ├── model.py               # YOLO模型类
│   │   ├── predictor.py           # 推理基类
│   │   ├── trainer.py             # 训练器基类
│   │   ├── validator.py           # 验证器基类
│   │   ├── exporter.py            # 模型导出工具
│   │   └── callbacks/             # 回调函数
│   │
│   ├── data/                      # 数据处理
│   │   ├── dataset.py             # 数据集类
│   │   ├── build.py               # 数据加载构建
│   │   ├── augment.py             # 数据增强
│   │   ├── utils.py               # 数据工具
│   │   ├── dataloaders/           # 加载器
│   │   │   ├── stream_loaders.py  # 视频/图像流
│   │   │   ├── v5loader.py        # YOLOv5兼容
│   │   │   └── v5augmentations.py # 增强方法
│   │   └── datasets/              # 数据集配置
│   │
│   ├── utils/                     # 工具函数
│   │   ├── torch_utils.py         # PyTorch工具
│   │   ├── ops.py                 # 操作算子
│   │   ├── metrics.py             # 指标计算
│   │   ├── loss.py                # 损失函数
│   │   ├── checks.py              # 验证函数
│   │   ├── plotting.py            # 可视化
│   │   ├── files.py               # 文件操作
│   │   └── callbacks/             # 回调集成
│   │
│   └── configs/                   # 配置管理
│       ├── default.yaml           # 默认配置
│       ├── hydra_patch.py         # Hydra补丁
│       └── __init__.py            # 初始化
│
├── nn/                             # 神经网络模块
│   ├── tasks.py                   # 任务模型
│   ├── modules.py                 # 网络模块
│   └── autobackend.py             # 后端选择
│
├── models/                        # 模型配置
│   └── v8/                        # YOLOv8配置
│       ├── yolov8n.yaml           # Nano
│       ├── yolov8s.yaml           # Small
│       ├── yolov8m.yaml           # Medium
│       ├── yolov8l.yaml           # Large
│       └── yolov8x.yaml           # XLarge
│
├── runs/                          # 输出结果
│   └── detect/train/              # 推理输出
│
├── yolov8n.pt                     # 权重文件
├── yolov8s.pt                     # 权重文件
├── requirements.txt               # 依赖清单
├── LICENSE                        # 许可证
└── README.md                      # 文档
```

---

## 核心文件说明

### 🎯 detect_and_trk.py - 主程序入口
**主要功能**: 检测和追踪的完整管道

- `init_tracker()` - 初始化SORT追踪器
- `DetectionPredictor.preprocess()` - 图像预处理
- `DetectionPredictor.postprocess()` - 后处理（NMS）
- `DetectionPredictor.write_results()` - 结果处理和可视化
- `predict()` - Hydra配置入口点

### 📊 sort.py - 多目标追踪算法
**实现**: SORT（Simple Online and Realtime Tracking）

- `KalmanFilter` - 卡尔曼滤波器(7维状态)
- `linear_assignment()` - 匈牙利算法关联
- `iou_batch()` - IOU相似度计算
- `Track` 类 - 追踪对象生命周期管理

### 🔧 predictor.py - 推理基类
**提供**: 统一的推理接口

- `BasePredictor.setup()` - 初始化模型
- `BasePredictor.__call__()` - 主推理循环
- 支持多种输入源（视频、图像、摄像头等）

### 🧠 model.py - 模型管理
**功能**: 统一的YOLO模型接口

- `YOLO` 类 - 模型封装
- 自动任务识别（分类、检测、分割）
- 模型加载、导出、转换

---

## 程序完整执行流程

### 📹 视频处理流程示例

执行: `python yolo/v8/detect/detect_and_trk.py model=yolov8s.pt source="video.mp4" show=True`

#### 详细流程步骤:

```
【步骤1】初始化阶段
├─ detect_and_trk.py的predict()函数启动
│  ├─ init_tracker() 创建SORT追踪器
│  │  (max_age=5, min_hits=2, iou_threshold=0.2)
│  ├─ random_color_list() 生成5005个颜色
│  └─ 加载Hydra配置参数
│
└─ 加载YOLOv8s权重并初始化数据加载器

【步骤2】模型加载
├─ 检查本地权重(yolov8s.pt)
├─ 若无则自动下载
└─ 加载到GPU/CPU

【步骤3】创建推理器
├─ 实例化DetectionPredictor(cfg)
└─ 启动主推理循环

【步骤4】对每一帧处理（循环）

  [4.1] 加载视频帧
        LoadStreams → 读取单帧
        输入: (H,W,3) uint8, [0-255]
        
        ↓
        
  [4.2] 图像预处理 (preprocess)
        • 转张量 → GPU → 数据类型转换 → 归一化
        输出: (1,3,640,640) float32, [0-1]
        
        ↓
        
  [4.3] 神经网络推理
        Backbone → FPN → Detect Head
        输出: (N,85) 检测框预测
        
        ↓
        
  [4.4] 后处理 (postprocess)
        • NMS过滤低置信度和重叠框
        • 坐标缩放回原始尺寸
        输出: (M,6) x1,y1,x2,y2,conf,cls
        
        ↓
        
  [4.5] 追踪和异常检测 (write_results)
        
        [A] SORT追踪更新 (tracker.update)
            ├─ Kalman预测
            ├─ IOU相似度计算
            ├─ 匈牙利算法匹配
            └─ 追踪状态管理
        
        [B] 异常行为检测
            ├─ 车辆: 检测突然大位移
            └─ 人物: 检测躺倒姿态
        
        [C] 可视化
            ├─ 绘制边界框(红=异常 青=正常)
            ├─ 显示追踪ID
            └─ 绘制运动轨迹
        
        ↓
        
  [4.6] 保存处理后的帧
        VideoWriter → runs/detect/train/video.mp4
        
        ↓
        
  返回步骤4.1处理下一帧
  
【步骤5】完成
└─ 视频处理结束，释放资源
```

### 🖼️ 图像处理
```
加载 → 预处理 → 推理 → 后处理 → 追踪 → 异常检测 → 可视化 → 保存
```

### 🎥 摄像头实时处理
```
摄像头流 → 连续读取帧 → 每帧执行: 预处理→推理→后处理→追踪→可视化
```

---

## SORT追踪算法详解

### 算法流程

```
【输入】
当前帧检测框: D_t = {d1, d2, ..., dm}
前帧活跃追踪: T_{t-1} = {t1, t2, ..., tn}

【处理步骤】

[1] 预测阶段
    对每个追踪t_i ∈ T_{t-1}，Kalman滤波预测新位置p_i
    → 预测集合 P_t = {p1, p2, ..., pn}

[2] 相似度计算  
    计算IOU(p_i, d_j) ∀ i,j
    → n×m 代价矩阵: cost[i,j] = 1 - IOU(p_i, d_j)

[3] 数据关联
    匈牙利算法求最小二部匹配
    → 已关联(IOU>thresh) | 未匹配追踪 | 未匹配检测

[4] 状态更新
    ✓ 已关联: 用新检测更新Kalman状态
    ✗ 未匹配检测: 初始化新追踪
    ✗ 未匹配追踪: age++，若age>max_age则删除

【输出】
更新后的追踪集合T_t和当前帧追踪结果
```

### Kalman滤波器状态

```
状态向量 (7维):
x = [x, y, s, r, vx, vy, vs]

含义:
  (x,y)      = 边界框中心
  s          = 面积 (宽×高)
  r          = 宽高比 (宽/高)
  (vx,vy,vs) = 速度 (帧间位移)

两个步骤:
  预测: x_{t|t-1} = F×x_{t-1|t-1}  (恒定速度)
  更新: x_{t|t} = x_{t|t-1} + K(z_t - H·x_{t|t-1})  (融合测量)
```

---

## 配置参数

### 命令行参数

```bash
python yolo/v8/detect/detect_and_trk.py \
    model=yolov8s.pt    # 模型权重
    source="video.mp4"  # 输入源
    show=True           # 显示结果
    save=True           # 保存结果
    conf=0.25           # 置信度阈值
    iou=0.45            # NMS IOU阈值
    device=0            # GPU编号
    imgsz=640           # 推理尺寸
```

### SORT参数（detect_and_trk.py）

```python
sort_max_age = 5          # 丢失多少帧后删除
sort_min_hits = 2         # 多少次检测后创建追踪
sort_iou_thresh = 0.2     # 关联IOU阈值
```

### 异常检测阈值

```python
CAR_SPEED_THRESHOLD = 100              # 车辆速度阈值(px/帧)
PERSON_ASPECT_RATIO_THRESHOLD = 0.5    # 人物宽高比阈值
```

---

## 模型对比

| 模型 | 参数量 | 速度 | 精度 | 场景 |
|------|--------|------|------|------|
| YOLOv8n | 3.2M | 最快 | 低 | 边缘设备 |
| YOLOv8s | 11.2M | 快 | 中 | **推荐** |
| YOLOv8m | 25.9M | 中 | 中高 | 生产环境 |
| YOLOv8l | 43.7M | 慢 | 高 | 高精度 |
| YOLOv8x | 68.2M | 最慢 | 最高 | 最高精度 |

---

## 常见问题

### GPU内存不足
```bash
# 使用轻量级模型
python yolo/v8/detect/detect_and_trk.py model=yolov8n.pt source="video.mp4"

# 或使用CPU
python yolo/v8/detect/detect_and_trk.py model=yolov8s.pt device=cpu source="video.mp4"
```

### 追踪ID频繁变化
```python
# detect_and_trk.py中调整:
sort_max_age = 30         # 增加允许丢失的帧数
sort_min_hits = 1         # 减少创建追踪所需次数
sort_iou_thresh = 0.3     # 增加关联容度
```

### 检测效果不理想
```bash
python yolo/v8/detect/detect_and_trk.py \
    model=yolov8m.pt \
    source="video.mp4" \
    conf=0.5 \         # 提高置信度
    iou=0.5            # 提高NMS阈值
```

---

## 输出说明

### 输出目录

```
runs/detect/train/
├── video_name.mp4          # 处理后的视频
├── labels/                 # YOLO格式标签
│   ├── frame_001.txt
│   └── ...
└── crops/                  # 裁剪的对象
    ├── car/
    └── person/
```

### 标签格式

```
class_id track_id x1 y1 x2 y2 confidence
```

---

## 性能指标

| 模型 | 输入 | 推理时间 | FPS | mAP50 |
|------|------|---------|-----|-------|
| YOLOv8n | 640 | 25ms | 40 | 50.4 |
| YOLOv8s | 640 | 40ms | 25 | 55.2 |
| YOLOv8m | 640 | 70ms | 14.3 | 60.1 |
| YOLOv8l | 640 | 95ms | 10.5 | 63.4 |
| YOLOv8x | 640 | 130ms | 7.7 | 65.2 |

*基于GPU推理 (RTX 3080)*

---

## 参考资源

- **官方YOLOv8**: https://github.com/ultralytics/ultralytics
- **SORT论文**: https://arXiv.org/pdf/1602.00763.pdf
- **SORT实现**: https://github.com/abewley/sort
- **YOLOv8文档**: https://docs.ultralytics.com/
- **训练教程**: https://muhammadrizwanmunawar.medium.com/train-yolov8-on-custom-data-6d28cd348262

### 作者文章

| 标题 | 日期 |
|------|------|
| [YOLO11: 目标检测和分割](https://muhammadrizwanmunawar.medium.com/ultralytics-yolo11-object-detection-and-instance-segmentation-88ef0239a811) | 2024-10-27 |
| [停车位管理系统](https://muhammadrizwanmunawar.medium.com/parking-management-using-ultralytics-yolo11-fba4c6bc62bc) | 2024-11-10 |
| [计算机视觉爱好项目](https://muhammadrizwanmunawar.medium.com/my-️computer-vision-hobby-projects-that-yielded-earnings-7923c9b9eead) | 2023-09-10 |
| [CV学习资源汇总](https://muhammadrizwanmunawar.medium.com/best-resources-to-learn-computer-vision-311352ed0833) | 2023-06-30 |
| [CV工程师路线图](https://medium.com/augmented-startups/roadmap-for-computer-vision-engineer-45167b94518c) | 2022-08-07 |

---

## 检测效果示例

<table>
  <tr>
    <td>YOLOv8s 目标检测和追踪</td>
    <td>YOLOv8m 目标检测和追踪</td>
  </tr>
  <tr>
    <td><img src="https://user-images.githubusercontent.com/62513924/211671576-7d39829a-f8f5-4e25-b30a-530548c11a24.png"></td>
    <td><img src="https://user-images.githubusercontent.com/62513924/211672010-7415ef8b-7941-4545-8434-377d94675299.png"></td>
  </tr>
</table>

---

## 许可证

GPL-3.0 - 详见 [LICENSE](LICENSE) 文件

---

## 联系方式

**原项目作者**: Muhammad Rizwan Munawar
- Medium: https://muhammadrizwanmunawar.medium.com/
- LinkedIn: https://www.linkedin.com/in/muhammadrizwanmunawar/

有任何问题或建议，欢迎提Issue或PR！
