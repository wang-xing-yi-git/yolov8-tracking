# Ultralytics YOLO 🚀, GPL-3.0 license
"""
在图像、视频、目录、汜伟、YouTube、网络摄像头、流等上运行预测。
使用 - 来源：
    $ yolo task=... mode=predict  model=s.pt --source 0                         # 网络摄像头
                                                img.jpg                         # 图像
                                                vid.mp4                         # 视频
                                                screen                          # 截屏
                                                path/                           # 目录
                                                list.txt                        # 图像列表
                                                list.streams                    # 流列表
                                                'path/*.jpg'                    # glob模式
                                                'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP流
使用 - 格式：
    $ yolo task=... mode=predict --weights yolov8n.pt          # PyTorch
                                    yolov8n.torchscript        # TorchScript
                                    yolov8n.onnx               # ONNX Runtime或OpenCV DNN，带--dnn
                                    yolov8n_openvino_model     # OpenVINO
                                    yolov8n.engine             # TensorRT
                                    yolov8n.mlmodel            # CoreML (macOS专用)
                                    yolov8n_saved_model        # TensorFlow SavedModel
                                    yolov8n.pb                 # TensorFlow GraphDef
                                    yolov8n.tflite             # TensorFlow Lite
                                    yolov8n_edgetpu.tflite     # TensorFlow Edge TPU
                                    yolov8n_paddle_model       # PaddlePaddle
    """
import platform
from collections import defaultdict
from pathlib import Path
import cv2
from nn.autobackend import AutoBackend
from yolo.configs import get_config
from yolo.data.dataloaders.stream_loaders import LoadImages, LoadScreenshots, LoadStreams
from yolo.data.utils import IMG_FORMATS, VID_FORMATS
from yolo.utils import DEFAULT_CONFIG, LOGGER, SETTINGS, callbacks, colorstr, ops
from yolo.utils.checks import check_file, check_imgsz, check_imshow
from yolo.utils.files import increment_path
from yolo.utils.torch_utils import select_device, smart_inference_mode



class BasePredictor:
    """
    基类预测器

    用于创建预测器的基类。

    属性：
        args (OmegaConf): 预测器的配置。
        save_dir (Path): 保存结果的目录。
        done_setup (bool): 预测器是否完成了设置。
        model (nn.Module): 用于预测的模型。
        data (dict): 数据配置。
        device (torch.device): 用于预测的设备。
        dataset (Dataset): 用于预测的数据集。
        vid_path (str): 视频文件的路径。
        vid_writer (cv2.VideoWriter): 用于保存视频输出的视频作者。
        annotator (Annotator): 用于预测的注释器。
        data_path (str): 数据路径。
    """

    def __init__(self, config=DEFAULT_CONFIG, overrides=None):
        """
        初始化BasePredictor类。

        参数：
            config (str, optional): 配置文件的路径。默认为DEFAULT_CONFIG。
            overrides (dict, optional): 配置覆盖。默认为None。
        """
        if overrides is None:
            overrides = {}
        self.args = get_config(config, overrides)
        project = self.args.project or Path(SETTINGS['runs_dir']) / self.args.task
        name = self.args.name or f"{self.args.mode}"
        self.save_dir = increment_path(Path(project) / name, exist_ok=self.args.exist_ok)
        if self.args.save:
            (self.save_dir / 'labels' if self.args.save_txt else self.save_dir).mkdir(parents=True, exist_ok=True)
        if self.args.conf is None:
            self.args.conf = 0.25  # 默认conf=0.25
        self.done_setup = False
        
        
        # 如果设置对了可以使用
        self.model = None
        self.data = self.args.data  # data_dict
        self.device = None
        self.dataset = None
        self.vid_path, self.vid_writer = None, None
        self.annotator = None
        self.data_path = None
        self.callbacks = defaultdict(list, {k: [v] for k, v in callbacks.default_callbacks.items()})  # add callbacks
        #callbacks.add_integration_callbacks(self) 用不到就注释掉了

    
    def preprocess(self, img):
        pass

    def get_annotator(self, img):
        raise NotImplementedError("get_annotator function needs to be implemented")
     
    def write_results(self, pred, batch, print_string):
        raise NotImplementedError("print_results function needs to be implemented")

    def postprocess(self, preds, img, orig_img):
        return preds

    def setup(self, source=None, model=None):
    
        # 来源
        source = str(source if source is not None else self.args.source)
        is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
        is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
        webcam = source.isnumeric() or source.endswith('.streams') or (is_url and not is_file)
        screenshot = source.lower().startswith('screen')
        if is_url and is_file:
            source = check_file(source)  # 下载
        
        
        # 模型
        device = select_device(self.args.device)
        model = model or self.args.model
        self.args.half &= device.type != 'cpu'  # 半精度仅在CUDA上支持
        model = AutoBackend(model, device=device, dnn=self.args.dnn, fp16=self.args.half)
        stride, pt = model.stride, model.pt
        imgsz = check_imgsz(self.args.imgsz, stride=stride)  # 检查图像大小

        # 数据加载器
        bs = 1  # batch_size
        if self.args.show:
            self.args.show = check_imshow(warn=True)
        if webcam:
            self.dataset = LoadStreams(source,
                                       imgsz=imgsz,
                                       stride=stride,
                                       auto=pt,
                                       transforms=getattr(model.model, 'transforms', None),
                                       vid_stride=self.args.vid_stride)
            bs = len(self.dataset)
        elif screenshot:
            self.dataset = LoadScreenshots(source,
                                           imgsz=imgsz,
                                           stride=stride,
                                           auto=pt,
                                           transforms=getattr(model.model, 'transforms', None))
        else:
            self.dataset = LoadImages(source,
                                      imgsz=imgsz,
                                      stride=stride,
                                      auto=pt,
                                      transforms=getattr(model.model, 'transforms', None),
                                      vid_stride=self.args.vid_stride)
        self.vid_path, self.vid_writer = [None] * bs, [None] * bs
        model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # 预热

        self.model = model
        self.webcam = webcam
        self.screenshot = screenshot
        self.imgsz = imgsz
        self.done_setup = True
        self.device = device

        return model

    @smart_inference_mode()
    def __call__(self, source=None, model=None):
        
        self.run_callbacks("on_predict_start")
        model= self.model if self.done_setup else self.setup(source, model)
        model.eval()
        self.seen, self.windows, self.dt = 0, [], (ops.Profile(), ops.Profile(), ops.Profile())
        self.all_outputs = []
        for batch in self.dataset:
            self.run_callbacks("on_predict_batch_start")
            path, im, im0s, vid_cap, s = batch
            visualize = increment_path(self.save_dir / Path(path).stem, mkdir=True) if self.args.visualize else False
            with self.dt[0]:
                im = self.preprocess(im)
                if len(im.shape) == 3:
                    im = im[None]  # 添加批次元
            # 推理
            with self.dt[1]:
                preds = model(im, augment=self.args.augment, visualize=visualize)

            # 后处理
            with self.dt[2]:
                preds = self.postprocess(preds, im, im0s)

            for i in range(len(im)):
                if self.webcam:
                    path, im0s = path[i], im0s[i]
                p = Path(path)
                s += self.write_results(i, preds, (p, im, im0s))

                if self.args.show:
                    self.show(p)

                if self.args.save:
                    self.save_preds(vid_cap, i, str(self.save_dir / p.name))

            # 打印时间（仅需要推理時間）
            LOGGER.info(f"{s}{'' if len(preds) else '(没有检测到), '}{self.dt[1].dt * 1E3:.1f}ms")

            self.run_callbacks("on_predict_batch_end")

        # 打印结果
        t = tuple(x.t / self.seen * 1E3 for x in self.dt)  # 每个图像的速度
        LOGGER.info(
            f'速度: %.1fms预处理, %.1fms推理, %.1fms后处理, 图像形状 {(1, 3, *self.imgsz)}'
            % t)
        
        if self.args.save_txt or self.args.save:
            s = f"\n{len(list(self.save_dir.glob('labels/*.txt')))} 个标签已保存到 {self.save_dir / 'labels'}" if self.args.save_txt else ''
            LOGGER.info(f"结果已保存到 {colorstr('bold', self.save_dir)}{s}")

        self.run_callbacks("on_predict_end")
        return self.all_outputs

    def show(self, p):
        im0 = self.annotator.result()
        if platform.system() == 'Linux' and p not in self.windows:
            self.windows.append(p)
            cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # 允许窗口调整大小（Linux）
            cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
        cv2.imshow(str(p), im0)
        cv2.waitKey(1)  # 1 毫秒

    def save_preds(self, vid_cap, idx, save_path):
        im0 = self.annotator.result()
        # 保存图像
        if self.dataset.mode == 'image':
            cv2.imwrite(save_path, im0)
        else:  # '视频'或'流'
            if self.vid_path[idx] != save_path:  # 新视频
                self.vid_path[idx] = save_path
                if isinstance(self.vid_writer[idx], cv2.VideoWriter):
                    self.vid_writer[idx].release()  # 释放之前的视频作者
                if vid_cap:  # 视频
                    fps = vid_cap.get(cv2.CAP_PROP_FPS)
                    w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                else:  # 流
                    fps, w, h = 30, im0.shape[1], im0.shape[0]
                save_path = str(Path(save_path).with_suffix('.mp4'))  # 士制强制*.mp4后缀在结果视频上
                self.vid_writer[idx] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
            self.vid_writer[idx].write(im0)

    def run_callbacks(self, event: str):
        for callback in self.callbacks.get(event, []):
            callback(self)
