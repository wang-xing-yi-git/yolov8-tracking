# Ultralytics YOLO 🚀, GPL-3.0 license

import contextlib
from copy import deepcopy

import thop
import torch
import torch.nn as nn

from nn.modules import (C1, C2, C3, C3TR, SPP, SPPF, Bottleneck, BottleneckCSP, C2f, C3Ghost, C3x, Classify,
                                    Concat, Conv, ConvTranspose, Detect, DWConv, DWConvTranspose2d, Ensemble, Focus,
                                    GhostBottleneck, GhostConv, Segment)
from yolo.utils import DEFAULT_CONFIG_DICT, DEFAULT_CONFIG_KEYS, LOGGER, colorstr, yaml_load
from yolo.utils.checks import check_yaml
from yolo.utils.torch_utils import (fuse_conv_and_bn, initialize_weights, intersect_dicts, make_divisible,
                                                model_info, scale_img, time_sync)


class BaseModel(nn.Module):
    '''
     BaseModel简是一个Ultralytics YOLO性關中所有模型的基类。
    '''

    def forward(self, x, profile=False, visualize=False):
        """
        > `forward`是`_forward_once`的包装管，即在单个尺度上运行模型

        参数：
          x: 输入图像
          profile: 是否批次处理模型。默认为False
          visualize: 如果为True，将返回中闖特征每。默认为False

        返回：
          网络的输出。
        """
        return self._forward_once(x, profile, visualize)

    def _forward_once(self, x, profile=False, visualize=False):
        """
        > 网络的前向传播

        参数：
          x: 模式的输入
          profile: 如果为True，将打印每个层払洋的时间。默认为False
          visualize: 如果为True，它会保存模型的特征图。默认为False

        返回：
          模型的最后一层。
        """
        y, dt = [], []  # 输出
        for m in self.model:
            if m.f != -1:  # 如果不来自前一层
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # 来自提前的层
            if profile:
                self._profile_one_layer(m, x, dt)
            x = m(x)  # 运行
            y.append(x if m.i in self.save else None)  # 保存输出
            if visualize:
                pass
                # TODO: feature_visualization(x, m.type, m.i, save_dir=visualize)
        return x

    def _profile_one_layer(self, m, x, dt):
        """
        它伟了一个模型、一个输入和一个時間6我们列表，它批句模型上的输入
        时间到列表中。

        参数：
          m: 模型
          x: 输入图像
          dt: 每个层払洋的时间列表
        """
        c = m == self.model[-1]  # is final layer, copy input as inplace fix
        o = thop.profile(m, inputs=(x.copy() if c else x,), verbose=False)[0] / 1E9 * 2 if thop else 0  # FLOPs
        t = time_sync()
        for _ in range(10):
            m(x.copy() if c else x)
        dt.append((time_sync() - t) * 100)
        if m == self.model[0]:
            LOGGER.info(f"{'time (ms)':>10s} {'GFLOPs':>10s} {'params':>10s}  module")
        LOGGER.info(f'{dt[-1]:10.2f} {o:10.2f} {m.np:10.0f}  {m.type}')
        if c:
            LOGGER.info(f"{sum(dt):10.2f} {'-':>10s} {'-':>10s}  Total")

    def fuse(self):
        """
        > 它需要了一个模型，并将Conv2d()和BatchNorm2d()层融合成一个单一的层

        返回：
          返回模型。
        """
        LOGGER.info('融合层...')
        for m in self.model.modules():
            if isinstance(m, (Conv, DWConv)) and hasattr(m, 'bn'):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # 更新conv
                delattr(m, 'bn')  # 移除batchnorm
                m.forward = m.forward_fuse  # 更新前向
        self.info()
        return self

    def info(self, verbose=False, imgsz=640):
        """
        打印模式信息

        参数：
          verbose: 如果为True，打印模式信息。默认为False
          imgsz: 模型将要训练的图像的大小。默认为640
        """
        model_info(self, verbose, imgsz)

    def _apply(self, fn):
        """
        `_apply()`是一个函数0，可以将一个函数0应用于模型中的所有汁张
        参数或已注册缓冲区

        参数：
          fn: 要应用于模型的函数0

        返回：
          一个是检测()对象的模型。
        """
        self = super()._apply(fn)
        m = self.model[-1]  # Detect()
        if isinstance(m, (Detect, Segment)):
            m.stride = fn(m.stride)
            m.anchors = fn(m.anchors)
            m.strides = fn(m.strides)
        return self

    def load(self, weights):
        """
        > 此函数会从一个文件中加载模型的权重

        参数：
          weights: 要加载到模型中的权重。
        """
        # Force all tasks to implement this function
        raise NotImplementedError("This function needs to be implemented by derived classes!")


class DetectionModel(BaseModel):
    # YOLOv5検测模列
    def __init__(self, cfg='yolov8n.yaml', ch=3, nc=None, verbose=True):  # 模列、输入通道、等级个数
        super().__init__()
        self.yaml = cfg if isinstance(cfg, dict) else yaml_load(check_yaml(cfg), append_filename=True)  # cfg字典

        # 定义模列
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)  # 输入通道
        if nc and nc != self.yaml['nc']:
            LOGGER.info(f"打厚模式.yaml nc={self.yaml['nc']}为nc={nc}")
            self.yaml['nc'] = nc  # 覆盖了yaml值
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch], verbose=verbose)  # 模列、保存列表
        self.names = {i: f'{i}' for i in range(self.yaml['nc'])}  # 默认名称字典
        self.inplace = self.yaml.get('inplace', True)

        # 构次步數倏
        m = self.model[-1]  # 検测()
        if isinstance(m, (Detect, Segment)):
            s = 256  # 2x的最小步整
            m.inplace = self.inplace
            forward = lambda x: self.forward(x)[0] if isinstance(m, Segment) else self.forward(x)
            m.stride = torch.tensor([s / x.shape[-2] for x in forward(torch.zeros(1, ch, s, s))])  # 前向
            self.stride = m.stride
            m.bias_init()  # 仅执行一次

        # 初始化权重、偏不
        initialize_weights(self)
        if verbose:
            self.info()
            LOGGER.info('')

    def forward(self, x, augment=False, profile=False, visualize=False):
        if augment:
            return self._forward_augment(x)  # 增强了推理，无
        return self._forward_once(x, profile, visualize)  # 单一例業推理、训练

    def _forward_augment(self, x):
        img_size = x.shape[-2:]  # 高度、宽度
        s = [1, 0.83, 0.67]  # 不同比例
        f = [None, 3, None]  # 翰转 (2-ud, 3-lr)
        y = []  # 输出
        for si, fi in zip(s, f):
            xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
            yi = self._forward_once(xi)[0]  # 前向
            # cv2.imwrite(f'img_{si}.jpg', 255 * xi[0].cpu().numpy().transpose((1, 2, 0))[:, :, ::-1])  # 保存
            yi = self._descale_pred(yi, fi, si, img_size)
            y.append(yi)
        y = self._clip_augmented(y)  # 剪檻增幼高地尾已
        return torch.cat(y, -1), None  # 增强了推理、训练

    @staticmethod
    def _descale_pred(p, flips, scale, img_size, dim=1):
        # 根据增强了的推理反转遄下扶鸡(反向操作)
        p[:, :4] /= scale  # 反视化
        x, y, wh, cls = p.split((1, 1, 2, p.shape[dim] - 4), dim)
        if flips == 2:
            y = img_size[0] - y  # 反转 ud
        elif flips == 3:
            x = img_size[1] - x  # 反转 lr
        return torch.cat((x, y, wh, cls), dim)

    def _clip_augmented(self, y):
        # 剪此徏YOLOv5事业增幼尾
        nl = self.model[-1].nl  # 検测层数(P3-P5)
        g = sum(4 ** x for x in range(nl))  # 网格点
        e = 1  # 除去层数
        i = (y[0].shape[-1] // g) * sum(4 ** x for x in range(e))  # 索引
        y[0] = y[0][..., :-i]  # 大
        i = (y[-1].shape[-1] // g) * sum(4 ** (nl - 1 - x) for x in range(e))  # 索引
        y[-1] = y[-1][..., i:]  # 小
        return y

    def load(self, weights, verbose=True):
        csd = weights.float().state_dict()  # 检骮FP32中皋然低基准事业
        csd = intersect_dicts(csd, self.state_dict())  # 相交
        self.load_state_dict(csd, strict=False)  # 加载
        if verbose:
            LOGGER.info(f'介入了预训练烨重中司{len(csd)}/{len(self.model.state_dict())}个項目')


class SegmentationModel(DetectionModel):
    # YOLOv5分割模列
    def __init__(self, cfg='yolov8n-seg.yaml', ch=3, nc=None, verbose=True):
        super().__init__(cfg, ch, nc, verbose)


class ClassificationModel(BaseModel):
    # YOLOv5分签模列
    def __init__(self,
                 cfg=None,
                 model=None,
                 ch=3,
                 nc=1000,
                 cutoff=10,
                 verbose=True):  # yaml、模列、砂数、架技批次
        super().__init__()
        self._from_detection_model(model, nc, cutoff) if model is not None else self._from_yaml(cfg, ch, nc, verbose)

    def _from_detection_model(self, model, nc=1000, cutoff=10):
        # 弶一个YOLOv5検测模型潭索一个填坡分签模型
        from nn.autobackend import AutoBackend
        if isinstance(model, AutoBackend):
            model = model.model  # 拆店DetectMultiBackend
        model.model = model.model[:cutoff]  # 主东
        m = model.model[-1]  # 最后一层
        ch = m.conv.in_channels if hasattr(m, 'conv') else m.cv1.conv.in_channels  # 截至模块的ch
        c = Classify(ch, nc)  # 分类()
        c.i, c.f, c.type = m.i, m.f, 'models.common.Classify'  # 索引、来自、類別
        model.model[-1] = c  # 決予
        self.model = model.model
        self.stride = model.stride
        self.save = []
        self.nc = nc

    def _from_yaml(self, cfg, ch, nc, verbose):
        self.yaml = cfg if isinstance(cfg, dict) else yaml_load(check_yaml(cfg), append_filename=True)  # cfg dict
        # Define model
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)  # input channels
        if nc and nc != self.yaml['nc']:
            LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml['nc'] = nc  # override yaml value
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch], verbose=verbose)  # model, savelist
        self.names = {i: f'{i}' for i in range(self.yaml['nc'])}  # default names dict
        self.info()

    def load(self, weights):
        model = weights["model"] if isinstance(weights, dict) else weights  # torchvision models are not dicts
        csd = model.float().state_dict()
        csd = intersect_dicts(csd, self.state_dict())  # intersect
        self.load_state_dict(csd, strict=False)  # load

    @staticmethod
    def reshape_outputs(model, nc):
        # Update a TorchVision classification model to class count 'n' if required
        name, m = list((model.model if hasattr(model, 'model') else model).named_children())[-1]  # last module
        if isinstance(m, Classify):  # YOLO Classify() head
            if m.linear.out_features != nc:
                m.linear = nn.Linear(m.linear.in_features, nc)
        elif isinstance(m, nn.Linear):  # ResNet, EfficientNet
            if m.out_features != nc:
                setattr(model, name, nn.Linear(m.in_features, nc))
        elif isinstance(m, nn.Sequential):
            types = [type(x) for x in m]
            if nn.Linear in types:
                i = types.index(nn.Linear)  # nn.Linear index
                if m[i].out_features != nc:
                    m[i] = nn.Linear(m[i].in_features, nc)
            elif nn.Conv2d in types:
                i = types.index(nn.Conv2d)  # nn.Conv2d index
                if m[i].out_channels != nc:
                    m[i] = nn.Conv2d(m[i].in_channels, nc, m[i].kernel_size, m[i].stride, bias=m[i].bias is not None)


# Functions ------------------------------------------------------------------------------------------------------------


def attempt_load_weights(weights, device=None, inplace=True, fuse=False):
    # Loads an ensemble of models weights=[a,b,c] or a single model weights=[a] or weights=a
    from yolo.utils.downloads import attempt_download

    model = Ensemble()
    for w in weights if isinstance(weights, list) else [weights]:
        ckpt = torch.load(attempt_download(w), map_location='cpu')  # load，高版本torch报错，改成低版本
        args = {**DEFAULT_CONFIG_DICT, **ckpt['train_args']}  # combine model and default args, preferring model args
        ckpt = (ckpt.get('ema') or ckpt['model']).to(device).float()  # FP32 model

        # Model compatibility updates
        ckpt.args = {k: v for k, v in args.items() if k in DEFAULT_CONFIG_KEYS}  # attach args to model
        ckpt.pt_path = weights  # attach *.pt file path to model
        if not hasattr(ckpt, 'stride'):
            ckpt.stride = torch.tensor([32.])

        # Append
        model.append(ckpt.fuse().eval() if fuse and hasattr(ckpt, 'fuse') else ckpt.eval())  # model in eval mode

    # Module compatibility updates
    for m in model.modules():
        t = type(m)
        if t in (nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU, Detect, Segment):
            m.inplace = inplace  # torch 1.7.0 compatibility
        elif t is nn.Upsample and not hasattr(m, 'recompute_scale_factor'):
            m.recompute_scale_factor = None  # torch 1.11.0 compatibility

    # Return model
    if len(model) == 1:
        return model[-1]

    # Return ensemble
    print(f'Ensemble created with {weights}\n')
    for k in 'names', 'nc', 'yaml':
        setattr(model, k, getattr(model[0], k))
    model.stride = model[torch.argmax(torch.tensor([m.stride.max() for m in model])).int()].stride  # max stride
    assert all(model[0].nc == m.nc for m in model), f'Models have different class counts: {[m.nc for m in model]}'
    return model


def attempt_load_one_weight(weight, device=None, inplace=True, fuse=False):
    # Loads a single model weights
    from yolo.utils.downloads import attempt_download

    ckpt = torch.load(attempt_download(weight), map_location='cpu')  # load
    args = {**DEFAULT_CONFIG_DICT, **ckpt['train_args']}  # combine model and default args, preferring model args
    model = (ckpt.get('ema') or ckpt['model']).to(device).float()  # FP32 model

    # Model compatibility updates
    model.args = {k: v for k, v in args.items() if k in DEFAULT_CONFIG_KEYS}  # attach args to model
    model.pt_path = weight  # attach *.pt file path to model
    if not hasattr(model, 'stride'):
        model.stride = torch.tensor([32.])

    model = model.fuse().eval() if fuse and hasattr(model, 'fuse') else model.eval()  # model in eval mode

    # Module compatibility updates
    for m in model.modules():
        t = type(m)
        if t in (nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU, Detect, Segment):
            m.inplace = inplace  # torch 1.7.0 compatibility
        elif t is nn.Upsample and not hasattr(m, 'recompute_scale_factor'):
            m.recompute_scale_factor = None  # torch 1.11.0 compatibility

    # Return model and ckpt
    return model, ckpt


def parse_model(d, ch, verbose=True):  # model_dict, input_channels(3)
    # Parse a YOLO model.yaml dictionary
    if verbose:
        LOGGER.info(f"\n{'':>3}{'from':>20}{'n':>3}{'params':>10}  {'module':<45}{'arguments':<30}")
    nc, gd, gw, act = d['nc'], d['depth_multiple'], d['width_multiple'], d.get('activation')
    if act:
        Conv.default_act = eval(act)  # redefine default activation, i.e. Conv.default_act = nn.SiLU()
        if verbose:
            LOGGER.info(f"{colorstr('activation:')} {act}")  # print

    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):  # from, number, module, args
        m = eval(m) if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            with contextlib.suppress(NameError):
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings

        n = n_ = max(round(n * gd), 1) if n > 1 else n  # depth gain
        if m in {
                Classify, Conv, ConvTranspose, GhostConv, Bottleneck, GhostBottleneck, SPP, SPPF, DWConv, Focus,
                BottleneckCSP, C1, C2, C2f, C3, C3TR, C3Ghost, nn.ConvTranspose2d, DWConvTranspose2d, C3x}:
            c1, c2 = ch[f], args[0]
            if c2 != nc:  # if c2 not equal to number of classes (i.e. for Classify() output)
                c2 = make_divisible(c2 * gw, 8)

            args = [c1, c2, *args[1:]]
            if m in {BottleneckCSP, C1, C2, C2f, C3, C3TR, C3Ghost, C3x}:
                args.insert(2, n)  # number of repeats
                n = 1
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum(ch[x] for x in f)
        elif m in {Detect, Segment}:
            args.append([ch[x] for x in f])
            if m is Segment:
                args[2] = make_divisible(args[2] * gw, 8)
        else:
            c2 = ch[f]

        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace('__main__.', '')  # module type
        m.np = sum(x.numel() for x in m_.parameters())  # number params
        m_.i, m_.f, m_.type = i, f, t  # attach index, 'from' index, type
        if verbose:
            LOGGER.info(f'{i:>3}{str(f):>20}{n_:>3}{m.np:10.0f}  {t:<45}{str(args):<30}')  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)
