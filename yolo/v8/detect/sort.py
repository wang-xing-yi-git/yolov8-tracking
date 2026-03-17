from __future__ import print_function

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage import io

import glob
import time
import argparse
from filterpy.kalman import KalmanFilter

np.random.seed(0)

def linear_assignment(cost_matrix):
    try:
        import lap #linear assignment problem solver
        _, x, y = lap.lapjv(cost_matrix, extend_cost = True)
        return np.array([[y[i],i] for i in x if i>=0])
    except ImportError:
        from scipy.optimize import linear_sum_assignment
        x,y = linear_sum_assignment(cost_matrix)
        return np.array(list(zip(x,y)))


"""From SORT: Computes IOU between two boxes in the form [x1,y1,x2,y2]"""
def iou_batch(bb_test, bb_gt):
    
    bb_gt = np.expand_dims(bb_gt, 0)
    bb_test = np.expand_dims(bb_test, 1)
    
    xx1 = np.maximum(bb_test[...,0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])                                      
    + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) - wh)
    return(o)


"""Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form [x,y,s,r] where x,y is the center of the box and s is the scale/area and r is the aspect ratio"""
def convert_bbox_to_z(bbox):
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w/2.
    y = bbox[1] + h/2.
    s = w * h    
    #scale is just area
    r = w / float(h)
    return np.array([x, y, s, r]).reshape((4, 1))


"""将边界框从中心形式[x,y,s,r]转换为[x1,y1,x2,y2]格式
    其中x1,y1是左上角坐标，x2,y2是右下角坐标"""
def convert_x_to_bbox(x, score=None):
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    if(score==None):
        return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.]).reshape((1,4))
    else:
        return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.,score]).reshape((1,5))

"""此类表示被观察为边界框的单个被跟踪对象的内部状态"""
class KalmanBoxTracker(object):
    
    count = 0
    def __init__(self, bbox):
        """
        使用初始边界框初始化跟踪器
        
        参数'bbox'必须在-1位置具有'检测到的类别'整数
        """
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([[1,0,0,0,1,0,0],[0,1,0,0,0,1,0],[0,0,1,0,0,0,1],[0,0,0,1,0,0,0],[0,0,0,0,1,0,0],[0,0,0,0,0,1,0],[0,0,0,0,0,0,1]])
        self.kf.H = np.array([[1,0,0,0,0,0,0],[0,1,0,0,0,0,0],[0,0,1,0,0,0,0],[0,0,0,1,0,0,0]])

        self.kf.R[2:,2:] *= 10.  # R: 测量噪声的协方差矩阵（对于有噪声的输入设置为高值 -> 更多框的'惯性')
        self.kf.P[4:,4:] *= 1000.  # 给予观测不到的初始速度高不确定性
        self.kf.P *= 10.
        self.kf.Q[-1,-1] *= 0.5  # Q: 过程噪声的协方差矩阵（对于运动不规则的物体设置为高值）
        self.kf.Q[4:,4:] *= 0.5

        self.kf.x[:4] = convert_bbox_to_z(bbox)  # 状态向量
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        self.centroidarr = []
        CX = (bbox[0]+bbox[2])//2
        CY = (bbox[1]+bbox[3])//2
        self.centroidarr.append((CX,CY))
        
        # 保留yolov5检测到的类别信息
        self.detclass = bbox[5]

        # 如果我们想存储边界框
        self.bbox_history = [bbox]
        
    def update(self, bbox):
        """
        用观察到的边界框更新状态向量
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(convert_bbox_to_z(bbox))
        self.detclass = bbox[5]
        CX = (bbox[0]+bbox[2])//2
        CY = (bbox[1]+bbox[3])//2
        self.centroidarr.append((CX,CY))
        self.bbox_history.append(bbox)
    
    def predict(self):
        """
        推进状态向量并返回预测的边界框估计
        """
        if((self.kf.x[6]+self.kf.x[2])<=0):
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if(self.time_since_update>0):
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.kf.x))
        # bbox=self.history[-1]
        # CX = (bbox[0]+bbox[2])/2
        # CY = (bbox[1]+bbox[3])/2
        # self.centroidarr.append((CX,CY))
        
        return self.history[-1]
    
    
    def get_state(self):
        """
        \u8fd4\u56de\u5f53\u524d\u8fb9\u754c\u6846\u4f30\u8ba1
        # test
        arr1 = np.array([[1,2,3,4]])
        arr2 = np.array([0])
        arr3 = np.expand_dims(arr2, 0)
        np.concatenate((arr1,arr3), axis=1)
        """
        arr_detclass = np.expand_dims(np.array([self.detclass]), 0)
        
        arr_u_dot = np.expand_dims(self.kf.x[4],0)
        arr_v_dot = np.expand_dims(self.kf.x[5],0)
        arr_s_dot = np.expand_dims(self.kf.x[6],0)
        
        return np.concatenate((convert_x_to_bbox(self.kf.x), arr_detclass, arr_u_dot, arr_v_dot, arr_s_dot), axis=1)
    
def associate_detections_to_trackers(detections, trackers, iou_threshold = 0.3):
    """
    将检测分配给跟踪的对象（都表示为边界框）
    返回3个列表：
    1. 匹配对
    2. 未匹配的检测
    3. 未匹配的跟踪器
    """
    if(len(trackers)==0):
        return np.empty((0,2),dtype=int), np.arange(len(detections)), np.empty((0,5),dtype=int)
    
    iou_matrix = iou_batch(detections, trackers)
    
    if min(iou_matrix.shape) > 0:
        a = (iou_matrix > iou_threshold).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() ==1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            matched_indices = linear_assignment(-iou_matrix)
    else:
        matched_indices = np.empty(shape=(0,2))
    
    unmatched_detections = []
    for d, det in enumerate(detections):
        if(d not in matched_indices[:,0]):
            unmatched_detections.append(d)
    
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if(t not in matched_indices[:,1]):
            unmatched_trackers.append(t)
    
    # 过滤掉IOU低的匹配
    matches = []
    for m in matched_indices:
        if(iou_matrix[m[0], m[1]]<iou_threshold):
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1,2))
    
    if(len(matches)==0):
        matches = np.empty((0,2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)
        
    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)
    

class Sort(object):
    def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3):
        """
        SORT算法的参数
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0
    def getTrackers(self,):
        return self.trackers
        
    def update(self, dets= np.empty((0,6))):
        """
        参数：
        'dets' - 检测结果的numpy数组，格式为[[x1, y1, x2, y2, score], [x1,y1,x2,y2,score],...]
        
        即使帧中没有检测到对象也要调用此方法（传入np.empty((0,5))）
        
        返回类似的数组，其中最后一列是对象ID（替换置信度分数）
        
        注意：返回的对象数量可能与提供的对象数量不同
        """
        self.frame_count += 1
        
        # 从现有跟踪器获取预测位置
        trks = np.zeros((len(self.trackers), 6))
        to_del = []
        ret = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0, 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)
        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets, trks, self.iou_threshold)
        
        # 用分配的检测更新匹配的跟踪器
        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :])
            
        # 为未匹配的检测创建并初始化新的跟踪器
        for i in unmatched_dets:
            trk = KalmanBoxTracker(np.hstack((dets[i,:], np.array([0]))))
            # trk = KalmanBoxTracker(np.hstack(dets[i,:]))
            self.trackers.append(trk)
        
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()[0]
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append(np.concatenate((d, [trk.id+1])).reshape(1,-1))  # +1是因为MOT基准要求正值
            i -= 1
            # 删除死亡的轨迹
            if(trk.time_since_update >self.max_age):
                self.trackers.pop(i)
        if(len(ret) > 0):
            return np.concatenate(ret)
        return np.empty((0,6))

def parse_args():
    """解析命令行输入参数"""
    parser = argparse.ArgumentParser(description='SORT demo')
    parser.add_argument('--display', dest='display', help='Display online tracker output (slow) [False]',action='store_true')
    parser.add_argument("--seq_path", help="Path to detections.", type=str, default='data')
    parser.add_argument("--phase", help="Subdirectory in seq_path.", type=str, default='train')
    parser.add_argument("--max_age", 
                        help="不带关联检测的轨迹保活的最大帧数", 
                        type=int, default=1)
    parser.add_argument("--min_hits", 
                        help="轨迹初始化前的最小关联检测数", 
                        type=int, default=3)
    parser.add_argument("--iou_threshold", help="匹配的最小IOU", type=float, default=0.3)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    # 处理所有训练数据
    args = parse_args()
    display = args.display
    phase = args.phase
    total_time = 0.0
    total_frames = 0
    colours = np.random.rand(32, 3)  # 仅用于显示
    if(display):
        if not os.path.exists('mot_benchmark'):
            print('\n\t错误：mot_benchmark链接未找到!\n\n    创建指向MOT基准的符号链接\n    (https://motchallenge.net/data/2D_MOT_2015/#download). 例如:\n\n    $ ln -s /path/to/MOT2015_challenge/2DMOT2015 mot_benchmark\n\n')
        exit()
    plt.ion()
    fig = plt.figure()
    ax1 = fig.add_subplot(111, aspect='equal')

    if not os.path.exists('output'):
        os.makedirs('output')
    pattern = os.path.join(args.seq_path, phase, '*', 'det', 'det.txt')
    for seq_dets_fn in glob.glob(pattern):
        mot_tracker = Sort(max_age=args.max_age, 
                   min_hits=args.min_hits,
                   iou_threshold=args.iou_threshold)  # 创建SORT跟踪器实例
    seq_dets = np.loadtxt(seq_dets_fn, delimiter=',')
    seq = seq_dets_fn[pattern.find('*'):].split(os.path.sep)[0]
    
    with open(os.path.join('output', '%s.txt'%(seq)),'w') as out_file:
        print("处理 %s."%(seq))
        for frame in range(int(seq_dets[:,0].max())):
            frame += 1  # 检测和帧号从1开始
            dets = seq_dets[seq_dets[:, 0]==frame, 2:7]
            dets[:, 2:4] += dets[:, 0:2]  # 将[x1,y1,w,h]转换为[x1,y1,x2,y2]
            total_frames += 1

        if(display):
            fn = os.path.join('mot_benchmark', phase, seq, 'img1', '%06d.jpg'%(frame))
            im =io.imread(fn)
            ax1.imshow(im)
            plt.title(seq + ' 跟踪目标')

        start_time = time.time()
        trackers = mot_tracker.update(dets)
        cycle_time = time.time() - start_time
        total_time += cycle_time

        for d in trackers:
            print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1'%(frame,d[4],d[0],d[1],d[2]-d[0],d[3]-d[1]),file=out_file)
            if(display):
                d = d.astype(np.int32)
                ax1.add_patch(patches.Rectangle((d[0],d[1]),d[2]-d[0],d[3]-d[1],fill=False,lw=3,ec=colours[d[4]%32,:]))

        if(display):
            fig.canvas.flush_events()
            plt.draw()
            ax1.cla()

    print("总跟踪时间：%.3f 秒，共 %d 帧，帧率 %.1f FPS" % (total_time, total_frames, total_frames / total_time))

    if(display):
        print("注意：为了获得真实的运行时结果，请运行不带 --display 选项的命令")
