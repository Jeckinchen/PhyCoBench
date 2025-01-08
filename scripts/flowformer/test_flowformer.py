import sys
import os
# 获取当前脚本目录的上级目录
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, "../.."))
print("root_dir:", root_dir)
sys.path.append(root_dir)

from PIL import Image
from glob import glob
import argparse
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from flowformer.configs.submissions import get_cfg
from flowformer.utils.misc import process_cfg

from flowformer.utils import flow_viz
from flowformer.utils import frame_utils
import cv2
import math
import os.path as osp

from flowformer import build_flowformer

from flowformer.utils.utils import InputPadder, forward_interpolate
import itertools

from matplotlib.collections import LineCollection
from scipy.ndimage import zoom

TRAIN_SIZE = [432, 960]
RESULTS_DIR = "/data/oss_bucket_0/yongfan/flow_crafter/flow_vis_test"


def compute_grid_indices(image_shape, patch_size=TRAIN_SIZE, min_overlap=20):
  if min_overlap >= TRAIN_SIZE[0] or min_overlap >= TRAIN_SIZE[1]:
    raise ValueError(
        f"Overlap should be less than size of patch (got {min_overlap}"
        f"for patch size {patch_size}).")
  if image_shape[0] == TRAIN_SIZE[0]:
    hs = list(range(0, image_shape[0], TRAIN_SIZE[0]))
  else:
    hs = list(range(0, image_shape[0], TRAIN_SIZE[0] - min_overlap))
  if image_shape[1] == TRAIN_SIZE[1]:
    ws = list(range(0, image_shape[1], TRAIN_SIZE[1]))
  else:
    ws = list(range(0, image_shape[1], TRAIN_SIZE[1] - min_overlap))

  # Make sure the final patch is flush with the image boundary
  hs[-1] = image_shape[0] - patch_size[0]
  ws[-1] = image_shape[1] - patch_size[1]
  return [(h, w) for h in hs for w in ws]

def compute_weight(hws, image_shape, patch_size=TRAIN_SIZE, sigma=1.0, wtype='gaussian'):
    patch_num = len(hws)
    h, w = torch.meshgrid(torch.arange(patch_size[0]), torch.arange(patch_size[1]))
    h, w = h / float(patch_size[0]), w / float(patch_size[1])
    c_h, c_w = 0.5, 0.5 
    h, w = h - c_h, w - c_w
    weights_hw = (h ** 2 + w ** 2) ** 0.5 / sigma
    denorm = 1 / (sigma * math.sqrt(2 * math.pi))
    weights_hw = denorm * torch.exp(-0.5 * (weights_hw) ** 2)

    weights = torch.zeros(1, patch_num, *image_shape)
    for idx, (h, w) in enumerate(hws):
        weights[:, idx, h:h+patch_size[0], w:w+patch_size[1]] = weights_hw
    weights = weights.cuda()
    patch_weights = []
    for idx, (h, w) in enumerate(hws):
        patch_weights.append(weights[:, idx:idx+1, h:h+patch_size[0], w:w+patch_size[1]])

    return patch_weights

def compute_flow(model, image1, image2, weights=None):
    print(f"computing flow...")

    image_size = image1.shape[1:]

    image1, image2 = image1[None].cuda(), image2[None].cuda()

    hws = compute_grid_indices(image_size)
    if weights is None:     # no tile
        padder = InputPadder(image1.shape)
        image1, image2 = padder.pad(image1, image2)

        flow_pre, _ = model(image1, image2)

        flow_pre = padder.unpad(flow_pre)
        flow = flow_pre[0].permute(1, 2, 0).cpu().numpy()
    else:                   # tile
        flows = 0
        flow_count = 0

        for idx, (h, w) in enumerate(hws):
            image1_tile = image1[:, :, h:h+TRAIN_SIZE[0], w:w+TRAIN_SIZE[1]]
            image2_tile = image2[:, :, h:h+TRAIN_SIZE[0], w:w+TRAIN_SIZE[1]]    
            flow_pre, _ = model(image1_tile, image2_tile)
            padding = (w, image_size[1]-w-TRAIN_SIZE[1], h, image_size[0]-h-TRAIN_SIZE[0], 0, 0)
            flows += F.pad(flow_pre * weights[idx], padding)
            flow_count += F.pad(weights[idx], padding)

        flow_pre = flows / flow_count
        flow = flow_pre[0].permute(1, 2, 0).cpu().numpy()

    return flow

def compute_adaptive_image_size(image_size):
    target_size = TRAIN_SIZE
    scale0 = target_size[0] / image_size[0]
    scale1 = target_size[1] / image_size[1] 

    if scale0 > scale1:
        scale = scale0
    else:
        scale = scale1

    image_size = (int(image_size[1] * scale), int(image_size[0] * scale))

    return image_size

def prepare_image(root_dir, viz_root_dir, fn1, fn2, keep_size, seq_mode, vis_mode):
    print(f"preparing image...")
    #print(f"root dir = {root_dir}, fn = {fn1}")

    image1 = frame_utils.read_gen(osp.join(root_dir, fn1))
    image2 = frame_utils.read_gen(osp.join(root_dir, fn2))
    image1 = np.array(image1).astype(np.uint8)[..., :3]
    image2 = np.array(image2).astype(np.uint8)[..., :3]
    if not keep_size:
        dsize = compute_adaptive_image_size(image1.shape[0:2])
        image1 = cv2.resize(image1, dsize=dsize, interpolation=cv2.INTER_CUBIC)
        image2 = cv2.resize(image2, dsize=dsize, interpolation=cv2.INTER_CUBIC)
    image1 = torch.from_numpy(image1).permute(2, 0, 1).float()
    image2 = torch.from_numpy(image2).permute(2, 0, 1).float()

    print("=========>image1 shape:", image1.shape)
    print("=========>image2 shape:", image2.shape)

    dirname = osp.dirname(fn1)
    filename = osp.splitext(osp.basename(fn2))[0]

    viz_dir = osp.join(viz_root_dir, dirname)
    if not osp.exists(viz_dir):
        os.makedirs(viz_dir)
    save_dir = os.path.join(RESULTS_DIR, seq_mode, vis_mode)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    viz_fn = osp.join(save_dir, filename + '.png')

    return image1, image2, viz_fn

def build_model():
    print(f"building  model...")
    cfg = get_cfg()
    model = torch.nn.DataParallel(build_flowformer(cfg))
    model.load_state_dict(torch.load(cfg.model))

    model.cuda()
    model.eval()

    return model

def plot_grid(x,y, ax=None, **kwargs):
    ax = ax or plt.gca()
    segs1 = np.stack((x,y), axis=2)
    segs2 = segs1.transpose(1,0,2)
    ax.add_collection(LineCollection(segs1, **kwargs))
    ax.add_collection(LineCollection(segs2, **kwargs))
    ax.autoscale()
    ax.invert_yaxis()  # y轴反向

def show_flow_grid(flow, img=None, step=15, save_path=""):
    '''
    :param flow:
    :param img:
    :param step:
    :return:
    '''
    h, w = flow.shape[:2]
    plt.figure()
    if img is not None:
        plt.imshow(img)

    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
    plot_grid(grid_x[::step, ::step], grid_y[::step, ::step], color="lightgrey")
    grid_x2 = grid_x + flow[..., 0]
    grid_y2 = grid_y + flow[..., 1]

    plot_grid(grid_x2[::step, ::step], grid_y2[::step, ::step], color="C0")
    
    plt.axis('off')  # 关闭坐标轴
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def downsample_optical_flow(flow, target_size=(64, 64)):
    """
    使用双线性插值将光流图像从原始分辨率下采样到目标分辨率，
    并相应地缩放位移量。
    
    参数:
        flow (numpy.ndarray): 原始光流图像，形状为 (H, W, 2)。
        target_size (tuple): 目标分辨率，形状为 (target_height, target_width)。
        
    返回:
        numpy.ndarray: 下采样后的光流图像，形状为 (target_height, target_width, 2)。
    """
    # 确保输入是一个 3D 数组 (H, W, 2)
    if flow.ndim != 3 or flow.shape[2] != 2:
        raise ValueError("光流图像的形状应该是 (H, W, 2)。")
    
    # 获取原始图像的尺寸
    original_height, original_width = flow.shape[:2]
    
    # 计算缩放因子
    height_scale = target_size[0] / original_height
    width_scale = target_size[1] / original_width
    
    # 对光流图像的每个通道进行下采样
    downsampled_flow = np.zeros((target_size[0], target_size[1], 2), dtype=flow.dtype)
    
    for i in range(2):  # 遍历光流的两个通道
        downsampled_flow[:, :, i] = zoom(flow[:, :, i], (height_scale, width_scale), order=1)
    
    # 按比例缩放光流的位移量
    downsampled_flow[:, :, 0] *= width_scale  # 缩放x方向位移
    downsampled_flow[:, :, 1] *= height_scale  # 缩放y方向位移
    
    return downsampled_flow

def visualize_flow(root_dir, viz_root_dir, model, img_pairs, keep_size, seq_mode, vis_mode):
    weights = None
    for img_pair in img_pairs:
        fn1, fn2 = img_pair
        print(f"processing {fn1}, {fn2}...")

        image1, image2, viz_fn = prepare_image(root_dir, viz_root_dir, fn1, fn2, keep_size, seq_mode, vis_mode)
        flow = compute_flow(model, image1, image2, weights)
        downsampled_flow = downsample_optical_flow(flow)
        print(f"downsampled_flow shape:{downsampled_flow.shape}")
        if vis_mode == "grid":
            show_flow_grid(flow, save_path=viz_fn)
        elif vis_mode == "rgb":
            #print(f"flow shape:{flow.shape[:2]}")
            flow_img = flow_viz.flow_to_image(flow)
            success = cv2.imwrite(viz_fn, flow_img[:, :, [2,1,0]])
            if not success:
                print(f">>>>>>>>>>>>>>>>>Failed to save image to {viz_fn}")
            else:
                print(f">>>>>>>>>>>>>>>>>saved to {viz_fn}")

def generate_pairs(dirname, start_idx, end_idx, seq_mode):
    img_pairs = []
    for idx in range(start_idx, end_idx):
        if seq_mode == "oneone":
            img1 = osp.join(dirname, f'fall_cardboard_09_foam_01_Camera_1_{idx:03}.png')
            img2 = osp.join(dirname, f'fall_cardboard_09_foam_01_Camera_1_{idx+1:03}.png')
        elif seq_mode == "fromstart":
            img1 = osp.join(dirname, f'fall_cardboard_09_foam_01_Camera_1_001.png')
            img2 = osp.join(dirname, f'fall_cardboard_09_foam_01_Camera_1_{idx+1:03}.png')
        # img1 = f'{idx:06}.png'
        # img2 = f'{idx+1:06}.png'
        img_pairs.append((img1, img2))

    return img_pairs

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', default='.')
    parser.add_argument('--sintel_dir', default='datasets/Sintel/test/clean')
    parser.add_argument('--seq_dir', default='/data/oss_bucket_0/yongfan/flow_crafter/flow_test_videos/fall_cardboard_09_foam_01_Camera_1')
    parser.add_argument('--start_idx', type=int, default=1)     # starting index of the image sequence
    parser.add_argument('--end_idx', type=int, default=28)    # ending index of the image sequence
    parser.add_argument('--viz_root_dir', default='viz_results')
    parser.add_argument('--keep_size', action='store_true')     # keep the image size, or the image will be adaptively resized.
    parser.add_argument('--seq_mode', type=str, default="oneone", choices=["fromstart", "oneone"]) #pair是从0到后续每一帧还是两两帧 fromstart, oneone
    parser.add_argument('--vis_mode', type=str, default="grid", choices=["rgb", "grid"]) #用rgb 或者 grid可视化光流
    args = parser.parse_args()

    root_dir = args.root_dir
    viz_root_dir = args.viz_root_dir

    model = build_model()


    img_pairs = generate_pairs(args.seq_dir, args.start_idx, args.end_idx, args.seq_mode)
    #print("img_pairs:", img_pairs)
    with torch.no_grad():
        visualize_flow(root_dir, viz_root_dir, model, img_pairs, args.keep_size, args.seq_mode, args.vis_mode)
        #visualize_flow(root_dir, viz_root_dir, model, [img_pairs[0]], args.keep_size, args.seq_mode, args.vis_mode)