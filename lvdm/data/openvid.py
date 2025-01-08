import os
import random
from tqdm import tqdm
import pandas as pd
from decord import VideoReader, cpu

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import functional as F
from PIL import Image


def resize_and_pad_frames(frames, size):
    """
    Resize and pad frames to the target size while maintaining aspect ratio.
    Args:
        frames: Tensor of shape [C, T, H, W]
        size: Tuple or list [H_out, W_out]
    Returns:
        Tensor of shape [C, T, H_out, W_out]
    """
    C, T, H, W = frames.shape
    target_h, target_w = size
    frames_resized_padded = []
    for t in range(T):
        frame = frames[:, t, :, :]  # [C, H, W]
        h, w = frame.shape[1], frame.shape[2]
        # Compute the scaling factor
        scale = min(target_h / h, target_w / w)
        new_h = int(h * scale)
        new_w = int(w * scale)
        # Resize the frame
        frame_resized = F.resize(frame, [new_h, new_w], antialias=True)
        # Calculate padding
        pad_left = (target_w - new_w) // 2
        pad_right = target_w - new_w - pad_left
        pad_top = (target_h - new_h) // 2
        pad_bottom = target_h - new_h - pad_top
        padding = [pad_left, pad_top, pad_right, pad_bottom]  # Left, Top, Right, Bottom

        # Pad the frame
        frame_padded = F.pad(frame_resized, padding)
        frames_resized_padded.append(frame_padded)
    frames_resized_padded = torch.stack(frames_resized_padded, dim=1)  # [C, T, H_out, W_out]
    return frames_resized_padded


class ResizeAndPad(object):
    """Custom transform to resize and pad frames."""
    def __init__(self, size):
        self.size = size  # [H_out, W_out]

    def __call__(self, frames):
        return resize_and_pad_frames(frames, self.size)


class OpenVid(Dataset):
    """
    OpenVid Dataset.
    Assumes data is structured as follows.
    OpenVid/
        fall_cardboard_01_foam_01_Camera_1.mp4           (videoname.mp4)
        ...
        fall_cardboard_01_glass_01_Camera_2.mp4
        ...
    """
    def __init__(self,
                 meta_path,
                 data_dir,
                 subsample=None,
                 video_length=16,
                 resolution=[256, 256],
                 frame_stride=1,
                 frame_stride_min=1,
                 spatial_transform=None,
                 crop_resolution=None,
                 fps_max=None,
                 load_raw_resolution=False,
                 fixed_fps=None,
                 random_fs=False,
                 ):
        self.meta_path = meta_path
        self.data_dir = data_dir
        self.subsample = subsample
        self.video_length = video_length
        self.resolution = [resolution, resolution] if isinstance(resolution, int) else resolution
        self.fps_max = fps_max
        self.frame_stride = frame_stride
        self.frame_stride_min = frame_stride_min
        self.fixed_fps = fixed_fps
        self.load_raw_resolution = load_raw_resolution
        self.random_fs = random_fs
        self._load_metadata()
        if spatial_transform is not None:
            if spatial_transform == "random_crop":
                self.spatial_transform = transforms.RandomCrop(crop_resolution)
            elif spatial_transform == "center_crop":
                self.spatial_transform = transforms.Compose([
                    transforms.CenterCrop(resolution),
                ])            
            elif spatial_transform == "resize_center_crop":
                # assert(self.resolution[0] == self.resolution[1])
                self.spatial_transform = transforms.Compose([
                    transforms.Resize(min(self.resolution)),
                    transforms.CenterCrop(self.resolution),
                ])
            elif spatial_transform == "resize":
                self.spatial_transform = transforms.Resize(self.resolution)
            elif spatial_transform == "resize_and_padding":
                self.spatial_transform = ResizeAndPad(self.resolution)
            else:
                raise NotImplementedError
        else:
            self.spatial_transform = None

                
    def _load_metadata(self):
        metadata = pd.read_csv(self.meta_path, dtype=str)
        print(f'>>> {len(metadata)} data samples loaded.')
        if self.subsample is not None:
            metadata = metadata.sample(self.subsample, random_state=0)
   
        #metadata['caption'] = metadata['name']
        #del metadata['name']
        self.metadata = metadata
        self.metadata.dropna(inplace=True)

    def _get_video_path(self, sample):
        #rel_video_fp = os.path.join(sample['page_dir'], str(sample['videoid']) + '.mp4')
        #full_video_fp = os.path.join(self.data_dir, 'videos', rel_video_fp)
        OpenVid_path = "/data/oss_bucket_0/yongfan/data/OpenVid-1M"
        full_video_fp = os.path.join(OpenVid_path, str(sample['video']))
        return full_video_fp
    
    def __getitem__(self, index):
        if self.random_fs:
            frame_stride = random.randint(self.frame_stride_min, self.frame_stride)
        else:
            frame_stride = self.frame_stride

        ## get frames until success
        while True:
            index = index % len(self.metadata)
            sample = self.metadata.iloc[index]
            video_path = self._get_video_path(sample)
            #video_path = "/data/oss_bucket_0/yongfan/data/phys101/train_whole/fall_cardboard_09_foam_01_Camera_1.mp4"

            caption = sample['caption']

            try:
                if self.load_raw_resolution:
                    video_reader = VideoReader(video_path, ctx=cpu(0))
                else:
                    video_reader = VideoReader(video_path, ctx=cpu(0), width=530, height=300)
                if len(video_reader) < self.video_length:
                    print(f"video length ({len(video_reader)}) is smaller than target length({self.video_length})")
                    index += 1
                    continue
                else:
                    pass
            except:
                index += 1
                print(f"Load video failed! path = {video_path}")
                continue
            #print(f"Load video success! path = {video_path}")
            fps_ori = video_reader.get_avg_fps()
            if self.fixed_fps is not None:
                frame_stride = int(frame_stride * (1.0 * fps_ori / self.fixed_fps))

            ## to avoid extreme cases when fixed fps is used
            frame_stride = max(frame_stride, 1)
            
            ## get valid range (adapting case by case)
            required_frame_num = frame_stride * (self.video_length-1) + 1
            frame_num = len(video_reader)
            if frame_num < required_frame_num:
                ## drop extra samples if fixed fps is required
                if self.fixed_fps is not None and frame_num < required_frame_num * 0.5:
                    index += 1
                    continue
                else:
                    frame_stride = frame_num // self.video_length
                    required_frame_num = frame_stride * (self.video_length-1) + 1

            ## select a random clip
            random_range = frame_num - required_frame_num
            start_idx = random.randint(0, random_range) if random_range > 0 else 0

            ## calculate frame indices
            frame_indices = [start_idx + frame_stride*i for i in range(self.video_length)]
            try:
                frames = video_reader.get_batch(frame_indices)
                break
            except:
                print(f"Get frames failed! path = {video_path}; [max_ind vs frame_total:{max(frame_indices)} / {frame_num}]")
                index += 1
                continue
        
        ## process data
        assert(frames.shape[0] == self.video_length),f'{len(frames)}, self.video_length={self.video_length}'
        frames = torch.tensor(frames.asnumpy()).permute(3, 0, 1, 2).float() # [t,h,w,c] -> [c,t,h,w]
        # 读取并归一化帧
        #frames = torch.tensor(frames.asnumpy()).permute(3, 0, 1, 2).float() / 255.0  # [t,h,w,c] -> [c,t,h,w]

        #print("===========ori frames shape:", frames.shape)
        if self.spatial_transform is not None:
            frames = self.spatial_transform(frames)
        #print(f"调整后的帧尺寸：{frames.shape}")
        # 恢复像素值到 [0, 255] 范围
        #frames = frames * 255.0
        #frames = frames.clamp(0, 255)
        #frames = frames.type(torch.uint8)


        if self.resolution is not None:
            assert (frames.shape[2], frames.shape[3]) == (self.resolution[0], self.resolution[1]), f'frames={frames.shape}, self.resolution={self.resolution}'
        
        ## turn frames tensors to [-1,1]
        frames = (frames / 255 - 0.5) * 2
        fps_clip = fps_ori // frame_stride
        if self.fps_max is not None and fps_clip > self.fps_max:
            fps_clip = self.fps_max

        data = {'video': frames, 'caption': caption, 'path': video_path, 'fps': fps_clip, 'frame_stride': frame_stride}
        return data
    
    def __len__(self):
        return len(self.metadata)
