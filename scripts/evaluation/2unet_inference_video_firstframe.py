import argparse, os, sys, glob
import datetime, time
from omegaconf import OmegaConf
from tqdm import tqdm
from einops import rearrange, repeat
from collections import OrderedDict

import torch
import torchvision
import torchvision.transforms as transforms
from pytorch_lightning import seed_everything
from PIL import Image
sys.path.insert(1, os.path.join(sys.path[0], '..', '..'))
from lvdm.models.samplers.ddim import DDIMSampler
from lvdm.models.samplers.ddim_multiplecond import DDIMSampler as DDIMSampler_multicond
from utils.utils import instantiate_from_config
import random
import decord
from decord import VideoReader
from torchvision.transforms import functional as F
import torch.nn.functional as F_nn
import cv2
import numpy as np
import imageio
import json


def save_video_with_opencv(grid, path, fps):
    T, H, W, C = grid.shape
    # 使用 MJPG 编码器
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    # 将文件扩展名改为 .avi
    path = path.replace('.mp4', '.avi')
    # 创建 VideoWriter 对象
    video_writer = cv2.VideoWriter(path, fourcc, fps, (W, H))
    
    for frame in grid:
        if isinstance(frame, torch.Tensor):
            frame = frame.numpy()
        # 确保帧数据类型为 uint8
        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8)
        # 检查帧的形状，应为 [H, W, 3]
        if frame.shape[2] == 3:
            # OpenCV 使用 BGR 顺序，需要转换颜色空间
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        else:
            print("Error: Frame has incorrect number of channels")
            continue
        video_writer.write(frame)
    video_writer.release()


def resize_and_pad_image(image, size):
    """
    Resize and pad a single PIL image to the target size while maintaining aspect ratio.
    Args:
        image: PIL Image
        size: Tuple or list [H_out, W_out]
    Returns:
        Tensor of shape [C, H_out, W_out]
    """
    # Convert the image to a tensor and normalize
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    image_tensor = transform(image)  # shape [C, H, W]

    C, H, W = image_tensor.shape
    target_h, target_w = size

    # Compute the scaling factor
    scale = min(target_h / H, target_w / W)
    new_h = int(H * scale)
    new_w = int(W * scale)

    # Resize the image
    image_resized = F_nn.interpolate(image_tensor.unsqueeze(0), size=[new_h, new_w], mode='bilinear', align_corners=False).squeeze(0)

    # Calculate padding
    pad_left = (target_w - new_w) // 2
    pad_right = target_w - new_w - pad_left
    pad_top = (target_h - new_h) // 2
    pad_bottom = target_h - new_h - pad_top
    padding = [pad_left, pad_right, pad_top, pad_bottom]  # Left, Right, Top, Bottom

    # Pad the image
    image_padded = F_nn.pad(image_resized, padding, value=0)

    return image_padded



class ResizeAndPad(object):
    """Custom transform to resize and pad frames."""
    def __init__(self, size):
        self.size = size  # [H_out, W_out]

    def __call__(self, frames):

        return resize_and_pad_image(frames, self.size)


def get_filelist(data_dir, postfixes):
    patterns = [os.path.join(data_dir, f"*.{postfix}") for postfix in postfixes]
    file_list = []
    for pattern in patterns:
        file_list.extend(glob.glob(pattern))
    file_list.sort()
    return file_list

def load_model_checkpoint(model, ckpt):
    state_dict = torch.load(ckpt, map_location="cpu")
    #with open('/mnt/merchant/yongfan/code/DynamiCrafter_2/DynamiCrafter/visual_model_keys.txt', 'w') as f:
    #    for k, v in state_dict["state_dict"].items():
    #        f.write(f'{k}\n')
    if "state_dict" in list(state_dict.keys()):
        state_dict = state_dict["state_dict"]
        try:
            model.load_state_dict(state_dict, strict=True)
        except:
            ## rename the keys for 256x256 model
            new_pl_sd = OrderedDict()
            for k,v in state_dict.items():
                new_pl_sd[k] = v

            for k in list(new_pl_sd.keys()):
                if "framestride_embed" in k:
                    new_key = k.replace("framestride_embed", "fps_embedding")
                    new_pl_sd[new_key] = new_pl_sd[k]
                    del new_pl_sd[k]
            model.load_state_dict(new_pl_sd, strict=True)
    else:
        # deepspeed
        new_pl_sd = OrderedDict()
        for key in state_dict['module'].keys():
            new_pl_sd[key[16:]]=state_dict['module'][key]
        model.load_state_dict(new_pl_sd)
    print('>>> model checkpoint loaded.')
    return model

def load_prompts(prompt_file):
    f = open(prompt_file, 'r')
    prompt_list = []
    for idx, line in enumerate(f.readlines()):
        l = line.strip()
        if len(l) != 0:
            prompt_list.append(l)
        f.close()
    return prompt_list

def load_videos_prompts(data_dir, video_size=(256,256), video_frames=16):
    transform = transforms.Compose([
        transforms.Resize(min(video_size)),
        transforms.CenterCrop(video_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
    spatial_transform = ResizeAndPad([320, 512])
    spatial_transform_256 = ResizeAndPad([256, 256])
    ## load prompts
    prompt_file = get_filelist(data_dir, ['txt'])
    assert len(prompt_file) > 0, "Error: found NO prompt file!"
    ###### default prompt
    default_idx = 0
    default_idx = min(default_idx, len(prompt_file)-1)
    if len(prompt_file) > 1:
        print(f"Warning: multiple prompt files exist. The one {os.path.split(prompt_file[default_idx])[1]} is used.")
    ## only use the first one (sorted by name) if multiple exist
    
    ## load video
    #file_list = get_filelist(data_dir, ['jpg', 'png', 'jpeg', 'JPEG', 'PNG'])
    file_list = get_filelist(data_dir, ['mp4'])
    # assert len(file_list) == n_samples, "Error: data and prompts are NOT paired!"
    data_list = [] 
    data_list_256 = []
    filename_list = []
    prompt_list = load_prompts(prompt_file[default_idx])
    n_samples = len(prompt_list)
    for idx in range(n_samples):
        video_path = file_list[idx]
        vr = VideoReader(video_path)
        frame_tensor = []
        frame_tensor_256 = []
        for frame in vr:
            #print("=======frame shape:",frame.shape)
            frame = frame.asnumpy()  # Convert NDArray to numpy array
            frame = Image.fromarray(frame)  # Convert numpy array to PIL Image
            frame = frame.convert("RGB")
            #frame = frame.permute(2,0,1)
            frame_512 = spatial_transform(frame)
            frame_256 = spatial_transform_256(frame)
            #print("frame shape:", frame.shape)
            #print("=======frame shape:",frame.shape)
            frame_tensor.append(frame_512)
            frame_tensor_256.append(frame_256)
        frame_tensor = torch.stack(frame_tensor) # frame_num c h w
        frame_tensor = frame_tensor.permute(1, 0, 2, 3)
        frame_tensor_256 = torch.stack(frame_tensor_256)
        frame_tensor_256 = frame_tensor_256.permute(1, 0, 2, 3)
        #print("==========frame_tensor shape:", frame_tensor.shape)

        _, filename = os.path.split(file_list[idx])

        data_list.append(frame_tensor)
        data_list_256.append(frame_tensor_256)
        filename_list.append(filename)
        
    return filename_list, data_list, prompt_list, data_list_256


def save_results(prompt, samples, filename, fakedir, fps=8):
    filename = filename.split('.')[0]+'.mp4'
    prompt = prompt[0] if isinstance(prompt, list) else prompt

    ## save video
    videos = [samples]
    savedirs = [fakedir]
    for idx, video in enumerate(videos):
        if video is None:
            continue
        # b,c,t,h,w
        video = video.detach().cpu()
        video = torch.clamp(video.float(), -1., 1.)
        n = video.shape[0]
        video = video.permute(2, 0, 1, 3, 4) # t,n,c,h,w
        
        frame_grids = [torchvision.utils.make_grid(framesheet, nrow=int(n), padding=0) for framesheet in video] #[3, 1*h, n*w]
        grid = torch.stack(frame_grids, dim=0) # stack in temporal dim [t, 3, h, n*w]
        grid = (grid + 1.0) / 2.0
        grid = (grid * 255).to(torch.uint8).permute(0, 2, 3, 1)
        path = os.path.join(savedirs[idx], filename)
        torchvision.io.write_video(path, grid, fps=fps, video_codec='h264', options={'crf': '10'}) ## crf indicates the quality


def save_results_seperate(prompt, samples, filename, fakedir, fps=10):
    prompt = prompt[0] if isinstance(prompt, list) else prompt

    ## save video
    videos = [samples]
    savedirs = [fakedir]
    for idx, video in enumerate(videos):
        if video is None:
            continue
        # b,c,t,h,w
        video = video.detach().cpu()
        video = torch.clamp(video.float(), -1., 1.)
        n = video.shape[0]
        for i in range(n):
            grid = video[i,...]
            grid = (grid + 1.0) / 2.0
            grid = (grid * 255).to(torch.uint8).permute(1, 2, 3, 0) #thwc
            path = os.path.join(savedirs[idx].replace('samples', 'samples_separate'), f'{filename.split(".")[0]}_sample{i}.mp4')
            #torchvision.io.write_video(path, grid, fps=fps, video_codec='h264', options={'crf': '10'})
            #torchvision.io.write_video(path, grid, fps=fps, video_codec='libx264', options={'crf': '10'})
            save_video_with_opencv(grid, path, fps)

def save_results_seperate_gif(prompt, samples, filename, fakedir, fps=10):

    prompt = prompt[0] if isinstance(prompt, list) else prompt

    # 保存视频
    videos = [samples]
    savedirs = [fakedir]
    for idx, video in enumerate(videos):
        if video is None:
            continue
        video = video.detach().cpu()
        # 打印初始像素值范围
        #print(f"Initial video min: {video.min()}, max: {video.max()}")
        video = torch.clamp(video.float(), -1., 1.)
        # 打印截断后的像素值范围
        #print(f"Clamped video min: {video.min()}, max: {video.max()}")
        min_val = video.min()
        max_val = video.max()
        # 进行线性映射
        video = 2 * (video - min_val) / (max_val - min_val) - 1

        #print(f"映射后 video min: {video.min()}, max: {video.max()}")
        n = video.shape[0]  # 样本数量
        for i in range(n):
            grid = video[i, ...]  # [C, T, H, W]
            # 打印每个样本的初始像素值范围
            #print(f"Sample {i} initial grid min: {grid.min()}, max: {grid.max()}")
            grid = (grid + 1.0) / 2.0  # 将像素值调整到 [0, 1]
            # 打印调整到 [0, 1] 后的像素值范围
            #print(f"Sample {i} scaled to [0, 1] grid min: {grid.min()}, max: {grid.max()}")
            grid = (grid * 255).to(torch.uint8)  # 将像素值调整到 [0, 255]
            # 打印调整到 [0, 255] 后的像素值范围
            #print(f"Sample {i} scaled to [0, 255] grid min: {grid.min()}, max: {grid.max()}")
            grid = grid.permute(1, 2, 3, 0)  # [T, H, W, C]
            grid = grid.numpy()  # 转换为 NumPy 数组

            # 确保保存路径存在
            save_dir = savedirs[idx].replace('samples', 'samples_separate')
            os.makedirs(save_dir, exist_ok=True)

            # 设置保存路径
            path = os.path.join(save_dir, f'{filename.split(".")[0]}_sample{i}.gif')

            # 使用 imageio 保存为 GIF
            try:
                # 将帧列表保存为 GIF
                imageio.mimsave(path, grid, format='GIF-PIL', fps=fps)
                print(f"Saved GIF to {path}")
            except Exception as e:
                print(f"Error saving GIF {path}: {e}")


def get_latent_z(model, videos):
    b, c, t, h, w = videos.shape
    x = rearrange(videos, 'b c t h w -> (b t) c h w')
    z = model.encode_first_stage(x)
    z = rearrange(z, '(b t) c h w -> b c t h w', b=b, t=t)
    return z


def flow_guided_synthesis(model, prompts, videos, latent_flow, noise_shape, n_samples=1, ddim_steps=50, ddim_eta=1., \
                        unconditional_guidance_scale=1.0, cfg_img=None, fs=None, text_input=False, multiple_cond_cfg=False, timestep_spacing='uniform', guidance_rescale=0.0, **kwargs):
    ddim_sampler = DDIMSampler(model) if not multiple_cond_cfg else DDIMSampler_multicond(model)
    batch_size = noise_shape[0]
    fs = torch.tensor([fs] * batch_size, dtype=torch.long, device=model.device)

    if not text_input:
        prompts = [""]*batch_size
    #videos的形状是[bs, c, video_frames, h, w]，其实是input img在video_frames维度复制多次
    img = videos[:,:,0] #bchw
    img_emb = model.embedder(img) ## blc
    img_emb = model.image_proj_model(img_emb)

    cond_emb = model.get_learned_conditioning(prompts)
    cond = {"c_crossattn": [torch.cat([cond_emb,img_emb], dim=1)]}
    if model.model.conditioning_key == 'hybrid':
        concat_flow = model.flow_adapter(latent_flow).to("cuda") # b 4 16 32 32
        z = get_latent_z(model, videos) # b c t h w
        img_cat_cond = z[:,:,:1,:,:]
        img_cat_cond = repeat(img_cat_cond, 'b c t h w -> b c (repeat t) h w', repeat=16)
        cond["c_concat"] = [concat_flow + img_cat_cond]

    
    if unconditional_guidance_scale != 1.0:
        if model.uncond_type == "empty_seq":
            prompts = batch_size * [""]
            uc_emb = model.get_learned_conditioning(prompts)
        elif model.uncond_type == "zero_embed":
            uc_emb = torch.zeros_like(cond_emb)
        uc_img_emb = model.embedder(torch.zeros_like(img)) ## b l c
        uc_img_emb = model.image_proj_model(uc_img_emb)
        uc = {"c_crossattn": [torch.cat([uc_emb,uc_img_emb],dim=1)]}
        if model.model.conditioning_key == 'hybrid':
            uc["c_concat"] = [concat_flow + img_cat_cond]
    else:
        uc = None

    ## we need one more unconditioning image=yes, text=""
    if multiple_cond_cfg and cfg_img != 1.0:
        uc_2 = {"c_crossattn": [torch.cat([uc_emb,img_emb],dim=1)]}
        if model.model.conditioning_key == 'hybrid':
            uc_2["c_concat"] = [img_cat_cond]
        kwargs.update({"unconditional_conditioning_img_nonetext": uc_2})
    else:
        kwargs.update({"unconditional_conditioning_img_nonetext": None})

    z0 = None
    cond_mask = None

    batch_variants = []
    for _ in range(n_samples):

        if z0 is not None:
            cond_z0 = z0.clone()
            kwargs.update({"clean_cond": True})
        else:
            cond_z0 = None
        if ddim_sampler is not None:

            samples, _ = ddim_sampler.sample(S=ddim_steps,
                                            conditioning=cond,
                                            batch_size=batch_size,
                                            shape=noise_shape[1:],
                                            verbose=False,
                                            unconditional_guidance_scale=unconditional_guidance_scale,
                                            unconditional_conditioning=uc,
                                            eta=ddim_eta,
                                            cfg_img=cfg_img, 
                                            mask=cond_mask,
                                            x0=cond_z0,
                                            fs=fs,
                                            timestep_spacing=timestep_spacing,
                                            guidance_rescale=guidance_rescale,
                                            **kwargs
                                            )

        ## reconstruct from latent to pixel space
        batch_images = model.decode_first_stage(samples)
        batch_variants.append(batch_images)
    ## variants, batch, c, t, h, w
    batch_variants = torch.stack(batch_variants)
    return batch_variants.permute(1, 0, 2, 3, 4, 5)

def image_guided_synthesis(model, prompts, videos, noise_shape, n_samples=1, ddim_steps=50, ddim_eta=1., \
                        unconditional_guidance_scale=1.0, cfg_img=None, fs=None, text_input=False, multiple_cond_cfg=False, timestep_spacing='uniform', guidance_rescale=0.0, **kwargs):
    ddim_sampler = DDIMSampler(model) if not multiple_cond_cfg else DDIMSampler_multicond(model)
    batch_size = noise_shape[0]
    fs = torch.tensor([fs] * batch_size, dtype=torch.long, device=model.device)
    #print("----------------->nosie_shape:", noise_shape)
    if not text_input:
        prompts = [""]*batch_size
    #videos的形状是[bs, c, video_frames, h, w]，其实是input img在video_frames维度复制多次
    img = videos[:,:,0] #bchw
    videos = repeat(img, 'b c h w -> b c t h w', t=noise_shape[2])
    #print("img shape:", img.shape)
    cond_emb = model.get_learned_conditioning(prompts)
    cond = {"c_crossattn": [cond_emb]}
    if model.model.conditioning_key == 'hybrid':
        z = get_latent_z(model, videos) # b 4 t ori_h/8 ori_w/8
        #print("z shape:", z.shape)
        new_z = model.latent_adapter(z) # b 2 t ori_h/8 ori_w/8
        #print("new_z shape:", new_z.shape)
        img_cat_cond = new_z[:,:,:1,:,:] #new_z的首帧
        img_cat_cond = repeat(img_cat_cond, 'b c t h w -> b c (repeat t) h w', repeat=new_z.shape[2])
        #print("img_cat_cond shape:", img_cat_cond.shape)
        cond["c_concat"] = [img_cat_cond] # b 2 t h w

    
    if unconditional_guidance_scale != 1.0:
        if model.uncond_type == "empty_seq":
            prompts = batch_size * [""]
            uc_emb = model.get_learned_conditioning(prompts)
        elif model.uncond_type == "zero_embed":
            uc_emb = torch.zeros_like(cond_emb)
        #uc_img_emb = model.embedder(torch.zeros_like(img)) ## b l c
        #uc_img_emb = model.image_proj_model(uc_img_emb)
        #uc = {"c_crossattn": [torch.cat([uc_emb,uc_img_emb],dim=1)]}
        uc = {"c_crossattn": [uc_emb]}
        if model.model.conditioning_key == 'hybrid':
            uc["c_concat"] = [img_cat_cond]
    else:
        uc = None

    ## we need one more unconditioning image=yes, text=""
    if multiple_cond_cfg and cfg_img != 1.0:
        uc_2 = {"c_crossattn": [torch.cat([uc_emb,img_emb],dim=1)]}
        if model.model.conditioning_key == 'hybrid':
            uc_2["c_concat"] = [img_cat_cond]
        kwargs.update({"unconditional_conditioning_img_nonetext": uc_2})
    else:
        kwargs.update({"unconditional_conditioning_img_nonetext": None})

    z0 = None
    cond_mask = None

    batch_variants = []
    for _ in range(n_samples):

        if z0 is not None:
            cond_z0 = z0.clone()
            kwargs.update({"clean_cond": True})
        else:
            cond_z0 = None
        if ddim_sampler is not None:

            samples, _ = ddim_sampler.sample(S=ddim_steps,
                                            conditioning=cond,
                                            batch_size=batch_size,
                                            shape=noise_shape[1:],
                                            verbose=False,
                                            unconditional_guidance_scale=unconditional_guidance_scale,
                                            unconditional_conditioning=uc,
                                            eta=ddim_eta,
                                            cfg_img=cfg_img, 
                                            mask=cond_mask,
                                            x0=cond_z0,
                                            fs=fs,
                                            timestep_spacing=timestep_spacing,
                                            guidance_rescale=guidance_rescale,
                                            **kwargs
                                            )
        batch_variants.append(samples) 
    ## variants, batch, c, t, h, w
    batch_variants = torch.stack(batch_variants)
    #print("batch_variants shape:", batch_variants.shape) #[n_samples, batch_size, c, t, h, w]
    #raise RuntimeError("中断程序")
    #return batch_variants.permute(1, 0, 2, 3, 4, 5)
    return batch_variants[0] #[batch_size, c, t, h, w]


def run_inference(args, gpu_num, gpu_no):
    ## model config
    config = OmegaConf.load(args.config)
    model_config = config.pop("model", OmegaConf.create())
    flow_diffusion_config = OmegaConf.load(args.flow_diffusion_model_config)
    flow_diffusion_model_config = flow_diffusion_config.pop("model", OmegaConf.create())
    
    
    ## set use_checkpoint as False as when using deepspeed, it encounters an error "deepspeed backend not set"
    model_config['params']['unet_config']['params']['use_checkpoint'] = False
    flow_diffusion_model_config['params']['unet_config']['params']['use_checkpoint'] = False
    model = instantiate_from_config(model_config)
    model = model.cuda(gpu_no)
    model.perframe_ae = args.perframe_ae
    flow_diffusion_model = instantiate_from_config(flow_diffusion_model_config)
    flow_diffusion_model = flow_diffusion_model.cuda(gpu_no)
    flow_diffusion_model.perframe_ae = args.perframe_ae

    assert os.path.exists(args.ckpt_path), "Error: checkpoint Not Found!"
    assert os.path.exists(args.flow_diffusion_model_ckpt_path), "Error: checkpoint Not Found!"
    model = load_model_checkpoint(model, args.ckpt_path)
    model.eval()
    flow_diffusion_model = load_model_checkpoint(flow_diffusion_model, args.flow_diffusion_model_ckpt_path)
    flow_diffusion_model.eval()

    #raise RuntimeError("中断程序")

    ## run over data
    assert (args.height % 16 == 0) and (args.width % 16 == 0), "Error: image size [h,w] should be multiples of 16!"
    assert args.bs == 1, "Current implementation only support [batch size = 1]!"
    ## latent noise shape
    h, w = args.height // 8, args.width // 8
    channels = model.model.diffusion_model.out_channels
    n_frames = args.video_length
    print(f'Inference with {n_frames} frames')
    noise_shape = [args.bs, channels, n_frames, h, w]

    fakedir = os.path.join(args.savedir, "samples")
    fakedir_separate = os.path.join(args.savedir, "samples_separate")
    score_path = os.path.join(fakedir_separate, "score.json")
    score_data = []

    # os.makedirs(fakedir, exist_ok=True)
    os.makedirs(fakedir_separate, exist_ok=True)

    ## prompt file setting
    assert os.path.exists(args.prompt_dir), "Error: prompt file Not Found!"
    filename_list, data_list, prompt_list, data_list_256 = load_videos_prompts(args.prompt_dir, video_size=(args.height, args.width), video_frames=n_frames)
    num_samples = len(prompt_list)
    samples_split = num_samples // gpu_num
    print('Prompts testing [rank:%d] %d/%d samples loaded.'%(gpu_no, samples_split, num_samples))
    #indices = random.choices(list(range(0, num_samples)), k=samples_per_device)
    indices = list(range(samples_split*gpu_no, samples_split*(gpu_no+1)))
    prompt_list_rank = [prompt_list[i] for i in indices]
    data_list_rank = [data_list[i] for i in indices]
    data_list_rank_256 = [data_list_256[i] for i in indices]
    filename_list_rank = [filename_list[i] for i in indices]

    start = time.time()
    with torch.no_grad(), torch.cuda.amp.autocast():
        for idx, indice in tqdm(enumerate(range(0, len(prompt_list_rank), args.bs)), desc='Sample Batch'):
            prompts = prompt_list_rank[indice:indice+args.bs]
            videos = data_list_rank[indice:indice+args.bs]
            videos_256 = data_list_rank_256[indice:indice+args.bs]
            filenames = filename_list_rank[indice:indice+args.bs]
            if isinstance(videos, list):
                videos = torch.stack(videos, dim=0).to("cuda")#[bs, c, video_frames, h, w]
                videos_256 = torch.stack(videos_256, dim=0).to("cuda")
            else:
                videos = videos.unsqueeze(0).to("cuda")
                videos_256 = videos_256.unsqueeze(0).to("cuda")

            batch_flow = image_guided_synthesis(flow_diffusion_model, prompts, videos_256, [args.bs, 2, n_frames, 32, 32], args.n_samples, args.ddim_steps, args.ddim_eta, \
                                args.unconditional_guidance_scale, args.cfg_img, args.frame_stride, args.text_input, args.multiple_cond_cfg, args.timestep_spacing, args.guidance_rescale)
            batch_flow = flow_diffusion_model.downsample_optical_flow_batch(batch_flow, target_size=(40, 64)).to("cuda")
            #print("batch_flow shape:", batch_flow.shape)
            #raise RuntimeError("中断程序")
            batch_samples = flow_guided_synthesis(model, prompts, videos, batch_flow, [args.bs, 4, n_frames, h, w], args.n_samples, args.ddim_steps, args.ddim_eta, \
                                args.unconditional_guidance_scale, args.cfg_img, args.frame_stride, args.text_input, args.multiple_cond_cfg, args.timestep_spacing, args.guidance_rescale)
            #print("===========batch samples shape:", batch_samples.shape)
            ## save each example individually
            #print(f"输入 video min: {videos.min()}, max: {videos.max()}")
            for nn, samples in enumerate(batch_samples):
                ## samples : [n_samples,c,t,h,w]
                prompt = prompts[nn]
                filename = filenames[nn]
                #save_results_seperate(prompt, samples, filename, fakedir, fps=8)
                save_results_seperate_gif(prompt, samples, filename, fakedir, fps=8)
            #"""
            # 比较预测光流和video和原光流、video的相似性
            video_frames = videos.shape[2]  # 原始视频的帧数
            #if video_frames > 16:
            #    videos_ = videos[:, :, :16, :, :]
            #print("videos shape:", videos.shape)
            #ori_flow = flow_diffusion_model.compute_optical_flow_for_batch(videos[:, :, :16, :, :], flow_diffusion_model.flow_estimator).to("cuda")
            ori_flow = flow_diffusion_model.compute_optical_flow_for_batch(videos, flow_diffusion_model.flow_estimator).to("cuda")
            #print("ori_flow shape:", ori_flow.shape)
            # 1 2 frame_num 40 64
            ori_latent_flow = flow_diffusion_model.downsample_optical_flow_batch(ori_flow, target_size=(40, 64)).to("cuda") 
            #print("ori_latent_flow shape:", ori_latent_flow.shape)
            # 1 3 16 320 512
            pred_video = batch_samples.permute(1, 0, 2, 3, 4, 5)[0]

            #将ori_latent_flow插值到16帧
            interpolated_ori_latent_flow = F_nn.interpolate(
                ori_latent_flow, 
                size=(16, ori_latent_flow.shape[3], ori_latent_flow.shape[4]), 
                mode='trilinear', 
                align_corners=False
            )
            #print("interpolated_ori_latent_flow shape:", interpolated_ori_latent_flow.shape)
            #对videos进行均匀采样到16帧
            if video_frames >= 16:
                indices = torch.linspace(0, video_frames - 1, steps=16).long().to(videos.device)
                sampled_videos = videos[:, :, indices, :, :]
                #sampled_videos = videos[:, :, :16, :, :]
            else:
                # 如果帧数少于 16，则进行插值
                sampled_videos = F_nn.interpolate(
                    videos, 
                    size=(16, videos.shape[3], videos.shape[4]), 
                    mode='trilinear', 
                    align_corners=False
                )
            
            flow_mse = F_nn.mse_loss(batch_flow, interpolated_ori_latent_flow)
            print(f"Flow MSE: {flow_mse.item()}")

            # Determine if the original flow is abnormal
            flow_threshold = 0.1  # Adjust this threshold as needed
            #if flow_mse.item() > flow_threshold:
            #    print("The original optical flow is abnormal.")
            #else:
            #    print("The original optical flow is normal.")

            # Normalize predicted and sampled videos to [0, 1] range
            pred_video_norm = (pred_video + 1.0) / 2.0
            sampled_videos_norm = (sampled_videos + 1.0) / 2.0

            # Compute MSE between predicted video and original video
            video_mse = F_nn.mse_loss(pred_video_norm, sampled_videos_norm)
            print(f"Video MSE: {video_mse.item()}")

            # Determine if the original video is abnormal
            video_threshold = 0.1  # Adjust this threshold as needed
            #if video_mse.item() > video_threshold:
            #    print("The original video is abnormal.")
            #else:
            #    print("The original video is normal.")
            score = {
                "video": filename,
                "flow_score": flow_mse.item(),
                "video_scre": video_mse.item()
            }
            score_data.append(score)
            
            #"""
            #print("interpolated_ori_latent_flow shape:", interpolated_ori_latent_flow.shape)
            #print("sampled_videos shape:", sampled_videos.shape)
            #raise RuntimeError("中断")
    with open(score_path, 'w', encoding='utf-8') as file:
        json.dump(score_data, file, ensure_ascii=False, indent=4)
    print(f"分数已成功写入到JSON文件{score_path}")

    print(f"Saved in {args.savedir}. Time used: {(time.time() - start):.2f} seconds")


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--savedir", type=str, default=None, help="results saving path")
    parser.add_argument("--ckpt_path", type=str, default=None, help="checkpoint path")
    parser.add_argument("--flow_diffusion_model_ckpt_path", type=str, default="/mnt/merchant/yongfan/weights/DynamiCrafter/epoch=119-step=4050.ckpt", help="checkpoint path")
    parser.add_argument("--config", type=str, help="config (yaml) path")
    parser.add_argument("--flow_diffusion_model_config", type=str, default="/mnt/merchant/yongfan/code/DynamiCrafter_2/DynamiCrafter/configs/inference_flow_512_v1.0.yaml", help="config (yaml) path")
    parser.add_argument("--prompt_dir", type=str, default=None, help="a data dir containing videos and prompts")
    parser.add_argument("--n_samples", type=int, default=1, help="num of samples per prompt",)
    parser.add_argument("--ddim_steps", type=int, default=50, help="steps of ddim if positive, otherwise use DDPM",)
    parser.add_argument("--ddim_eta", type=float, default=1.0, help="eta for ddim sampling (0.0 yields deterministic sampling)",)
    parser.add_argument("--bs", type=int, default=1, help="batch size for inference, should be one")
    parser.add_argument("--height", type=int, default=512, help="image height, in pixel space")
    parser.add_argument("--width", type=int, default=512, help="image width, in pixel space")
    parser.add_argument("--frame_stride", type=int, default=3, help="frame stride control for 256 model (larger->larger motion), FPS control for 512 or 1024 model (smaller->larger motion)")
    parser.add_argument("--unconditional_guidance_scale", type=float, default=1.0, help="prompt classifier-free guidance")
    parser.add_argument("--seed", type=int, default=123, help="seed for seed_everything")
    parser.add_argument("--video_length", type=int, default=16, help="inference video length")
    parser.add_argument("--negative_prompt", action='store_true', default=False, help="negative prompt")
    parser.add_argument("--text_input", action='store_true', default=False, help="input text to I2V model or not")
    parser.add_argument("--multiple_cond_cfg", action='store_true', default=False, help="use multi-condition cfg or not")
    parser.add_argument("--cfg_img", type=float, default=None, help="guidance scale for image conditioning")
    parser.add_argument("--timestep_spacing", type=str, default="uniform", help="The way the timesteps should be scaled. Refer to Table 2 of the [Common Diffusion Noise Schedules and Sample Steps are Flawed](https://huggingface.co/papers/2305.08891) for more information.")
    parser.add_argument("--guidance_rescale", type=float, default=0.0, help="guidance rescale in [Common Diffusion Noise Schedules and Sample Steps are Flawed](https://huggingface.co/papers/2305.08891)")
    parser.add_argument("--perframe_ae", action='store_true', default=False, help="if we use per-frame AE decoding, set it to True to save GPU memory, especially for the model of 576x1024")

    return parser


if __name__ == '__main__':
    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    print("@DynamiCrafter cond-Inference: %s"%now)
    parser = get_parser()
    args = parser.parse_args()

    seed = args.seed
    if seed < 0:
        seed = random.randint(0, 2 ** 31)
    seed_everything(seed)
    rank, gpu_num = 0, 1
    run_inference(args, gpu_num, rank)