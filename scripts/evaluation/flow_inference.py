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
from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter, FuncAnimation
import numpy as np
from torchvision.transforms import functional as F


MODEL_PATH="/data/oss_bucket_0/yongfan/flow_crafter/worwork_dirs/training_512_latent_flow_0.00001rt_2/checkpoints/epoch=1-step=800.ckpt"

def resize_and_pad_image(image, size):
    """
    Resize and pad a single PIL image to the target size while maintaining aspect ratio.
    Args:
        image: PIL Image
        size: Tuple or list [H_out, W_out]
    Returns:
        Tensor of shape [C, H_out, W_out]
    """
    # Convert the image to a tensor
    image_tensor = transforms.ToTensor()(image)  # shape [C, H, W]
    
    C, H, W = image_tensor.shape
    target_h, target_w = size
    
    # Compute the scaling factor
    scale = min(target_h / H, target_w / W)
    new_h = int(H * scale)
    new_w = int(W * scale)
    
    # Resize the image
    image_resized = F.resize(image_tensor, [new_h, new_w], antialias=True)
    
    # Calculate padding
    pad_left = (target_w - new_w) // 2
    pad_right = target_w - new_w - pad_left
    pad_top = (target_h - new_h) // 2
    pad_bottom = target_h - new_h - pad_top
    padding = [pad_left, pad_top, pad_right, pad_bottom]  # Left, Top, Right, Bottom

    # Pad the image
    image_padded = F.pad(image_resized, padding)
    return image_padded


class ResizeAndPad(object):
    """Custom transform to resize and pad frames."""
    def __init__(self, size):
        self.size = size  # [H_out, W_out]

    def __call__(self, frames):

        return resize_and_pad_image(frames, self.size)



def plot_grid(x, y, ax=None, **kwargs):
    ax = ax or plt.gca()
    segs1 = np.stack((x, y), axis=2)
    segs2 = segs1.transpose(1, 0, 2)
    ax.add_collection(LineCollection(segs1, **kwargs))
    ax.add_collection(LineCollection(segs2, **kwargs))
    ax.autoscale()
    ax.invert_yaxis()

def save_flow_grid(flow, save_path="", step=1):
    '''
    可视化多帧光流场网格并保存为 GIF 文件。
    
    :param flow: 形状为 (batch_size, 2, frame_num, h, w) 的光流数据。
    :param step: 网格线的步长。
    :param save_path: 保存 GIF 文件的路径。
    '''
    print("flow shape:", flow.shape)
    batch_size, _, frame_num, h, w = flow.shape

    # 创建图形
    fig, ax = plt.subplots()
    ax.axis('off')  # 关闭坐标轴

    def update(frame):
        ax.clear()  # 清除当前的图形
        ax.axis('off')  # 关闭坐标轴
        img = np.zeros((h, w))  # 创建背景图像（可以自定义）

        # 取出光流数据
        flow_frame = flow[0, :, frame].cpu().numpy()  # 选择第一个 batch 的光流帧
        grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))

        # 绘制原始网格
        plot_grid(grid_x[::step, ::step], grid_y[::step, ::step], color="lightgrey", ax=ax)
        
        # 计算光流后的网格坐标
        grid_x2 = grid_x + flow_frame[0]
        grid_y2 = grid_y + flow_frame[1]
        
        # 绘制光流后的网格
        plot_grid(grid_x2[::step, ::step], grid_y2[::step, ::step], color="C0", ax=ax)

        return ax

    # 创建动画
    anim = FuncAnimation(fig, update, frames=frame_num, repeat=False)
    writer = PillowWriter(fps=2)  # 设置 GIF 的帧率
    anim.save(save_path, writer=writer)

    plt.close(fig)



def get_filelist(data_dir, postfixes):
    patterns = [os.path.join(data_dir, f"*.{postfix}") for postfix in postfixes]
    file_list = []
    for pattern in patterns:
        file_list.extend(glob.glob(pattern))
    file_list.sort()
    return file_list

def load_model_checkpoint(model, ckpt):
    state_dict = torch.load(ckpt, map_location="cpu")
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

def load_data_prompts(data_dir, video_size=(256,256), video_frames=16):
    transform = transforms.Compose([
        transforms.Resize(min(video_size)),
        transforms.CenterCrop(video_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
    spatial_transform = ResizeAndPad([256, 256])
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
    file_list = get_filelist(data_dir, ['jpg', 'png', 'jpeg', 'JPEG', 'PNG'])
    print("file_list:", file_list)
    # assert len(file_list) == n_samples, "Error: data and prompts are NOT paired!"
    data_list = []
    filename_list = []
    prompt_list = load_prompts(prompt_file[default_idx])
    print("prompt_list:", prompt_list)
    n_samples = len(prompt_list)
    for idx in range(n_samples):
        #if interp:
        #    image1 = Image.open(file_list[2*idx]).convert('RGB')
        #    image_tensor1 = transform(image1).unsqueeze(1) # [c,1,h,w]
        #    image2 = Image.open(file_list[2*idx+1]).convert('RGB')
        #    image_tensor2 = transform(image2).unsqueeze(1) # [c,1,h,w]
        #    frame_tensor1 = repeat(image_tensor1, 'c t h w -> c (repeat t) h w', repeat=video_frames//2)
        #    frame_tensor2 = repeat(image_tensor2, 'c t h w -> c (repeat t) h w', repeat=video_frames//2)
        #    frame_tensor = torch.cat([frame_tensor1, frame_tensor2], dim=1)
        #    _, filename = os.path.split(file_list[idx*2])
        #else:
        image = Image.open(file_list[idx]).convert('RGB')
        image_tensor = spatial_transform(image).unsqueeze(1) # [c,1,h,w]
        frame_tensor = repeat(image_tensor, 'c t h w -> c (repeat t) h w', repeat=video_frames)
        _, filename = os.path.split(file_list[idx])

        data_list.append(frame_tensor)
        filename_list.append(filename)
        
    return filename_list, data_list, prompt_list



def get_latent_z(model, videos):
    b, c, t, h, w = videos.shape
    x = rearrange(videos, 'b c t h w -> (b t) c h w')
    z = model.encode_first_stage(x)
    z = rearrange(z, '(b t) c h w -> b c t h w', b=b, t=t)
    return z


def image_guided_synthesis(model, prompts, videos, noise_shape, n_samples=1, ddim_steps=50, ddim_eta=1., \
                        unconditional_guidance_scale=1.0, cfg_img=None, fs=None, text_input=False, multiple_cond_cfg=False, timestep_spacing='uniform', guidance_rescale=0.0, **kwargs):
    ddim_sampler = DDIMSampler(model) if not multiple_cond_cfg else DDIMSampler_multicond(model)
    batch_size = noise_shape[0]
    fs = torch.tensor([fs] * batch_size, dtype=torch.long, device=model.device)

    if not text_input:
        prompts = [""]*batch_size
    print("================> video shape:", videos.shape)
    #videos的形状是[bs, c, video_frames, ori_h, ori_w]，其实就是input img在video_frames维度复制多次
    img = videos[:,:,0] #bchw 首帧
    #img_emb = model.embedder(img) ## blc
    #img_emb = model.image_proj_model(img_emb)

    cond_emb = model.get_learned_conditioning(prompts)
    #cond = {"c_crossattn": [torch.cat([cond_emb,img_emb], dim=1)]}
    cond = {"c_crossattn": [cond_emb]}
    if model.model.conditioning_key == 'hybrid':
        z = get_latent_z(model, videos) # b 4 t ori_h/8 ori_w/8
        #print("\nz shape:", z.shape)
        new_z = model.latent_adapter(z) # b 2 t ori_h/8 ori_w/8
        #print("new_z shape:", new_z.shape)
        #if loop or interp:
        #    img_cat_cond = torch.zeros_like(z)
        #    img_cat_cond[:,:,0,:,:] = z[:,:,0,:,:]
        #    img_cat_cond[:,:,-1,:,:] = z[:,:,-1,:,:]
        #else:
        img_cat_cond = new_z[:,:,:1,:,:] #new_z的首帧
        img_cat_cond = repeat(img_cat_cond, 'b c t h w -> b c (repeat t) h w', repeat=new_z.shape[2])
        #print("img_cat_cond shape:", img_cat_cond.shape)
        cond["c_concat"] = [img_cat_cond] # b 2 t h w
    #raise RuntimeError("中断")
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

        #print("samples shape:", samples.shape) # torch.Size([1, 2, 16, 32, 32])
        #raise RuntimeError("中断")

        batch_variants.append(samples) 
    ## variants, batch, 2, t, h, w
    batch_variants = torch.stack(batch_variants) #[n_samples, batch_size, c, t, h, w]
    #print("batch_variants shape:", batch_variants.shape)
    return batch_variants.permute(1, 0, 2, 3, 4, 5)


def run_inference(args, gpu_num, gpu_no):
    ## model config
    config = OmegaConf.load(args.config)
    model_config = config.pop("model", OmegaConf.create())
    
    ## set use_checkpoint as False as when using deepspeed, it encounters an error "deepspeed backend not set"
    model_config['params']['unet_config']['params']['use_checkpoint'] = False
    model = instantiate_from_config(model_config)
    model = model.cuda(gpu_no)
    model.perframe_ae = args.perframe_ae
    print("读取权重路径:", args.ckpt_path)
    assert os.path.exists(args.ckpt_path), "Error: checkpoint Not Found!"
    model = load_model_checkpoint(model, args.ckpt_path)
    model.eval()

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

    os.makedirs(fakedir, exist_ok=True)
    #os.makedirs(fakedir_separate, exist_ok=True)

    ## prompt file setting
    assert os.path.exists(args.prompt_dir), "Error: prompt file Not Found!"
    filename_list, data_list, prompt_list = load_data_prompts(args.prompt_dir, video_size=(args.height, args.width), video_frames=n_frames)
    num_samples = len(prompt_list)
    samples_split = num_samples // gpu_num
    print('Prompts testing [rank:%d] %d/%d samples loaded.'%(gpu_no, samples_split, num_samples))
    #indices = random.choices(list(range(0, num_samples)), k=samples_per_device)
    indices = list(range(samples_split*gpu_no, samples_split*(gpu_no+1)))
    prompt_list_rank = [prompt_list[i] for i in indices]
    data_list_rank = [data_list[i] for i in indices]
    filename_list_rank = [filename_list[i] for i in indices]

    start = time.time()
    with torch.no_grad(), torch.cuda.amp.autocast():
        for idx, indice in tqdm(enumerate(range(0, len(prompt_list_rank), args.bs)), desc='Sample Batch'):
            prompts = prompt_list_rank[indice:indice+args.bs]
            videos = data_list_rank[indice:indice+args.bs]
            filenames = filename_list_rank[indice:indice+args.bs]
            if isinstance(videos, list):
                videos = torch.stack(videos, dim=0).to("cuda")#[bs, c, video_frames, h, w]
            else:
                videos = videos.unsqueeze(0).to("cuda")

            batch_samples = image_guided_synthesis(model, prompts, videos, noise_shape, args.n_samples, args.ddim_steps, args.ddim_eta, \
                                args.unconditional_guidance_scale, args.cfg_img, args.frame_stride, args.text_input, args.multiple_cond_cfg, args.timestep_spacing, args.guidance_rescale)

            #print("batch_samples shape:", batch_samples.shape)
            ## save each example individually
            for nn, samples in enumerate(batch_samples):
                ## samples : [args.n_samples,c,t,h,w]
                #print("======》samples shape:", samples.shape)
                #raise RuntimeError("中断")
                prompt = prompts[nn]
                filename = filenames[nn].replace(".png",".gif")
                print("fakedir:", fakedir)
                print("filename:", filename)
                #raise RuntimeError("中断")
                save_path = os.path.join(fakedir, filename)
                save_flow_grid(samples, save_path)
                #save_results_seperate(prompt, samples, filename, fakedir, fps=8)

    print(f"Saved in {args.savedir}. Time used: {(time.time() - start):.2f} seconds")


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--savedir", type=str, default=None, help="results saving path")
    parser.add_argument("--ckpt_path", type=str, default=MODEL_PATH, help="checkpoint path")
    parser.add_argument("--config", type=str, help="config (yaml) path")
    parser.add_argument("--prompt_dir", type=str, default=None, help="a data dir containing videos and prompts")
    parser.add_argument("--n_samples", type=int, default=1, help="num of samples per prompt",)
    parser.add_argument("--ddim_steps", type=int, default=50, help="steps of ddim if positive, otherwise use DDPM",)
    parser.add_argument("--ddim_eta", type=float, default=1.0, help="eta for ddim sampling (0.0 yields deterministic sampling)",)
    parser.add_argument("--bs", type=int, default=1, help="batch size for inference, should be one")
    parser.add_argument("--height", type=int, default=512, help="image height, in pixel space")
    parser.add_argument("--width", type=int, default=512, help="image width, in pixel space")
    parser.add_argument("--frame_stride", type=int, default=4, help="frame stride control for 256 model (larger->larger motion), FPS control for 512 or 1024 model (smaller->larger motion)")
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

    ## currently not support looping video and generative frame interpolation
    #parser.add_argument("--loop", action='store_true', default=False, help="generate looping videos or not")
    #parser.add_argument("--interp", action='store_true', default=False, help="generate generative frame interpolation or not")
    return parser


if __name__ == '__main__':
    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    print("@FlowCrafter cond-Inference: %s"%now)
    parser = get_parser()
    args = parser.parse_args()

    seed = args.seed
    if seed < 0:
        seed = random.randint(0, 2 ** 31)
    seed_everything(seed)
    rank, gpu_num = 0, 1
    run_inference(args, gpu_num, rank)