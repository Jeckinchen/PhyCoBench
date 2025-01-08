version=$1 ##1024, 512, 256
seed=123
name=2unetcrafter_$1_seed${seed}_keling_score_485epoch

ckpt=/mnt/workspace/yongfan/weights/DynamiCrafter/epoch=47-step=1750.ckpt
config=/mnt/workspace/yongfan/code/DynamiCrafter_2/DynamiCrafter/configs/inference_visual_512_v1.0.yaml

prompt_dir=prompts/keling1.5/
res_dir="/data/oss_bucket_0/yongfan/flow_crafter/work_dirs/latent_flow_infer/results"

if [ "$1" == "256" ]; then
    H=256
    FS=3  ## This model adopts frame stride=3, range recommended: 1-6 (larger value -> larger motion)
elif [ "$1" == "512" ]; then
    H=320
    FS=24 ## This model adopts FPS=24, range recommended: 15-30 (smaller value -> larger motion)
elif [ "$1" == "1024" ]; then
    H=576
    FS=10 ## This model adopts FPS=10, range recommended: 15-5 (smaller value -> larger motion)
else
    echo "Invalid input. Please enter 256, 512, or 1024."
    exit 1
fi


CUDA_VISIBLE_DEVICES=0 python3 scripts/evaluation/2unet_inference_video_firstframe.py \
--seed ${seed} \
--ckpt_path $ckpt \
--flow_diffusion_model_ckpt_path /mnt/workspace/yongfan/weights/DynamiCrafter/epoch=485-step=29600.ckpt \
--config $config \
--flow_diffusion_model_config /mnt/workspace/yongfan/code/DynamiCrafter_2/DynamiCrafter/configs/inference_flow_256_v1.0.yaml \
--savedir $res_dir/$name \
--n_samples 1 \
--bs 1 --height ${H} --width $1 \
--unconditional_guidance_scale 7.5 \
--ddim_steps 50 \
--ddim_eta 1.0 \
--prompt_dir $prompt_dir \
--text_input \
--video_length 16 \
--frame_stride ${FS} \
--timestep_spacing 'uniform_trailing' --guidance_rescale 0.7 --perframe_ae




## multi-cond CFG: the <unconditional_guidance_scale> is s_txt, <cfg_img> is s_img
#--multiple_cond_cfg --cfg_img 7.5
#--loop