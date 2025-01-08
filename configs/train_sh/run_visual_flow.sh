
# args
name="training_512_visual_debug"
config_file=configs/training_visual_flow_512.yaml

# save root dir for logs, checkpoints, tensorboard record, etc.
save_root="/data/oss_bucket_0/yongfan/flow_crafter/work_dirs"

mkdir -p $save_root/$name

## run
CUDA_VISIBLE_DEVICES=0 python3 -m torch.distributed.run \
--nproc_per_node=1 --nnodes=1 --master_addr=127.0.0.1 --master_port=12352 --node_rank=0 \
./main/trainer_flowvisual.py \
--base $config_file \
--train \
--name $name \
--logdir $save_root \
--devices 1 \
--num_nodes 1 \
lightning.trainer.num_nodes=1

## debugging
# CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.launch \
# --nproc_per_node=4 --nnodes=1 --master_addr=127.0.0.1 --master_port=12352 --node_rank=0 \
# ./main/trainer.py \
# --base $config_file \
# --train \
# --name $name \
# --logdir $save_root \
# --devices 4 \
# lightning.trainer.num_nodes=1