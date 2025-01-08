export NCCL_P2P_DISABLE=0          # Enable P2P communication
export NCCL_P2P_LEVEL=NVL          # Set the P2P level to NVLink or PCI (default is NVLink if available)
export NCCL_BLOCKING_WAIT=1        # Enable blocking wait mode
export NCCL_ASYNC_ERROR_HANDLING=1 # Enable asynchronous error handling
export NCCL_DEBUG=INFO             # Enable NCCL debug mode for more detailed logs
export NCCL_SOCKET_IFNAME=eth0     # Set the network interface name


# args
name="training_256_latent_flow"
config_file=configs/training_latent_flow_256.yaml

# save root dir for logs, checkpoints, tensorboard record, etc.
save_root="/data/oss_bucket_0/yongfan/flow_crafter/work_dirs"

mkdir -p $save_root/$name

## run
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.run \
--nproc_per_node=4 --nnodes=1 --master_addr=127.0.0.1 --master_port=12352 --node_rank=0 \
./main/trainer.py \
--base $config_file \
--train \
--name $name \
--logdir $save_root \
--devices 4 \
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