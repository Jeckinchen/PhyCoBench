import argparse, os, sys, datetime
from omegaconf import OmegaConf
from transformers import logging as transf_logging
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.trainer import Trainer
import torch
import logging

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from utils.utils import instantiate_from_config
from utils_train import get_trainer_callbacks, get_trainer_logger, get_trainer_strategy
from utils_train import set_logger, init_workspace, load_checkpoints

def get_parser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument("--seed", "-s", type=int, default=20240823, help="seed for seed_everything")
    parser.add_argument("--name", "-n", type=str, default="", help="experiment name, as saving folder")

    parser.add_argument("--base", "-b", nargs="*", metavar="base_config.yaml", help="paths to base configs. Loaded from left-to-right. "
                            "Parameters can be overwritten or added with command-line options of the form `--key value`.", default=list())
    
    parser.add_argument("--train", "-t", action='store_true', default=False, help='train')
    parser.add_argument("--val", "-v", action='store_true', default=False, help='val')
    parser.add_argument("--test", action='store_true', default=False, help='test')

    parser.add_argument("--logdir", "-l", type=str, default="logs", help="directory for logging dat shit")
    parser.add_argument("--auto_resume", action='store_true', default=False, help="resume from full-info checkpoint")
    parser.add_argument("--auto_resume_weight_only", action='store_true', default=False, help="resume from weight-only checkpoint")
    parser.add_argument("--debug", "-d", action='store_true', default=False, help="enable post-mortem debugging")

    return parser
    
def get_nondefault_trainer_args(args):
    parser = argparse.ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    default_trainer_args = parser.parse_args([])
    return sorted(k for k in vars(default_trainer_args) if getattr(args, k) != getattr(default_trainer_args, k))

if __name__ == "__main__":
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    local_rank = int(os.environ.get('LOCAL_RANK'))
    global_rank = int(os.environ.get('RANK'))
    num_rank = int(os.environ.get('WORLD_SIZE'))

    parser = get_parser()
    ## Extends existing argparse by default Trainer attributes
    parser = Trainer.add_argparse_args(parser)
    args, unknown = parser.parse_known_args()
    ## disable transformer warning
    transf_logging.set_verbosity_error()
    seed_everything(args.seed)

    try:
        oss_files = os.listdir("/data/oss_bucket_0/yongfan")
        print(f"oss files: {oss_files}")
        print("OSS MOUNT SUCCESS !!!")
    except Exception as e:
        print(f"OSS MOUNT FAILED !!!")
        raise ValueError(f"error info: {e}")

    ## yaml configs: "model" | "data" | "lightning"
    #print("args.base:", args.base)
    configs = [OmegaConf.load(cfg) for cfg in args.base]
    
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(*configs, cli)
    lightning_config = config.pop("lightning", OmegaConf.create())
    trainer_config = lightning_config.get("trainer", OmegaConf.create()) 

    ## setup workspace directories
    workdir, ckptdir, cfgdir, loginfo = init_workspace(args.name, args.logdir, config, lightning_config, global_rank)
    #ckptdir = "/data/oss_bucket_0/yongfan/work_dirs/training_1024_v1_phys101_fall_4nodes_firstframe_2/"
    if not os.path.exists(ckptdir):
        os.makedirs(ckptdir, exist_ok=True)
    logger = set_logger(logfile=os.path.join(loginfo, 'log_%d:%s.txt'%(global_rank, now)))
    logger.info("@lightning version: %s [>=1.8 required]"%(pl.__version__))  


    #ori_model_cfg = ["/mnt/workspace/yongfan/code/DynamiCrafter_2/DynamiCrafter/configs/config_1024.yaml"]
    #ori_configs = [OmegaConf.load(cfg) for cfg in ori_model_cfg]
    #ori_config = OmegaConf.merge(*ori_configs, cli)
    #ori_model = instantiate_from_config(ori_config.model)
    #import time
    #time.sleep(10)
    print("====================即将载入原始模型====================\n")
    #ori_model = load_checkpoints(ori_model, ori_config.model, strict=False)
    ori_ckpt = "/data/oss_bucket_0/yongfan/weights/DynamiCrafter/DynamiCrafter_512/model.ckpt"
    ori_model = torch.load(ori_ckpt, map_location="cpu")

    #print("====================即将打印原始模型====================\n")
    #for k, v in ori_model["state_dict"].items():
    #    if 'first_stage_model' in k:
    #        print(k)
    #with open('/data/oss_bucket_0/yongfan/weights/DynamiCrafter/DynamiCrafter_256/model_state_dict_1024.txt', 'w') as f:
    #    for k, v in ori_model.state_dict().items():
    #        # 打印键到文件
    #        f.write(f'{k}\n')
    #raise RuntimeError("中断程序")
    print("====================载入完成，即将替换vae权重====================\n")
    #  提取 VAE 的权重
    vae_weights = {k: v for k, v in ori_model["state_dict"].items() if 'first_stage_model' in k}
    #  替换键名中的 'first_stage_model.' 部分
    vae_weights = {k.replace('first_stage_model.', ''): v for k, v in vae_weights.items()}
    del ori_model

    # 这里故意引发一个异常以中断程序
    #raise RuntimeError("中断程序")


    ## MODEL CONFIG >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    logger.info("***** Configing Model *****")
    config.model.params.logdir = workdir
    model = instantiate_from_config(config.model)

    # 从原模型权重加载vae权重到要训练的模型中
    model.first_stage_model.load_state_dict(vae_weights, strict=True)
    #raise RuntimeError("中断程序")

    ## load checkpoints
    logger.info("Loading checkpoints from: %s" % config.model)
    model = load_checkpoints(model, config.model)
    #print("====================即将打印模型====================\n", model)
    #for k, v in model.state_dict().items():
    #    print(k)
    #raise RuntimeError("中断程序")

    ## register_schedule again to make ZTSNR work
    if model.rescale_betas_zero_snr:
        model.register_schedule(given_betas=model.given_betas, beta_schedule=model.beta_schedule, timesteps=model.timesteps,
                                linear_start=model.linear_start, linear_end=model.linear_end, cosine_s=model.cosine_s)

    ## update trainer config
    for k in get_nondefault_trainer_args(args):
        trainer_config[k] = getattr(args, k)
    #print("trainer_config:\n", trainer_config)
    trainer_config.num_nodes=8
    num_nodes = trainer_config.num_nodes
    ngpu_per_node = trainer_config.devices
    logger.info(f"Running on {num_rank}={num_nodes}x{ngpu_per_node} GPUs")
    #print("trainer_config:\n", trainer_config)

    ## setup learning rate
    base_lr = config.model.base_learning_rate
    bs = config.data.params.batch_size
    if getattr(config.model, 'scale_lr', True):
        model.learning_rate = num_rank * bs * base_lr
    else:
        model.learning_rate = base_lr

    ## DATA CONFIG >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    logger.info("***** Configing Data *****")
    data = instantiate_from_config(config.data)
    data.setup()
    for k in data.datasets:
        logger.info(f"{k}, {data.datasets[k].__class__.__name__}, {len(data.datasets[k])}")

    ## TRAINER CONFIG >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    logger.info("***** Configing Trainer *****")
    if "accelerator" not in trainer_config:
        trainer_config["accelerator"] = "gpu"

    ## setup trainer args: pl-logger and callbacks
    trainer_kwargs = dict()
    trainer_kwargs["num_sanity_val_steps"] = 0
    logger_cfg = get_trainer_logger(lightning_config, workdir, args.debug)
    trainer_kwargs["logger"] = instantiate_from_config(logger_cfg)
    
    ## setup callbacks
    callbacks_cfg = get_trainer_callbacks(lightning_config, config, workdir, ckptdir, logger)
    trainer_kwargs["callbacks"] = [instantiate_from_config(callbacks_cfg[k]) for k in callbacks_cfg]
    strategy_cfg = get_trainer_strategy(lightning_config)
    trainer_kwargs["strategy"] = strategy_cfg if type(strategy_cfg) == str else instantiate_from_config(strategy_cfg)
    trainer_kwargs['precision'] = lightning_config.get('precision', 32)
    trainer_kwargs["sync_batchnorm"] = False

    ## trainer config: others
    trainer_args = argparse.Namespace(**trainer_config)
    trainer = Trainer.from_argparse_args(trainer_args, **trainer_kwargs)

    ## allow checkpointing via USR1
    def melk(*args, **kwargs):
        ## run all checkpoint hooks
        if trainer.global_rank == 0:
            ckpt_path = os.path.join(ckptdir, "last_summoning.ckpt")
            logger.info(f"Summoning checkpoint at: {ckpt_path}")
            trainer.save_checkpoint(ckpt_path)

    def divein(*args, **kwargs):
        if trainer.global_rank == 0:
            import pudb
            pudb.set_trace()

    import signal
    signal.signal(signal.SIGUSR1, melk)
    signal.signal(signal.SIGUSR2, divein)

    ## Running LOOP >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    logger.info("********************** Running the Loop **********************")
    if args.train:
        try:
            if "strategy" in lightning_config and lightning_config['strategy'].startswith('deepspeed'):
                logger.info("<Training in DeepSpeed Mode>")
                ## deepspeed
                if trainer_kwargs['precision'] == 16:
                    with torch.cuda.amp.autocast():
                        trainer.fit(model, data)
                else:
                    trainer.fit(model, data)
            else:
                logger.info("<Training in DDPSharded Mode>") ## this is default
                ## ddpsharded
                print("=====================>start to train model.......")
                trainer.fit(model, data)
        except Exception as e:
            logger.error(f"Training failed with exception: {e}")
            raise

    # if args.val:
    #     trainer.validate(model, data)
    # if args.test or not trainer.interrupted:
    #     trainer.test(model, data)
