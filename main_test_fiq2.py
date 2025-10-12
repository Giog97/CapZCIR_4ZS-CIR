# Main che server per lanciare il train (o valutazione su split val)
import os
# os.environ["NCCL_P2P_DISABLE"] = "1"
# os.environ["NCCL_IB_DISABLE"] = "1"
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)
#torch.multiprocessing.set_sharing_strategy('file_system')
import torch
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch import nn
import random 
import numpy as np 
from trainer import Trainer
#from config import Config
from config_test_fiq import Config
import datetime
import wandb

from utils import get_model, set_grad, get_preprocess, get_laion_cirr_dataset, get_laion_fiq_dataset, get_laion_circo_dataset, extract_index_features, collate_fn, get_optimizer

import warnings
warnings.filterwarnings("ignore", category=UserWarning) 
from torchvision.io.image import ImageReadMode
warnings.filterwarnings("ignore", message="Failed to load image Python extension")

# per lanciare il codice dalla home:  python CapZCIR/main.py

def setup_seed(seed):
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	np.random.seed(seed)
	random.seed(seed)
	# torch.backends.cudnn.deterministic = True

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    """Destroy the process group after training."""
    dist.destroy_process_group()

def main(cfg): #rank, world_size,
    setup_seed(42)
    # setup(rank, world_size)
    # if rank == 0:

        # wandb.init(project='ZeroShot', notes=cfg.comment, config=wandb_config, name=cfg.comment)
    # model = model.to(rank)
    # model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    if cfg.load =='pretrained':
        model = get_model(cfg)
        set_grad(cfg, model)
        #model.pretrained_model.eval().float() #originale 1 text encoder
        # 2 text encoder
        model.caption_encoder.eval().float() #mod
        model.condition_encoder.eval().float() #mod
        total_params, trainable_params = count_parameters(model)
        print(f"Total Parameters: {total_params}")
        print(f"Trainable Parameters: {trainable_params}")
    else:
        model = get_model(cfg)
        set_grad(cfg, model)
        print("loading from trained model")
        model.load_state_dict(torch.load(cfg.eval_load_path))
        set_grad(cfg, model)
        #model.pretrained_model.eval().float() #originale 1 text encoder
        model.caption_encoder.eval().float() #mod 2 text encoder
        model.condition_encoder.eval().float() #mod 2 text encoder
        total_params, trainable_params = count_parameters(model)


        print(f"Total Parameters: {total_params}")
        print(f"Trainable Parameters: {trainable_params}")

    # input_dim = combiner.blip_model.visual.input_resolution
    if cfg.model_name.startswith('blip'):
        input_dim = 384
    elif cfg.model_name.startswith('clip'):
        # input_dim = model.module.pretrained_model.visual.input_resolution
        #input_dim = model.pretrained_model.visual.input_resolution #originale 1 text encoder
        input_dim = model.caption_encoder.visual.input_resolution #mod 2 text encoder
        #input_dim = model.condition_encoder.visual.input_resolution #mod 2 text encoder
    preprocess = get_preprocess(cfg, model, input_dim)

    if cfg.dataset == 'fiq':
        val_dress_types = ['dress', 'toptee', 'shirt']
        relative_train_dataset, relative_val_dataset, classic_val_dataset, idx_to_dress_mapping = get_laion_fiq_dataset(preprocess, val_dress_types, cfg.laion_type)
    # get dataset and dataloader
    elif cfg.dataset == 'cirr':
        relative_train_dataset, relative_val_dataset, classic_val_dataset = get_laion_cirr_dataset(preprocess, cfg.laion_type)
    elif cfg.dataset == 'circo':
        relative_train_dataset, relative_val_dataset, classic_val_dataset = get_laion_circo_dataset(preprocess, cfg.laion_type)

    # Parti commentate servono se si vuole usare DistributedDataParallel (DDP) con più GPU
    # train_sampler = DistributedSampler(relative_train_dataset, num_replicas = world_size, rank=rank, shuffle=True)
    # val_sampler = DistributedSampler(relative_val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    # classic_val_sampler = DistributedSampler(classic_val_dataset, num_replicas=world_size, rank=rank, shuffle=False)

    relative_train_loader = DataLoader(dataset=relative_train_dataset, batch_size=cfg.batch_size,
                                       num_workers=mp.cpu_count(), pin_memory=True,
                                       drop_last=True,shuffle=True, collate_fn=collate_fn) #sampler=train_sampler

    # relative_val_loader = DataLoader(
    #     dataset=relative_val_dataset, batch_size=cfg.batch_size, num_workers=8,
    #     pin_memory=True, drop_last=False,shuffle=False,  collate_fn=custom_circo_collate_fn
    # ) #sampler=val_sampler,
    #
    # classic_val_loader = DataLoader(
    #     dataset=classic_val_dataset, batch_size=cfg.batch_size, num_workers=8,
    #     pin_memory=True, drop_last=False, shuffle=False, collate_fn=collate_fn
    # ) #sampler=classic_val_sampler

    # When fine-tuning only the text encoder we can precompute the index features since they do not change over the epochs
    kwargs = {}
    if cfg.dataset == 'fiq':
        kwargs['val_index_features'] = []
        kwargs['val_index_names'] = []
        kwargs['val_total_index_features'] = []
        kwargs['idx_to_dress_mapping'] = idx_to_dress_mapping
    if cfg.dataset == 'cirr' and (cfg.encoder == 'text' or cfg.encoder == 'neither'):
        val_index_features, val_index_names, val_total_index_features = extract_index_features(classic_val_dataset, model, return_local=False)
        kwargs['val_index_features'], kwargs['val_index_names'], kwargs['val_total_index_features'] = val_index_features, val_index_names, val_total_index_features
    elif cfg.dataset == 'fiq' and (cfg.encoder == 'text' or cfg.encoder == 'neither'):
        for classic_val_dataset_ in classic_val_dataset:
            val_index_features, val_index_names, _ = extract_index_features(classic_val_dataset_, model, return_local=False)
            kwargs['val_index_features'].append(val_index_features)
            kwargs['val_index_names'].append(val_index_names)
            kwargs['val_total_index_features'].append(_)
    elif cfg.dataset == 'circo' and (cfg.encoder == 'text' or cfg.encoder == 'neither'):
        val_index_features, val_index_names, val_total_index_features = extract_index_features(classic_val_dataset, model, return_local=False)
        kwargs['val_index_features'], kwargs['val_index_names'], kwargs['val_total_index_features'] = val_index_features, val_index_names, val_total_index_features


    # Define the optimizer, the loss and the grad scaler
    optimizer = get_optimizer(model, cfg)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=cfg.num_epochs, eta_min=1e-2 * cfg.learning_rate, last_epoch=-1)
    crossentropy_criterion = nn.CrossEntropyLoss(ignore_index=-100)

    trainer = Trainer(cfg, model, relative_train_loader, optimizer, lr_scheduler, crossentropy_criterion, classic_val_dataset, relative_val_dataset, **kwargs) # rank
    #trainer.train() # commenta questo se voglio fare evaluation
    # Se voglio solo fare evaluation sul val set devo fare ad es:
    print(f"Loading trained model from {cfg.eval_load_path}")
    model.load_state_dict(torch.load(cfg.eval_load_path))
    trainer.eval_fiq()
    """
    if you just want to eval
        (1) model.load_state_dict(torch.load(model_path))
        (2) trainer.eval_cirr() or trainer.eval_fiq()
    """

if __name__ == '__main__':
    cfg = Config()
    now = datetime.datetime.now()
    current_time = now.strftime("%Y-%m-%d-%H-%M-%S")
    cfg.save_path = f"{cfg.save_path_prefix}/{current_time}_{cfg.comment}_best_arithmetic"

    wandb_config = vars(cfg)
    wandb.init(project='ZeroShot', notes=cfg.comment, config=wandb_config, name=cfg.comment)

    # Get number of available GPUs (world size)
    # world_size = torch.cuda.device_count()

    # Spawn processes for each GPU multiprocessing
    # mp.spawn(main,args=(world_size, cfg),
    #     nprocs=world_size,
    #     join=True,
    # )
    main(cfg)
    wandb.finish()

