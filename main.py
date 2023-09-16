import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import torchvision
from timm.optim import create_optimizer_v2
from timm.utils import ModelEmaV2
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import accuracy_score,confusion_matrix,roc_auc_score,roc_curve,mean_absolute_error,f1_score,r2_score
import argparse
import wandb
import os
import numpy as np
import warnings
import builtins
import math
import datetime
from omegaconf import OmegaConf
import omegaconf
from omegaconf.dictconfig import DictConfig
from itertools import product
from tqdm import tqdm
from imblearn.metrics import specificity_score
import warnings
import copy
from torch_ema import ExponentialMovingAverage
import collections
from dataset import medicalDataset,medicalDataset_task1
from model import Model
from mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
import misc
from sklearn.metrics import cohen_kappa_score
import torch.nn.functional as F
from util import calculate_dice_coefficient
import monai
from loss import MultiTaskLossWrapper, SupConLoss
import timm
import random

def adjust_learning_rate(optimizer, end_epoch, current_epoch, lrs,warmup=5):
    """Decays the learning rate with half-cycle cosine after warmup"""
    if current_epoch < warmup:
        lr_rate =  (current_epoch+1) / warmup
    else:
        lr_rate =  0.5 * (1. + math.cos(math.pi * (current_epoch - warmup) / (end_epoch - warmup)))
    
    temp = []
    for i,param_group in enumerate(optimizer.param_groups):
        param_group['lr'] = lr_rate*lrs[i]
        temp.append(lr_rate*lrs[i])
        
    return temp


def main_worker(gpu, ngpus_per_node, parser):
    warnings.filterwarnings('ignore')
    parser.gpu = gpu
    # suppress printing if not master
    if parser.multiprocessing_distribute and parser.gpu != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass
    if parser.gpu is not None:
        print("Use GPU: {} for training".format(parser.gpu))

    if parser.distribute:
        if parser.dist_url == "env://" and parser.rank == -1:
            parser.rank = int(os.environ["RANK"])
        if parser.multiprocessing_distribute:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            parser.rank = parser.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=parser.dist_backend, init_method=parser.dist_url,
                                world_size=parser.world_size, rank=parser.rank)
    
    if parser.gpu==0:
        """
        set experiment name
        """
        experiment_name = 'exp_'+str(datetime.datetime.now()).split('.')[0]
        os.mkdir('/data/mmac_multitask/exp/'+experiment_name)
        #with open('/data/mmac_multitask/exp/'+experiment_name+'/config.yaml', "w") as f:
        #    OmegaConf.save(conf, f)
        if parser.wandb:
            wandb.init(project='mmac_multitask',entity='medi-whale',name=experiment_name)
            wandb.config.update(parser)
# create model
    model = Model(parser.arch)
    criterion = MultiTaskLossWrapper()
    if parser.distribute:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if parser.gpu is not None:
            torch.cuda.set_device(parser.gpu)
            model.cuda(parser.gpu)
            criterion.cuda(parser.gpu)
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            parser.batch_size = int(parser.batch_size / ngpus_per_node)
            parser.num_workers = int((parser.num_workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[parser.gpu],find_unused_parameters=True)
            criterion = torch.nn.parallel.DistributedDataParallel(criterion, device_ids=[parser.gpu],find_unused_parameters=True)
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model,find_unused_parameters=True)
    else:
        model.cuda()

    scaler = None
    if parser.amp:
        scaler = GradScaler()
        if parser.rank == 0:
            print('Using native Torch AMP. Training in mixed precision.')

    param_dicts = [
        {
            "params": model.parameters(),
            "lr" : float(parser.lr)
        },
        {
            "params": criterion.parameters(),
            "lr": float(parser.lr)
        },]

    lrs = [p['lr'] for p in param_dicts]
    
    base_optimizer = None
    optimizer = None
        
    device = torch.device("cuda:"+str(parser.gpu))
    optimizer = create_optimizer_v2(param_dicts,'lookahead_adamw', lr=parser.lr, weight_decay=parser.weight_decay)#, layer_decay = 0.8)
    ema_model = timm.utils.ModelEmaV2(model,decay = 0.995)

    mixup_fn = None
    mixup_active = parser.mixup > 0 or parser.cutmix > 0. or parser.cutmix_minmax is not None
    if mixup_active:
        print("Mixup is activated!")
        mixup_fn = Mixup(
            mixup_alpha=parser.mixup, cutmix_alpha=parser.cutmix, cutmix_minmax=parser.cutmix_minmax,
            prob=parser.mixup_prob, switch_prob=parser.mixup_switch_prob, mode=parser.mixup_mode,
            label_smoothing=-1, num_classes=-1)
    

    best_score  = 0
    
    train_dataset = medicalDataset(0,parser.split)
    val_dataset = medicalDataset(0,parser.split,False)
    train_supcon_dataset = medicalDataset_task1(0,parser.split)

    #data_distribution = (train_dataset.data_distribution).cuda()

    if parser.distribute:
        train_dataset_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        train_dataset_sampler2 = torch.utils.data.distributed.DistributedSampler(train_supcon_dataset)
        val_dataset_sampler = None#torch.utils.data.distributed.DistributedSampler(val_dataset)
        
    else:
        train_dataset_sampler = None
        train_dataset_sampler2 = None
        val_dataset_sampler = None
    
    import random
    seed = random.randint(1,100000)
    train_dataset_dataloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=parser.batch_size, shuffle=(train_dataset_sampler is None),
    num_workers=parser.num_workers, pin_memory=True, sampler=train_dataset_sampler, drop_last=True,prefetch_factor=4,persistent_workers = True)

    train_dataset_dataloader_supcon = torch.utils.data.DataLoader(
    train_supcon_dataset, batch_size=parser.batch_size//2, shuffle=(train_dataset_sampler2 is None),
    num_workers=parser.num_workers, pin_memory=True, sampler=train_dataset_sampler2, drop_last=True,prefetch_factor=4,persistent_workers = True)

    val_dataset_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=16, shuffle=False,
        num_workers=parser.num_workers, pin_memory=True, sampler=val_dataset_sampler, drop_last=False,persistent_workers = True)
    

    
    
    patient = 100
    current_patient = 0
    prev_val_loss = 100

    for epoch in range(parser.epoch):
        
        temp_lrs = adjust_learning_rate(optimizer, parser.epoch ,epoch, lrs,parser.warmup)

        train_dataset_sampler.set_epoch(epoch+seed)
        train_dataset_dataloader.dataset.resample(epoch+seed)
        train_dataset_dataloader_supcon.dataset.resample(epoch+seed)

        wandb_dict_train = train_one_epoch(model,ema_model,criterion,mixup_fn,train_dataset_dataloader,train_dataset_dataloader_supcon,optimizer,device,scaler,epoch)
        wandb_dict_val = evaluate(model,ema_model,criterion, val_dataset_dataloader ,device,scaler)
        
        wandb_dict_train.update(wandb_dict_val)
        wandb_dict_train['lr'] = temp_lrs[0]
        wandb_dict_train['adjust_lr'] = temp_lrs[1]

        if parser.gpu==0:
            weight_dict = None
            weight_dict = {
                'epoch': epoch,
                'model_state_dict': ema_model.state_dict(),}
        
            torch.save(weight_dict,'/data/mmac_multitask/exp/'+experiment_name+'/weight_'+str(epoch)+'.pth')
            if parser.wandb:
                wandb.log(wandb_dict_train)
        if wandb_dict_val['validation_loss']<prev_val_loss:
            prev_val_loss = wandb_dict_val['validation_loss']
            current_patient = 0
        else:
            prev_val_loss = wandb_dict_val['validation_loss']
            current_patient = current_patient +1
        if current_patient==patient:
            break
    if parser.wandb:
        wandb.finish()
    return True

def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt

def train_one_epoch(model,ema_model,criterion,mixup_fn,train_dataset_dataloader,train_dataset_dataloader_supcon,optimizer,device,scaler,epoch):
    model.train()
    criterion.train()

    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 5
    optimizer.zero_grad()
    total_loss = 0
    total_loss_1 = 0
    total_loss_2 = 0
    total_loss_3 = 0
    loss = None

    supconloss = SupConLoss(temperature=0.1)

    for samples in metric_logger.log_every(train_dataset_dataloader, print_freq, header):
        
        imgs = samples['img'].to(device,non_blocking=True)

        task1_target = samples['task1_target'].to(device,non_blocking=True)
        task2_target = samples['task2_target'].to(device,non_blocking=True)
        task2_target = F.one_hot(task2_target,num_classes=2).permute(0,3,1,2).to(torch.float32)
        task3_target = samples['task3_target'].to(device,non_blocking=True)

        task = samples['task'].to(device,non_blocking=True)
        lam = np.random.beta(0.8, 0.8)
    
        if scaler is not None:
            with autocast():
                outputs = model(imgs,task,lam)
                loss_task1, loss_task21, loss_task22 , loss_task23, loss_task3 = criterion(outputs,task1_target,task2_target,task3_target,task,lam)

            loss = loss_task1+ loss_task21+ loss_task22 + loss_task23 + loss_task3
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            ema_model.update(model)
        else:
            outputs = model(imgs,task,lam)
            loss_task1, loss_task21, loss_task22 , loss_task23,loss_task3 = criterion(outputs,task1_target,task2_target,task3_target,task,lam)
            loss = loss_task1+ loss_task21+ loss_task22 + loss_task23 + loss_task3
            loss.backward()
            
            optimizer.step()
            optimizer.zero_grad()
            ema_model.update(model)

        loss = loss_task1+loss_task21+loss_task22+loss_task23+loss_task3
        loss = reduce_tensor(loss)
        total_loss += float(loss)
 
        metric_logger.update(loss=float(loss))
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        torch.cuda.synchronize()
        
    total_loss = total_loss/len(train_dataset_dataloader)

    print(min_lr, max_lr)


    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {"train_loss":total_loss}

@torch.no_grad()
def evaluate(model,ema_model,criterion, val_dataset_dataloader ,device,scaler):
    model.eval()
    criterion.eval()
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    outputs_task1_list =[]
    targets_task1_list =[]

    outputs_task3_list =[]
    targets_task3_list =[]

    total_loss = 0
    total_loss_1 = 0
    total_loss_2 = 0
    total_loss_3 = 0
    loss = None

    dice_metirc1 = monai.metrics.DiceMetric(include_background=False,reduction='none')
    dice_metirc2 = monai.metrics.DiceMetric(include_background=False,reduction='none')
    dice_metirc3 = monai.metrics.DiceMetric(include_background=False,reduction='none')


    
    for samples in metric_logger.log_every(val_dataset_dataloader, 10, header):

        imgs = samples['img'].to(device,non_blocking=True)

        task1_target = samples['task1_target'].to(device,non_blocking=True)
        task2_target = samples['task2_target'].to(device,non_blocking=True)
        task2_target = F.one_hot(task2_target,num_classes=2).permute(0,3,1,2).to(torch.float32)
        task3_target = samples['task3_target'].to(device,non_blocking=True)
        task3_target_se = samples['task3_se'].to(device,non_blocking=True)

        task = samples['task'].to(device,non_blocking=True)

        if scaler is not None:
            with autocast():
                outputs = ema_model.module(imgs,task,1)
                loss_task1, loss_task21, loss_task22 , loss_task23,loss_task3 = criterion(outputs,task1_target,task2_target,task3_target,task,1)
                outputs_task1,outputs_task21,outputs_task22,outputs_task23,outputs_task3,_ = outputs
        else:
            
            outputs = ema_model.module(imgs,task,1)
            loss_task1, loss_task21, loss_task22 , loss_task23,loss_task3 = criterion(outputs,task1_target,task2_target,task3_target,task,1)
            outputs_task1,outputs_task21,outputs_task22,outputs_task23,outputs_task3,_ = outputs

        loss_task2 = loss_task21+ loss_task22 +loss_task23
        loss = loss_task1+loss_task2 + loss_task3
        loss = reduce_tensor(loss)
        if loss_task1!=0:
            loss_task1 = reduce_tensor(loss_task1)
        if loss_task2!=0:
            loss_task2 = reduce_tensor(loss_task2)
        if loss_task3!=0:
            loss_task3 = reduce_tensor(loss_task3)
        total_loss += float(loss)
        total_loss_1 += float(loss_task1)
        total_loss_2 += float(loss_task2)
        total_loss_3 += float(loss_task3)
        metric_logger.update(loss=float(loss))

        outputs_task1 = outputs_task1[task==1]
        outputs_task3 = outputs_task3[task==3]

        outputs_task21 = outputs_task21[task==21]
        outputs_task22 = outputs_task22[task==22]
        outputs_task23 = outputs_task23[task==23]

        task1_target = task1_target[task==1]
        task3_target = task3_target[task==3]
        task3_target_se = task3_target_se[task==3]

        task2_target = torch.argmax(task2_target,dim=1)

        task21_target = task2_target[task==21]
        task22_target = task2_target[task==22]
        task23_target = task2_target[task==23]

        if len(outputs_task1)>0:
            _, outputs_task1 = torch.max(outputs_task1, 1)
            outputs_task1 = (outputs_task1).detach().cpu().to(torch.int64)

            _, task1_target = torch.max(task1_target, 1)
            task1_target = (task1_target).cpu().to(torch.int64)

            outputs_task1_list.extend(outputs_task1.tolist())
            targets_task1_list.extend(task1_target.tolist())
        
        if len(outputs_task3)>0:
            outputs_task3 = (torch.sigmoid(outputs_task3) *20) - 10
            outputs_task3_list.extend((outputs_task3).detach().cpu().tolist())
            targets_task3_list.extend((task3_target_se).cpu().tolist())

        if len(outputs_task21)>0:
            outputs_task21 = torch.argmax(outputs_task21,dim=1)
            dice_metirc1(outputs_task21,task21_target)
        
        if len(outputs_task22)>0:
            outputs_task22 = torch.argmax(outputs_task22,dim=1)
            dice_metirc2(outputs_task22,task22_target)
        
        if len(outputs_task23)>0:
            outputs_task23 = torch.argmax(outputs_task23,dim=1)
            dice_metirc3(outputs_task23,task23_target)
        
        torch.cuda.synchronize()

    total_loss = total_loss/len(val_dataset_dataloader)
    total_loss_1 = total_loss_1/len(val_dataset_dataloader)
    total_loss_2 = total_loss_2/len(val_dataset_dataloader)
    total_loss_3 = total_loss_3/len(val_dataset_dataloader)

    y_pred_task1 = np.array(outputs_task1_list)
    y_true_task1 = np.array(targets_task1_list)

    qwk_score = cohen_kappa_score(y_true_task1,y_pred_task1,weights='quadratic')
    f1_result = f1_score(y_true_task1,y_pred_task1,average='macro')
    f1_class0,f1_class1,f1_class2,f1_class3,f1_class4 = f1_score(y_true_task1,y_pred_task1,average=None)
    specificity_result = specificity_score(y_true_task1, y_pred_task1, average='macro')


    y_pred_task3 = np.array(outputs_task3_list)
    y_true_task3 = np.array(targets_task3_list)

    r2_score_result = r2_score(y_true_task3,y_pred_task3)
    mae_score = mean_absolute_error(y_true_task3,y_pred_task3)

    qwk_score = (torch.ones(1)*qwk_score).cuda()
    qwk_score = float(reduce_tensor(qwk_score))
    f1_result = (torch.ones(1)*f1_result).cuda()
    f1_result = float(reduce_tensor(f1_result))
    specificity_result = (torch.ones(1)*specificity_result).cuda()
    specificity_result = float(reduce_tensor(specificity_result))
    f1_class0 = (torch.ones(1)*f1_class0).cuda()
    f1_class0 = float(reduce_tensor(f1_class0))
    f1_class1 = (torch.ones(1)*f1_class1).cuda()
    f1_class1 = float(reduce_tensor(f1_class1))
    f1_class2 = (torch.ones(1)*f1_class2).cuda()
    f1_class2 = float(reduce_tensor(f1_class2))
    f1_class3 = (torch.ones(1)*f1_class3).cuda()
    f1_class3 = float(reduce_tensor(f1_class3))
    f1_class4 = (torch.ones(1)*f1_class4).cuda()
    f1_class4 = float(reduce_tensor(f1_class4))

    r2_score_result = (torch.ones(1)*r2_score_result).cuda()
    r2_score_result = float(reduce_tensor(r2_score_result))
    mae_score = (torch.ones(1)*mae_score).cuda()
    mae_score = float(reduce_tensor(mae_score))

    dice_task21 = (dice_metirc1.aggregate('mean'))
    dice_task21 = float(reduce_tensor(dice_task21))

    dice_task22 = (dice_metirc2.aggregate('mean'))
    dice_task22 = float(reduce_tensor(dice_task22))

    dice_task23 = (dice_metirc3.aggregate('mean'))
    dice_task23 = float(reduce_tensor(dice_task23))

    task2_score = (dice_task21+dice_task22+dice_task23)/3.0

    task1_score = (qwk_score+f1_result+specificity_result)/3.0
    metric_logger.meters['qwk_score'].update(qwk_score)
    metric_logger.synchronize_between_processes()

    print({"validation_loss":total_loss,"qwk_score":qwk_score,'f1_score':f1_result,'specificity_score':specificity_result,'task1_score':task1_score,'f1_class0':f1_class0,'f1_class1':f1_class1,'f1_class2':f1_class2
    ,'f1_class3':f1_class3,'f1_class4':f1_class4,
    'dice_task21':dice_task21,'dice_task22':dice_task22,'dice_task23':dice_task23,'task2_score':task2_score,'mae':mae_score,'r2_score':r2_score_result,
    'task1_loss':total_loss_1,'task2_loss':total_loss_2,'task3_loss':total_loss_3})

    return {"validation_loss":total_loss,"qwk_score":qwk_score,'f1_score':f1_result,'specificity_score':specificity_result,'task1_score':task1_score,'f1_class0':f1_class0,'f1_class1':f1_class1,'f1_class2':f1_class2
    ,'f1_class3':f1_class3,'f1_class4':f1_class4,
    'dice_task21':dice_task21,'dice_task22':dice_task22,'dice_task23':dice_task23,'task2_score':task2_score,'mae':mae_score,'r2_score':r2_score_result,
    'task1_loss':total_loss_1,'task2_loss':total_loss_2,'task3_loss':total_loss_3
    }

def get_args_parser():
    parser = argparse.ArgumentParser('mmac_multitask', add_help=False)

    parser.add_argument('--batch_size', default=16, type=int,)
    parser.add_argument('--lr', default=2e-4, type=float,)
    parser.add_argument('--weight_decay', default=1e-5, type=float,)
    parser.add_argument('--epoch', default=50, type=int,)
    parser.add_argument('--warmup', default=5, type=int,)

    parser.add_argument('--split', default=0, type=int,)
    parser.add_argument('--num_workers', default=48, type=int,)
    
    parser.add_argument('--arch', default="convnext_small.fb_in22k_ft_in1k_384", type=str,)

    parser.add_argument('--wandb', action="store_true")
    parser.add_argument('--amp', action="store_true")


    ##########################

    parser.add_argument('--mixup', type=float, default=0.8,
                        help='mixup alpha, mixup enabled if > 0.')
    parser.add_argument('--cutmix', type=float, default=1.0,
                        help='cutmix alpha, cutmix enabled if > 0.')
    parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup_prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup_switch_prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup_mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')
    ##########################

    parser.add_argument('--distrubute', default=False, type=bool,)
    parser.add_argument('--multiprocessing_distribute', default=True, type=bool,)
    parser.add_argument('--world_size', default=1, type=int,)
    parser.add_argument('--dist_url', default="tcp://localhost:10001", type=str,)
    parser.add_argument('--dist_backend', default="nccl", type=str,)
    parser.add_argument('--gpu', default=0, type=int,)
    parser.add_argument('--rank', default=0, type=int,)

    return parser

def main():
    parser = get_args_parser()
    parser = parser.parse_args()

    if parser.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if parser.dist_url == "env://" and parser.world_size == -1:
        parser.world_size= int(os.environ["WORLD_SIZE"])

    parser.distribute = parser.world_size > 1 or parser.multiprocessing_distribute
    ngpus_per_node = torch.cuda.device_count()
    if parser.distribute:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        parser.world_size = ngpus_per_node * parser.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, parser))
    else:
        # Simply call main_worker function
        main_worker(parser.gpu, ngpus_per_node, parser)

if __name__ == '__main__':
    main()
