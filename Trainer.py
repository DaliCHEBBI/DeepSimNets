"""
Source code to train Multi Scale Self-attentional feature fusion feature extractor and the MLP classiier on top of it !

"""

# IMPORTS
import os,sys
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.profiler import SimpleProfiler
from utils.io import load_config_train
import utils.utils as utils
from pathlib import Path
from datasets.CubeDataset import StereoTrAerialDatasetDenseN, StereoValAerialDatasetDenseN 
from utils.logger import Logger
import torch
import numpy as np
import argparse
import typing
#import torch.multiprocessing
#torch.multiprocessing.set_sharing_strategy('file_system')
def MemStatus(loc):
    total_memory, used_memory, free_memory = map(
        int, os.popen('free -t -m').readlines()[-1].split()[1:])
    # Memory usage
    print("RAM memory % used before @ "+ loc, round((used_memory/total_memory) * 100, 2))


def make_arg_parser():
    parserG = argparse.ArgumentParser()
    parserG.add_argument(
        "--config_file",
        "-cfg",
        type=str,
        help="Path to the yaml config file",
        required=True,
    )
    parserG.add_argument(
        "--model",
        "-mdl",
        type=str,
        help="Model name to train, possible names are: 'MS-AFF', 'U-Net32', 'U-Net_Attention'",
        required=True,
    )
    parserG.add_argument(
        "--checkpoint",
        "-ckpt",
        type=str,
        help="Model checkpoint to load",
        required=True,
    )
    return parserG

def main(arg_strings: typing.Sequence[str]):
    arg_parser=make_arg_parser()
    args = arg_parser.parse_args(arg_strings)
    torch.backends.cudnn.benchmark=True
    _Model2train=str(args.model)
    _cfg_file=args.config_file
    _Model_ckpt=args.checkpoint
    print(_Model2train=="MS-AFF")
    if (_Model2train=='MS-AFF'):
        # feature extractor = MS-AFF
        # decision network  = MLP 
        from models.MSNETPl import MSNETWithDecisionNetwork_Dense_LM_N_2
        experiment_name = 'MSNET_DECISION_AERIAL_DENSE_NORM'
        print("actual training path,   ",Path().cwd())
        root_dir = Path().cwd() / 'trained_models'
        param = load_config_train(root_dir, experiment_name,config_path=_cfg_file)
        param.version=0
        logger = Logger(param)
        paramConfig=utils.dict_to_keyvalue(param) 
        if _Model_ckpt is not None:
            """./trained_models/MS_AFF_MLP/MS_AFF_MLP.ckpt"""
            MS_AFF_MLP=MSNETWithDecisionNetwork_Dense_LM_N_2.load_from_checkpoint(_Model_ckpt)
        else:
            MS_AFF_MLP=MSNETWithDecisionNetwork_Dense_LM_N_2(32,0,1,4,0.0)
             # true 0 false 1 --> 4 offset intervals from which to sample negative examples
        MS_AFF_MLP=MS_AFF_MLP.cuda() 
        Train_dataset=StereoTrAerialDatasetDenseN(paramConfig['dataset.path_train'],
                                                  0.434583236,0.1948717255,paramConfig['dataset.name'])
        MemStatus("training dataset")
        # The dataset declaration lacks more information on augmentation
        Val_dataset=StereoValAerialDatasetDenseN(paramConfig['dataset.path_val'],
                                                 0.434583236,0.1948717255,paramConfig['dataset.name'])
        checkpointVal = ModelCheckpoint(
            dirpath=logger.log_files_dir,
            save_top_k=5,
            verbose=True,
            monitor='val_loss',
            mode='min'
            #prefix=param.experiment_name
        )
        checkpointTrain = ModelCheckpoint(
            dirpath=logger.log_files_dir,
            save_top_k=5,
            verbose=True,
            monitor='training_loss',
            mode='min'
        )
        profiler=SimpleProfiler()
        trainer = pl.Trainer(
            #profiler=profiler,
            enable_model_summary=True,
            logger=logger,
            max_epochs=param.train.epochs,
            callbacks=[checkpointTrain,checkpointVal, LearningRateMonitor("epoch")],
            check_val_every_n_epoch=param.logger.log_validation_every_n_epochs,
            val_check_interval=0.5,
            accumulate_grad_batches=param.train.accumulate_grad_batches,
            track_grad_norm=2,
            strategy='ddp',
            gpus=param.n_gpus,
            precision=16,
            #auto_lr_find=True,
        )
        #trainer.logger._log_graph=True
        pl.seed_everything(42)
        MemStatus("MemStatus before training loader")
        train_loader=torch.utils.data.DataLoader(Train_dataset, batch_size=param.train.bs, 
                                                 shuffle=True,drop_last=True,pin_memory=True, 
                                                 num_workers=param.train.num_workers)
        MemStatus("MemStatus after training loader")
        val_loader=torch.utils.data.DataLoader(Val_dataset, batch_size=param.val.bs, 
                                               shuffle=False,drop_last=True,pin_memory=True,
                                                 num_workers=param.val.num_workers)
        MemStatus("MemStatus after validation loader")
        #trainer.tune(UnetMlpDense,train_loader,val_loader)
        trainer.fit(MS_AFF_MLP, train_loader, val_loader)

    elif (_Model2train=="U-Net32"):
        # feature extractor = U-Net 32
        # decision network  = MLP
        from models.UNetDecisionEBM import UNETWithDecisionNetwork_Dense_LM_N_2
        experiment_name = 'UNET_DECISION_AERIAL_DENSE_NORM'
        print("actual training path,   ",Path().cwd())
        root_dir = Path().cwd() / 'trained_models'
        param = load_config_train(root_dir, experiment_name,config_path=_cfg_file)
        param.version=0
        logger = Logger(param)
        paramConfig=utils.dict_to_keyvalue(param) 
        if _Model_ckpt is not None:
            """./trained_models/U-Net32_MLP/U-Net32_MLP.ckpt"""
            U_NET32_MLP=UNETWithDecisionNetwork_Dense_LM_N_2.load_from_checkpoint(_Model_ckpt)
        else:
            U_NET32_MLP=UNETWithDecisionNetwork_Dense_LM_N_2(32,0,1,4,0.0) 
            # true 0 false 1 --> 4 offset intervals from which to sample negative examples
        U_NET32_MLP=U_NET32_MLP.cuda() 
        Train_dataset=StereoTrAerialDatasetDenseN(paramConfig['dataset.path_train'],
                                                  0.434583236,0.1948717255,paramConfig['dataset.name'])
        MemStatus("training dataset")
        # The dataset declaration lacks more information on augmentation
        Val_dataset=StereoValAerialDatasetDenseN(paramConfig['dataset.path_val'],
                                                 0.434583236,0.1948717255,paramConfig['dataset.name'])
        checkpointVal = ModelCheckpoint(
            dirpath=logger.log_files_dir,
            save_top_k=5,
            verbose=True,
            monitor='val_loss',
            mode='min'
            #prefix=param.experiment_name
        )
        checkpointTrain = ModelCheckpoint(
            dirpath=logger.log_files_dir,
            save_top_k=5,
            verbose=True,
            monitor='training_loss',
            mode='min'
        )
        profiler=SimpleProfiler()
        trainer = pl.Trainer(
            #profiler=profiler,
            enable_model_summary=True,
            logger=logger,
            max_epochs=param.train.epochs,
            callbacks=[checkpointTrain,checkpointVal, LearningRateMonitor("epoch")],
            check_val_every_n_epoch=param.logger.log_validation_every_n_epochs,
            val_check_interval=0.5,
            accumulate_grad_batches=param.train.accumulate_grad_batches,
            track_grad_norm=2,
            strategy='ddp',
            gpus=param.n_gpus,
            precision=16,
            #auto_lr_find=True,
        )
        #trainer.logger._log_graph=True
        pl.seed_everything(42)
        MemStatus("MemStatus before training loader")
        train_loader=torch.utils.data.DataLoader(Train_dataset, batch_size=param.train.bs, 
                                                 shuffle=True,drop_last=True,pin_memory=True, 
                                                 num_workers=param.train.num_workers)
        MemStatus("MemStatus after training loader")
        val_loader=torch.utils.data.DataLoader(Val_dataset, batch_size=param.val.bs, 
                                               shuffle=False,drop_last=True,pin_memory=True,
                                                 num_workers=param.val.num_workers)
        MemStatus("MemStatus after validation loader")
        #trainer.tune(UnetMlpDense,train_loader,val_loader)
        trainer.fit(U_NET32_MLP, train_loader, val_loader)
    elif (_Model2train=="U-Net_Attention"):
        # feature extractor = U-Net Attention
        # decision network  = MLP
        from models.unetGatedAttention import UNETGATTWithDecisionNetwork_Dense_LM_N_2
        experiment_name = 'UNET_GATED_ATTENTION_DECISION_AERIAL_DENSE_NORM'
        print("actual training path,   ",Path().cwd())
        root_dir = Path().cwd() / 'trained_models'
        param = load_config_train(root_dir, experiment_name,config_path=_cfg_file)
        param.version=0
        logger = Logger(param)
        paramConfig=utils.dict_to_keyvalue(param) 
        if _Model_ckpt is not None:
            """./trained_models/U-Net_Attention_MLP/U-Net_Attention_MLP.ckpt"""
            U_NET_ATTENTION_MLP=UNETGATTWithDecisionNetwork_Dense_LM_N_2.load_from_checkpoint(_Model_ckpt)
        else:
            U_NET_ATTENTION_MLP=UNETGATTWithDecisionNetwork_Dense_LM_N_2(32,0,1,4,0.0) 
            # true 0 false 1 --> 4 offset intervals from which to sample negative examples
        U_NET_ATTENTION_MLP=U_NET_ATTENTION_MLP.cuda() 
        Train_dataset=StereoTrAerialDatasetDenseN(paramConfig['dataset.path_train'],
                                                  0.434583236,0.1948717255,paramConfig['dataset.name'])
        MemStatus("training dataset")
        # The dataset declaration lacks more information on augmentation
        Val_dataset=StereoValAerialDatasetDenseN(paramConfig['dataset.path_val'],
                                                 0.434583236,0.1948717255,paramConfig['dataset.name'])
        checkpointVal = ModelCheckpoint(
            dirpath=logger.log_files_dir,
            save_top_k=5,
            verbose=True,
            monitor='val_loss',
            mode='min'
            #prefix=param.experiment_name
        )
        checkpointTrain = ModelCheckpoint(
            dirpath=logger.log_files_dir,
            save_top_k=5,
            verbose=True,
            monitor='training_loss',
            mode='min'
        )
        profiler=SimpleProfiler()
        trainer = pl.Trainer(
            #profiler=profiler,
            enable_model_summary=True,
            logger=logger,
            max_epochs=param.train.epochs,
            callbacks=[checkpointTrain,checkpointVal, LearningRateMonitor("epoch")],
            check_val_every_n_epoch=param.logger.log_validation_every_n_epochs,
            val_check_interval=0.5,
            accumulate_grad_batches=param.train.accumulate_grad_batches,
            track_grad_norm=2,
            strategy='ddp',
            gpus=param.n_gpus,
            precision=16,
            #auto_lr_find=True,
        )
        #trainer.logger._log_graph=True
        pl.seed_everything(42)
        MemStatus("MemStatus before training loader")
        train_loader=torch.utils.data.DataLoader(Train_dataset, batch_size=param.train.bs, 
                                                 shuffle=True,drop_last=True,pin_memory=True, 
                                                 num_workers=param.train.num_workers)
        MemStatus("MemStatus after training loader")
        val_loader=torch.utils.data.DataLoader(Val_dataset, batch_size=param.val.bs, 
                                               shuffle=False,drop_last=True,pin_memory=True,
                                                 num_workers=param.val.num_workers)
        MemStatus("MemStatus after validation loader")
        #trainer.tune(UnetMlpDense,train_loader,val_loader)
        trainer.fit(U_NET_ATTENTION_MLP, train_loader, val_loader)

    else:
        raise RuntimeError("Model name should be one of  these models : 'MS-AFF', 'U-Net32', 'U-Net_Attention'")


if __name__ == '__main__':
    main(sys.argv[1:])