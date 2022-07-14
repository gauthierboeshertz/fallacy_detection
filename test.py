import argparse
import pytorch_lightning as pl
import yaml
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import torch
from argument_dataset import ArgumentDataset
import numpy as np
from base_module import BaseModule
import random
import os
os.environ["TRANSFORMERS_CACHE"] = "/cluster/scratch/gboeshertz/huggingface_cache"

random.seed(0)
torch.manual_seed(0)
np.random.seed(0)

def test(config):
    model = BaseModule(config=config)
    model.load_ckpt(config['test_save_path'])
    trainer = pl.Trainer()
    tokenizer = AutoTokenizer.from_pretrained(config['tokenizer_name'])

    test_loader = ArgumentDataset(config["test_data_path"],tokenizer,use_sa=config["use_sa"],use_hypothesis=config["use_hypothesis"]).get_dataloader(1,False)
    
    trainer.test(model,test_loader)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--config_path', type=str, default='')

    args = parser.parse_args()
    with open(args.config_path, "r") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    print('Config:')
    print(config)
    test(config)
