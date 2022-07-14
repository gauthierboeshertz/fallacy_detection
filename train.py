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
random.seed(0)
torch.manual_seed(0)
np.random.seed(0)

trans_cache_dir =  "/cluster/scratch/gboeshertz/huggingface_cache"

os.environ["TRANSFORMERS_CACHE"] = trans_cache_dir

def get_model_path_weights(weight_dir):
    files = os.listdir(weight_dir)
    files = [f for f in files if ".ckpt" in f]
    paths = [os.path.join(weight_dir, basename) for basename in files]
    return max(paths, key=os.path.getctime)


def train(config):

    tokenizer = AutoTokenizer.from_pretrained(config["tokenizer_name"], do_lower_case=True,cache_dir =trans_cache_dir )

    train_ds = ArgumentDataset(config["train_data_path"],tokenizer,use_sa=config["use_sa"],use_hypothesis=config["use_hypothesis"])
    train_loader = train_ds.get_dataloader(config["batch_size"],True)

    val_loader = ArgumentDataset(config["val_data_path"],tokenizer,use_sa=config["use_sa"],use_hypothesis=config["use_hypothesis"]).get_dataloader(config["batch_size"],False)
    

    model = BaseModule(config=config,class_weights = train_ds.class_weights)
    callbacks = [EarlyStopping(monitor="val_loss", mode="min",verbose=True),
                 ModelCheckpoint(monitor='val_loss', dirpath=config['save_path'])]

    trainer = pl.Trainer(max_epochs=config['nepochs'], gpus= int(torch.cuda.is_available()), callbacks=callbacks,
                         check_val_every_n_epoch=config['val_freq'], gradient_clip_val=1)


    trainer.fit(model, train_loader, val_loader)
    
    test_loader = ArgumentDataset(config["test_data_path"],tokenizer,use_sa=config["use_sa"],use_hypothesis=config["use_hypothesis"]).get_dataloader(1,False)

    model.load_ckpt(get_model_path_weights(config["save_path"]))

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
    train(config)
