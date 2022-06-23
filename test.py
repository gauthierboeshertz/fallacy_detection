import argparse
import pytorch_lightning as pl
import yaml
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from modeling import *


def get_output_stats(filepath):

def test(config):
    model = BaseModule(config=config)
    model.load_ckpt(config['save_path'])
    trainer = pl.Trainer()
    tokenizer = AutoTokenizer.from_pretrained(config['tokenizer_name'])

    test_loader = ArgumentDataset(config["test_data_path"],tokenizer).get_dataloader(1,False)
    
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
