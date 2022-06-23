import argparse
import pytorch_lightning as pl
import yaml
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from modeling import *

def test(config):
    model = BaseModule(config=config)
    model.load_ckpt(config['save_path'])
    trainer = pl.Trainer()
    tokenizer = AutoTokenizer.from_pretrained(config['tokenizer_name'])

    test_ds = BaseTestDataset(tokenizer=tokenizer)
    

    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, drop_last=False, pin_memory=True,
                              num_workers=4)
    trainer.test(model,test_loader)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--config_path', type=str, default='')
    parser.add_argument('--save_path', type=str, default='')
    parser.add_argument('--model_name', type=str, default='bert-base-uncased')
    parser.add_argument('--tokenizer_name', type=str, default='bert-base-uncased')

    args = parser.parse_args()
    config = ""
    if args.config_path == '':
        config = vars(args)
    else:
        with open(args.config_path, "r") as stream:
            try:
                config = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)

    print('Config:')
    print(config)
    test(config)
