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
from transformers import pipeline
import pandas as pd
random.seed(0)
torch.manual_seed(0)
np.random.seed(0)
from sklearn.metrics import classification_report, plot_roc_curve
from prediction_stats import print_stats

import os
os.environ["TRANSFORMERS_CACHE"] = "/cluster/scratch/gboeshertz/huggingface_cache/"

def zero_shot(config):

    classifier = pipeline("zero-shot-classification", model=config["model_name"], device= int(torch.cuda.is_available()) - 1)

    test_df =  pd.read_csv(config["test_data_path"])
    test_df = test_df.dropna()
    test_df = test_df.drop_duplicates()

    test_sentences = test_df["argument"]
    test_sentences_labels = test_df["is_fallacy"]
    hypothesis_template = "This argument is {}"
    labels = ["valid","fallacious"]

    pred_and_labels = []

    for  sentence, sentence_label  in  zip(test_sentences,test_sentences_labels):
        pred = classifier(sentence,labels,hypothesis_template=hypothesis_template,multi_label=False)
        pred_label = int(pred["labels"][0] == "fallacious")
        pred_and_labels.append([pred_label,sentence_label])

    y = [pnl[1] for pnl in pred_and_labels]
    preds = [pnl[0] for pnl in pred_and_labels]

    print_stats(np.array([preds,y]), config["test_out_path"])
    report = classification_report(y, preds, target_names=labels)
    print(report)


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
    zero_shot(config)
