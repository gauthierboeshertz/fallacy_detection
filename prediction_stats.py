import argparse
import os

import string
import pandas as pd
import pickle
import numpy as np 

from sklearn.metrics import classification_report, plot_roc_curve
import matplotlib.pyplot as plt
import logging


def get_logger(log_file,level='DEBUG'):
    logger = logging.getLogger()
    logger.handlers = []
    file_log_handler = logging.FileHandler('logs/'+log_file+".log")
    logger.addHandler(file_log_handler)
    stderr_log_handler = logging.StreamHandler()
    #logger.addHandler(stderr_log_handler)
    logger.setLevel(level)
    return logger

def print_stats(pred_labels,log_file_path):

    print(pred_labels.shape)
    logger = get_logger(log_file=log_file_path,level="INFO")
    preds = pred_labels[0]
    y = pred_labels[1]
    labels = ["Non-fallacious","Fallacious"]

    report = classification_report(y, preds, target_names=labels)
    print(report)
    logger.info(report)

 
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-po", "--pred_path", help="path for output csv file")
    args = parser.parse_args()

    pl = np.load(args.pred_path)
    print_stats(pl,args.pred_path.split("/")[-1].split(".")[0])
