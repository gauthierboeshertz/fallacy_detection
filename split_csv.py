import argparse
import os

import string
import pandas as pd
import pickle
import re
import logging
import traceback
import numpy as np
import random
from sklearn.utils import shuffle


random.seed(0)
np.random.seed(0)

def split_csv(df_path,splits):
    """
    strat=1 -> only original article
    strat=2 -> only masked article
    strat=3 -> both
    """

    df_name = df_path.split(".")[0]
    df = pd.read_csv(df_path)

    df = shuffle(df)

    print("Original shape",df.shape)
    if len(splits) == 2:
        cut_off_length = int(df.shape[0]*splits[0])
        df_train, df_val = df[:cut_off_length],df[cut_off_length:] 

        print("Train shape",df_train.shape)
        print("Val shape",df_val.shape)
        df_train.to_csv(df_name+"_train.csv")
        df_val.to_csv(df_name+"_val.csv")

    if len(splits) == 3:
        train_cut_off_length = int(df.shape[0]*splits[0])
        val_cut_off_length = int(df.shape[0]*splits[1])
        df_train, df_val,df_test = df[:train_cut_off_length],df[train_cut_off_length:train_cut_off_length+val_cut_off_length],df[train_cut_off_length+val_cut_off_length:]
        print("Train shape",df_train.shape)
        print("Val shape",df_val.shape)
        print("test shape",df_test.shape)
        df_train.to_csv(df_name+"_train.csv",index=False)
        df_val.to_csv(df_name+"_val.csv",index=False)
        df_test.to_csv(df_name+"_test.csv",index=False)

    return 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", help="path for input csv file")
    parser.add_argument("-s", "--splits", default= "[0.8,0.1,0.1]",help="path for input csv file")


    args = parser.parse_args()
    split_csv(args.path,eval(args.splits))
