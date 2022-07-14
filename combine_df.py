import argparse
import os

import string
import pandas as pd
import pickle
import re
import logging
import traceback

import warnings

def fxn():
    warnings.warn("FutureWarning", FutureWarning)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()
import warnings
warnings.filterwarnings("ignore")

 
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-p1", "--path1", help="path for 1st input csv file")
    parser.add_argument("-p2", "--path2", help="path for 2ns input csv file")
    parser.add_argument("-po", "--outpath", help="path for output csv file")

    args = parser.parse_args()
    df1 = pd.read_csv(args.path1)
    df2 = pd.read_csv(args.path2)
    print("First df shape", df1.shape)
    print("Second df shape",df2.shape)
    #dfo = pd.concat([df1, df2], ignore_index=True)
    if df1.shape[0] < df2.shape[0]:

        arg_list = df2["argument"]
        for i, row in df1.iterrows():
            if row["argument"] in arg_list:
                print("Argument", row["argument"], "already in the argument list")
            else:
                #df2 = pd.concat([df2, row],ignore_index=True)
                df2 = df2.append(row)
        print("Output df shape",df2.shape)
        df2.to_csv(args.outpath,index=False)

    else:
        arg_list = df1["argument"]
        for i, row in df2.iterrows():
            if row["argument"] in arg_list:
                print("Argument", row["argument"], "already in the argument list")
            else:
                #df1 = pd.concat([df1, row],ignore_index=True)
                df1 = df1.append(row)
    

        print("Output df shape",df1.shape)
        df1.to_csv(args.outpath,index=False)
