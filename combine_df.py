import argparse
import os

import string
import pandas as pd
import pickle
import re
import logging
import traceback

 
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-p1", "--path1", help="path for 1st input csv file")
    parser.add_argument("-p2", "--path2", help="path for 2ns input csv file")
    parser.add_argument("-po", "--outpath", help="path for output csv file")

    args = parser.parse_args()
    df1 = pd.read_csv(args.path1)
    df2 = pd.read_csv(args.path2)
    print(df1.shape)
    print(df2.shape)
    dfo = pd.concat([df1, df2], ignore_index=True)
    print(dfo.shape)
    dfo.to_csv(args.outpath,index=False)
