import argparse
import os

import string
import pandas as pd
import pickle
import re
import logging
import traceback
import numpy as np
import warnings
from sklearn.utils import shuffle
import random
random.seed(0)
np.random.seed(0)
 
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", help="path for 1st input csv file")
    parser.add_argument("-r", "--num_rows", type=int, help="number of rows to keep")

    parser.add_argument("-po", "--outpath", help="path for output csv file")

    args = parser.parse_args()
    df = pd.read_csv(args.path)
    print("df shape", df.shape)
    df = shuffle(df)
    df = df.head(args.num_rows)
    print("new df shape", df.shape)
    df.to_csv(args.outpath,index=False)
