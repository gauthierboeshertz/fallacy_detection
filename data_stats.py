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
 
def data_stats(df):

    print(" df shape", df.shape)
    print("Nunber of fallacies", (df["is_fallacy"] == 1).sum())
    print("Nunber of valid", (df["is_fallacy"] == 0).sum())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", help="path for 1st input csv file")

    args = parser.parse_args()
    df = pd.read_csv(args.path)
    data_stats(df)