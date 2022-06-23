import argparse
import os

import string
import pandas as pd
import pickle
import re
import logging
import traceback



def add_is_fallacy_column(df,is_fallacy_csv):
    data = []

    df["is_fallacy"] = is_fallacy_csv
    df.to_csv(args.path, index=False)  
 
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", help="path for input csv file")
    parser.add_argument("-a", "--is_fallacy", type=int,help="1 if csv contains fallacies, 0 otherwise")

    args = parser.parse_args()
    csvdf = pd.read_csv(args.path)
    add_is_fallacy_column(csvdf,args.is_fallacy)
