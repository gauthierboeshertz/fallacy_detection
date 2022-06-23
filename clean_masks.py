import argparse
import os

import string
import pandas as pd
import pickle
import re
import logging
import traceback



def clean_mask(df):
    
    df["masked_articles"] = df["masked_articles"].str.replace(" n't", "n't", regex=False)
    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", help="path for input csv file")

    args = parser.parse_args()
    csvdf = pd.read_csv(args.path)
    csvdf = clean_mask(csvdf)
    csvdf.to_csv(args.path,index=False)