import argparse
import os

import string
import pandas as pd
import pickle
import re
import logging
import traceback



def create_struct_aware_csv(df):
    """
    strat=1 -> only original article
    strat=2 -> only masked article
    strat=3 -> both
    """
    data = []
    hypothesis = "This argument is fallacious."
    for i, row in df.iterrows():
        struct_text = row["masked_articles"]
        normal_text = row[args.text_path]
        entry = [normal_text, struct_text, hypothesis, row["is_fallacy"]]
        data.append(entry)
        
    return pd.DataFrame(data, columns=['argument', 'sa_argument',"hypothesis", 'is_fallacy'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", help="path for input csv file")
    parser.add_argument("-np", "--new_path", help="path for output csv file")

    parser.add_argument("-c", "--text_path", default='Argument',help="column which contains the main text")

    args = parser.parse_args()
    csvdf = pd.read_csv(args.path)
    new_df = create_struct_aware_csv(csvdf)
    new_df.to_csv(args.new_path)
