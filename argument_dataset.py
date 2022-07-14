import pickle
import cv2
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, AdamW, AutoModelForSequenceClassification
import pandas as pd
import random
from torch import nn
import sklearn
import numpy as np
import sklearn.metrics
import logging
import argparse
import torch.nn.functional as F
from sklearn.utils import compute_class_weight


class ArgumentDataset:

    def __init__(self, df_path,tokenizer, use_sa =True,use_hypothesis=True):

        self.use_sa = use_sa
        self.use_hypothesis = use_hypothesis

        self.df =  pd.read_csv(df_path)
        
        df_shape = self.df.shape
        self.df = self.df.dropna()
        print("Drop NA dropped", df_shape[0] - self.df.shape[0])
        df_shape = self.df.shape
        self.df = self.df.drop_duplicates()
        print("Drop duplicates dropped", df_shape[0] - self.df.shape[0])
        self.data_stats()
        self.tokenizer = tokenizer# AutoTokenizer.from_pretrained  (tokenizer_path, do_lower_case=True)
        special_tokens_dict = {
            'additional_special_tokens': ["[A]", "[B]", "[C]", "[D]", "[E]", "[F]", "[G]", "[H]", "[I]"]}
        
        #if self.use_sa :
        self.tokenizer.add_special_tokens(special_tokens_dict)

        self.data = self.load_data(self.df)

    def data_stats(self):
        print("The shape of the dataset is",self.df.shape)
        num_fallacies = (self.df["is_fallacy"] == 1).sum()
        num_non_fallacies = (self.df["is_fallacy"] == 0).sum()
        print("There are ",num_fallacies, "fallacies")
        print("There are ",num_non_fallacies, "non fallacies")

        self.class_weights = compute_class_weight(class_weight = "balanced", classes =np.unique(self.df["is_fallacy"]),y=self.df["is_fallacy"])
        print("Class weights",self.class_weights)

    def load_data(self, df):
        if df is None:
            return None

        MAX_LEN = 512

        premise_ids = []
        token_ids = []
        mask_ids = []
        seg_ids = []
        y = []

        #['argument', 'sa_argument',"hypothesis", 'entry_label', 'is_fallacy']
        if self.use_sa :
            premise_list = df['sa_argument'].to_list()
        else:
            premise_list = df['argument'].to_list()
        
        if self.use_hypothesis:
            hypothesis_list = df['hypothesis'].to_list()
        else:
            hypothesis_list = df['argument'].to_list()# not qctually used just for no bug in zip

        label_list = df['is_fallacy'].to_list()            

        for (premise, hypothesis, label) in zip(premise_list, hypothesis_list, label_list):

            premise_id = self.tokenizer.encode(premise, add_special_tokens=False)
            if len(premise_id) > MAX_LEN:
                continue

            premise_ids.append(torch.tensor(premise_id))
            attention_mask_ids = torch.tensor([1] * (len(premise_id) ))
            if self.use_hypothesis: 
                hypothesis_id = self.tokenizer.encode(hypothesis, add_special_tokens=False)
                #pair_token_ids = [self.tokenizer.cls_token_id] + premise_id + [
                #    self.tokenizer.sep_token_id] + hypothesis_id + [self.tokenizer.sep_token_id]
                pair_token_ids = self.tokenizer.build_inputs_with_special_tokens(premise_id,hypothesis_id)
                # pair_token_ids = premise_id + hypothesis_id
                # print("max token id=", max(pair_token_ids))
                premise_len = len(premise_id)
                hypothesis_len = len(hypothesis_id)

                #segment_ids = torch.tensor(
                #    [0] * (premise_len + 2) + [1] * (hypothesis_len + 1))  # sentence 0 and sentence 1
                segment_ids = torch.tensor(self.tokenizer.create_token_type_ids_from_sequences( premise_id,hypothesis_id))
                attention_mask_ids = torch.tensor([1] * len(pair_token_ids) )  # mask padded values
                if len(pair_token_ids) > MAX_LEN or len(segment_ids) > MAX_LEN or len(attention_mask_ids) > MAX_LEN:
                    continue
        
                token_ids.append(torch.tensor(pair_token_ids))
                seg_ids.append(segment_ids)

            mask_ids.append(attention_mask_ids)
            y.append(label)

        if self.use_hypothesis:
            token_ids = pad_sequence(token_ids, batch_first=True)
            mask_ids = pad_sequence(mask_ids, batch_first=True)
            seg_ids = pad_sequence(seg_ids, batch_first=True)
            y = torch.tensor(y).long()
            #y = F.one_hot(y)
            dataset = TensorDataset(token_ids, mask_ids, seg_ids, y)
        else:
            premise_ids = pad_sequence(premise_ids, batch_first=True)
            mask_ids = pad_sequence(mask_ids, batch_first=True)
            y = torch.tensor(y).long()
            dataset = TensorDataset(premise_ids, mask_ids, y)

        return dataset

    def get_dataloader(self, batch_size=32, shuffle=True):
        return DataLoader(
            self.data,
            shuffle=shuffle,
            batch_size=batch_size
        )






