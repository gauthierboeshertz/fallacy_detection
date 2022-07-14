import argparse
import pytorch_lightning as pl
import yaml
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import torch
from argument_dataset import ArgumentDataset
from base_module import BaseModule
from remove_content_words import mask_out_content
from sentence_transformers import SentenceTransformer
import pandas as pd
from stanza.server import CoreNLPClient
import numpy as np
import random
import os

random.seed(0)
torch.manual_seed(0)
np.random.seed(0)
os.environ["TRANSFORMERS_CACHE"] = "/cluster/scratch/gboeshertz/huggingface_cache"

MAX_LEN = 512

def demo(config,sentence):

    tokenizer = AutoTokenizer.from_pretrained(config["tokenizer_name"], do_lower_case=True)
    
    special_tokens_dict = {
        'additional_special_tokens': ["[A]", "[B]", "[C]", "[D]", "[E]", "[F]", "[G]", "[H]", "[I]"]}
    tokenizer.add_special_tokens(special_tokens_dict)

    model = BaseModule(config=config,class_weights = None)

    model.load_ckpt(config["test_save_path"])

    if config["use_sa"]:

        model_mask = SentenceTransformer("all-mpnet-base-v2")
        client = CoreNLPClient(
            annotators=['tokenize', 'ssplit', 'pos', 'lemma', 'ner', 'parse', 'depparse', 'coref'], timeout=300000)
        sentence_copy = mask_out_content(sentence, model_mask, client, debug=True)
        sentence_copy = sentence_copy.replace(" n't", "n't")
        
    else:
        sentence_copy = sentence

    sentence_id = tokenizer.encode(sentence_copy, add_special_tokens=False)
    if len(sentence_id) > MAX_LEN:
        print("Sentence is too big")

    
    attention_mask_ids = torch.tensor([1] * (len(sentence_id) ))

    if config["use_hypothesis"]: 
        hypothesis = "This argument is fallacious."
        hypothesis_id = tokenizer.encode(hypothesis, add_special_tokens=False)
        #pair_token_ids = [tokenizer.cls_token_id] + sentence_id + [
        #    tokenizer.sep_token_id] + hypothesis_id + [tokenizer.sep_token_id]
        pair_token_ids = tokenizer.build_inputs_with_special_tokens(sentence_id,hypothesis_id)
        # pair_token_ids = premise_id + hypothesis_id
        # print("max token id=", max(pair_token_ids))
        premise_len = len(sentence_id)
        hypothesis_len = len(hypothesis_id)

        #segment_ids = torch.tensor(
        #    [0] * (premise_len + 2) + [1] * (hypothesis_len + 1))  # sentence 0 and sentence 1
        segment_ids = torch.tensor(tokenizer.create_token_type_ids_from_sequences( sentence_id,hypothesis_id))
        attention_mask_ids = torch.tensor([1] * len(pair_token_ids) )  # mask padded values

        if len(pair_token_ids) > MAX_LEN or len(segment_ids) > MAX_LEN or len(attention_mask_ids) > MAX_LEN:
            print("Sentence is too big")
        
        pair_token_ids = torch.tensor(pair_token_ids)
        pair_token_ids = pair_token_ids.unsqueeze(0)
        segment_ids = segment_ids.unsqueeze(0)
        attention_mask_ids = attention_mask_ids.unsqueeze(0)
        prediction = model.predict_sentence(pair_token_ids, segs=segment_ids,masks=attention_mask_ids)

    else:
        sentence_id = torch.tensor(sentence_id)
        attention_mask_ids = attention_mask_ids.unsqueeze(0)
        sentence_id = sentence_id.unsqueeze(0)
        prediction = model.predict_sentence(sentence_id, segs=None,masks=attention_mask_ids)

    labels = ["Non fallacious", "Fallacious"]

    print("The argument",sentence, " is ",labels[prediction])
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--config_path', type=str, default='')
    parser.add_argument('--sentence', type=str, default='')

    args = parser.parse_args()
    with open(args.config_path, "r") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    print('Config:')
    print(config)
    demo(config,args.sentence)
