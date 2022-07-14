from statistics import mode
import pytorch_lightning as pl
from torch.nn import functional as F
from torch import optim
from transformers import AutoModelForSequenceClassification
import torch
import pandas as pd
import numpy as np
from prediction_stats import print_stats

trans_cache_dir =  "/cluster/scratch/gboeshertz/huggingface_cache"

class BaseModule(pl.LightningModule):

    def __init__(self, config,class_weights=None):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.model = AutoModelForSequenceClassification.from_pretrained(config['model_name'], num_labels=2,
                                                                        ignore_mismatched_sizes=True,cache_dir =trans_cache_dir)
        
        self.test_list_logits = []
        self.test_list_labels = []  
        if class_weights is not None:
            self.class_weights = torch.from_numpy(class_weights).float()

    def load_ckpt(self,path):
        model_dict = torch.load(path,map_location= "cuda" if torch.cuda.is_available() else "cpu" )['state_dict']
        model_dict = {k.replace('model.',''):v for k,v in model_dict.items() if 'model' in k}
        #if not torch.cuda.is_available():
        #    for k, v in model_dict.items():
        #        model_dict[k] = v.cpu()

        self.model.load_state_dict(model_dict)
        
    def forward(self, x):
        # x should be a dictionnary with at least a key input_ids
        return self.model(x).logits

    def get_loss(self,preds,y,weights=None):
        return F.cross_entropy(preds.logits,y,weight=weights)



    def predict_sentence(self, tokens,masks=None,segs = None):

        with torch.no_grad():
            tokens = tokens.to(self.model.device)
            if segs is not None:
                segs = segs.to(self.model.device)

            if masks is not None:
                masks = masks.to(self.model.device)

            output = self.model(tokens, token_type_ids=segs, attention_mask=masks)
            
            pred = output.logits.argmax(axis=1).item()
        return pred

    def batch_step(self,batch):
        if self.config["use_hypothesis"]:
            pair_token_ids, mask_ids, seg_ids, y = batch
            
            pair_token_ids = pair_token_ids.to(self.model.device)
            mask_ids = mask_ids.to(self.model.device)
            seg_ids = seg_ids.to(self.model.device)
            labels = y.to(self.model.device)
            output = self.model(pair_token_ids,
                                    token_type_ids=seg_ids,
                                    attention_mask=mask_ids,
                                    labels=labels)

        else:
            premise_ids, mask_ids, y = batch
            premise_ids = premise_ids.to(self.model.device)
            mask_ids = mask_ids.to(self.model.device)
            labels = y.to(self.model.device)

            output = self.model(premise_ids,
                            attention_mask=mask_ids,
                            labels=labels)

        return output

    def training_step(self, batch, batch_idx):

        if self.class_weights is not None:
            self.class_weights = self.class_weights.to(self.model.device)

        output = self.batch_step(batch)
        #accuracy = (output.logits.argmax(axis=0) == y).mean()
        y = batch[-1].to(self.model.device)
        loss = self.get_loss(output,y,self.class_weights)
        accuracy = (output.logits.argmax(axis=1) == y).float().mean().item()
        self.log("train_acc", accuracy,on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_loss", loss,on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        if self.class_weights is not None:
            self.class_weights = self.class_weights.to(self.model.device)

        output = self.batch_step(batch)
        y = batch[-1].to(self.model.device)
        loss = self.get_loss(output,y,self.class_weights)
        accuracy = (output.logits.argmax(axis=1) == y).float().mean().item()

        self.log("val_acc", accuracy,on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_loss", loss,on_step=True, on_epoch=True, prog_bar=True, logger=True)


    def test_step(self, batch, batch_idx):
        logits = self.batch_step(batch).logits
        self.test_list_logits.append(logits)
        self.test_list_labels.append(batch[-1].to(self.model.device))

    def test_epoch_end(self, outputs):
        test_outputs = torch.vstack(self.test_list_logits).cpu().numpy()
        test_labels = torch.vstack(self.test_list_labels).cpu().numpy()[:,0]
        
        test_outputs = test_outputs.argmax(axis=1)

        preds_labels = np.vstack((test_outputs,test_labels))
        with open('test_outputs/'+ self.config["test_out_path"] + '.npy', 'wb') as f:
            np.save(f, preds_labels)

        print_stats(preds_labels,self.config["test_out_path"])

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.config['lr'])
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1, 2], gamma=0.1)
        return [optimizer], [lr_scheduler]
