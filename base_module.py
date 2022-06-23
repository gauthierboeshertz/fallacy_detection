from statistics import mode
import pytorch_lightning as pl
from torch.nn import functional as F
from torch import optim
from transformers import AutoModelForSequenceClassification
import torch
import pandas as pd
import numpy as np

class BaseModule(pl.LightningModule):

    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.model = AutoModelForSequenceClassification.from_pretrained(config['model_name'], num_labels=2,
                                                                        ignore_mismatched_sizes=True)
        self.test_list = []

    def load_ckpt(self,path):
        model_dict = torch.load(path)['state_dict']
        model_dict = {k.replace('model.',''):v for k,v in model_dict.items() if 'model' in k}
        self.model.load_state_dict(model_dict)
        
    def forward(self, x):
        # x should be a dictionnary with at least a key input_ids
        return self.model(x).logits

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
        loss = self.batch_step(batch).loss
        #accuracy = (output.logits.argmax(axis=0) == y).mean()
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        #self.log("train_accuracy", accuracy, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        output = self.batch_step(batch)
        loss = output.loss
        #accuracy = (output.logits.argmax(axis=0) == y).mean()

        self.log("val_loss", loss,logger=True)


    def test_step(self, batch, batch_idx):
        logits = self.batch_step(batch).logits
        self.test_list.append(logits)

    def test_epoch_end(self, outputs):
        print(self.test_list)
        test_outputs = torch.vstack(self.test_list).cpu().numpy()
        test_outputs = test_outputs.argmax(axis=1)
        test_outputs[test_outputs == 0] = -1
        ids = np.arange(1,test_outputs.shape[0]+1)
        outdf = pd.DataFrame({"Id":ids,'Prediction':test_outputs})
        outdf.to_csv('output.csv',index=False)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.config['lr'])
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1, 2], gamma=0.1)
        return [optimizer], [lr_scheduler]