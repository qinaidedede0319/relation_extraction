# coding=utf-8
from typing import DefaultDict, Sequence
from torch._C import device, set_flush_denormal
from EE_data_process import EEProcessor
import pytorch_lightning as pl
# from sklearn import model_selection
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchcrf import CRF
import csv
from conlleval import evaluate
from transformers import (
    BertTokenizerFast,
    BertForTokenClassification,
    BertModel
)
from transformers import AutoTokenizer, AutoModelForMaskedLM,BertModel
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from torch.nn import CrossEntropyLoss

import re
import json


class EEModel(pl.LightningModule):
    def __init__(self, config):
        # 1. Init parameters
        super(EEModel, self).__init__()
        
        self.config=config

        self.tokenizer = BertTokenizerFast.from_pretrained(config.pretrained_path)  # 
        # self.tokenizer = AutoTokenizer.from_pretrained(config.pretrained_path)
        special_tokens_dict = {'additional_special_tokens': ["<COM>", "“", "”"] }  # 在词典中增加特殊字符
        self.tokenizer.add_special_tokens(special_tokens_dict)
        self.processor = EEProcessor(config, self.tokenizer)

        self.labels = len(self.processor.rellabel2ids)
        self.model = BertModel.from_pretrained(
            config.pretrained_path, num_labels=self.labels)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.layer_norm = torch.nn.LayerNorm(config.hidden_size*2)
        self.classifier = nn.Linear(config.hidden_size * 2, self.labels)
        self.loss_fct = CrossEntropyLoss()
        self.batch_size = config.batch_size
        self.optimizer = config.optimizer
        self.lr = config.lr

    def prepare_data(self):
        train_data = self.processor.get_train_data()
        dev_data = self.processor.get_dev_data()
        if self.config.train_num>0:
            train_data=train_data[:self.config.train_num]
        if self.config.dev_num>0:
            dev_data=dev_data[:self.config.dev_num]

        print("train_length:", len(train_data))
        print("valid_length:", len(dev_data))

        self.train_loader = self.processor.create_dataloader(
            train_data, batch_size=self.batch_size, shuffle=True)
        self.valid_loader = self.processor.create_dataloader(
            dev_data, batch_size=self.batch_size, shuffle=False)

    def forward(self, input_ids, attention_mask=None):
        feats = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )[0]  # [bs,len,numlabels]

        return feats

    def training_step(self, batch, batch_idx):
        index_ids, input_ids,  attention_mask,offset_mapping,  subs, objs, relations, output_subs_ends, output_objs_ends = batch
        sequence_output = self.forward(input_ids=input_ids,attention_mask=attention_mask)
        # sequence_output: torch.Size([8, 512, 768])
        # relations: torch.Size([8,100])
        sub_output = torch.cat([a[i].unsqueeze(0) for a, i in zip(sequence_output, subs)])
        obj_output = torch.cat([a[i].unsqueeze(0) for a, i in zip(sequence_output, objs)])
        rep = torch.cat((sub_output, obj_output), dim=1)
        rep = self.layer_norm(rep)
        rep = self.dropout(rep)
        logits = self.classifier(rep)
        
        loss = self.loss_fct(logits.view(-1, self.labels), relations.view(-1))
        self.log('train_loss', loss.item())

        return loss

    def validation_step(self, batch, batch_idx):
        index_ids, input_ids,  attention_mask,offset_mapping, subs, objs, relations, output_subs_ends, output_objs_ends = batch
        sequence_output = self.forward(input_ids=input_ids,attention_mask=attention_mask)
        sub_output = torch.cat([a[i].unsqueeze(0) for a, i in zip(sequence_output, subs)])
        obj_output = torch.cat([a[i].unsqueeze(0) for a, i in zip(sequence_output, objs)])
        rep = torch.cat((sub_output, obj_output), dim=1)
        rep = self.layer_norm(rep)
        rep = self.dropout(rep)
        logits = self.classifier(rep)
                        
        loss = self.loss_fct(logits.view(-1, self.labels), relations.view(-1))
        pred = logits.argmax(dim=-1)

        gold = relations
        pre = torch.tensor(pred).cuda()

        return loss.cpu(),gold.cpu(),pre.cpu()

    def validation_epoch_end(self, outputs):
        val_loss,gold,pre = zip(*outputs)

        val_loss = torch.stack(val_loss).mean()
        gold = torch.cat(gold)
        pre = torch.cat(pre)

        # print("")

        true_seqs = [self.processor.relid2labels[int(g)] for g in gold]
        pred_seqs = [self.processor.relid2labels[int(g)] for g in pre]

        print("true_seqs",len(true_seqs),true_seqs[:5])
        print("pred_seqs",len(pred_seqs),pred_seqs[:5])

        print('\n')
        prec, rec, f1 = evaluate(true_seqs, pred_seqs)

        self.log('val_loss', val_loss)
        self.log('val_pre', torch.tensor(prec))
        self.log('val_rec', rec)
        self.log('val_f1', torch.tensor(f1))

    def configure_optimizers(self):
        # if self.use_crf:
        #     crf_params_ids = list(map(id, self.crf.parameters()))
        #     base_params = filter(lambda p: id(p) not in crf_params_ids, [
        #                          p for p in self.parameters() if p.requires_grad])

        #     arg_list = [{'params': base_params}, {'params': self.crf.parameters(), 'lr': self.crf_lr}]
        # else:
        #     # label_embed_and_attention_params = list(map(id, self.label_embedding.parameters())) + list(map(id, self.self_attention.parameters()))
        #     # arg_list = [{'params': list(self.label_embedding.parameters()) + list(self.self_attention.parameters()), 'lr': self.lr}]
        #     arg_list = [p for p in self.parameters() if p.requires_grad]
        arg_list = [p for p in self.parameters() if p.requires_grad]

        print("Num parameters:", len(arg_list))
        if self.optimizer == 'Adam':
            return torch.optim.Adam(arg_list, lr=self.lr, eps=1e-8)
        elif self.optimizer == 'SGD':
            return torch.optim.SGD(arg_list, lr=self.lr, momentum=0.9)

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.valid_loader

class EEPredictor:
    def __init__(self, checkpoint_path, config):
        self.model = EEModel.load_from_checkpoint(checkpoint_path, config=config)

        self.test_data = self.model.processor.get_test_data()
        self.tokenizer = self.model.tokenizer
        self.dataloader = self.model.processor.create_dataloader(
            self.test_data,batch_size=config.batch_size,shuffle=False)

        print("The TEST num is:", len(self.test_data))
        print('load checkpoint:', checkpoint_path)

    def generate_result(self,outfile_txt):
        device = torch.device("cpu")
        self.model.to(device)
        self.model.eval()

        print(len(self.test_data),len(self.dataloader))
        with open(outfile_txt, 'w') as fout:
            for batch in tqdm.tqdm(self.dataloader):
                for i in range(len(batch)):
                    batch[i] = batch[i].to(device)
                index_ids, input_ids,  attention_mask,offset_mapping,  subs, objs, relations, output_subs_ends, output_objs_ends = batch
                emissions = self.model(input_ids, attention_mask)
                # print(emissions.shape)

                sub_output = torch.cat([a[i].unsqueeze(0) for a, i in zip(emissions, subs)])
                obj_output = torch.cat([a[i].unsqueeze(0) for a, i in zip(emissions, objs)])
                rep = torch.cat((sub_output, obj_output), dim=1)
                rep = self.model.layer_norm(rep)
                rep = self.model.dropout(rep)
                logits = self.model.classifier(rep)

                preds = logits.argmax(dim=-1).cpu()
                # print(subs.shape)
                # print(preds.shape,preds)

                d = DefaultDict(list)
                for index_id,offset,pred,sub,obj,sub_end,obj_end in zip(index_ids,offset_mapping,preds,subs,objs,output_subs_ends,output_objs_ends):
                    arg1_start = int(offset[sub][0])
                    arg1_end = int(offset[sub_end][1])
                    arg2_start = int(offset[obj][0])
                    arg2_end = int(offset[obj_end][1])
                    relation = self.model.processor.relid2labels[int(pred)]
                    if int(pred)!=0:
                        d[index_id].append([arg1_start,arg1_end,arg2_start,arg2_end,relation])
                   
                for index_id,relation_list in d.items():
                    item=dict(self.test_data[index_id].items())
                    item["relation"] = relation_list
                    fout.write(json.dumps(item,ensure_ascii=False)+"\n")
            
        print('done--all tokens.')




                

