from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import copy
import json
import math
import logging
import numpy as np

import torch
from torch import nn
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from torch.nn import LayerNorm
import torch.nn.functional as F
from config import BertConfig
from bert_models import BERT, PreTrainedBertModel, BertLMPredictionHead, TransformerBlock, gelu
import dill

logger = logging.getLogger(__name__)

CONFIG_NAME = 'bert_config.json'
WEIGHTS_NAME = 'pytorch_model.bin'


# 冻结反向传播
def freeze_afterwards(model):
    for p in model.parameters():
        p.requires_grad = False


class TSNE(PreTrainedBertModel):
    def __init__(self, config: BertConfig, dx_voc=None, rx_voc=None):
        super(TSNE, self).__init__(config)

        self.bert = BERT(config, dx_voc, rx_voc)
        self.dx_voc = dx_voc
        self.rx_voc = rx_voc

        freeze_afterwards(self)

    def forward(self, output_dir, output_file='graph_embedding.tsv'):
        # dx_graph_emb = self.bert.embedding.ontology_embedding.dx_embedding.embedding
        # rx_graph_emb = self.bert.embedding.ontology_embedding.rx_embedding.embedding

        if not self.config.graph:
            print('save embedding not graph')
            rx_graph_emb = self.bert.embedding.word_embeddings(
                torch.arange(3, len(self.rx_voc.word2idx) + 3, dtype=torch.long))
            dx_graph_emb = self.bert.embedding.word_embeddings(
                torch.arange(len(self.rx_voc.word2idx) + 3, len(self.rx_voc.word2idx) + 3 + len(self.dx_voc.word2idx),
                             dtype=torch.long))
        else:
            print('save embedding graph')

            dx_graph_emb = self.bert.embedding.ontology_embedding.dx_embedding.get_all_graph_emb()
            rx_graph_emb = self.bert.embedding.ontology_embedding.rx_embedding.get_all_graph_emb()

        np.savetxt(os.path.join(output_dir, 'dx-' + output_file),
                   dx_graph_emb.detach().numpy(), delimiter='\t')
        np.savetxt(os.path.join(output_dir, 'rx-' + output_file),
                   rx_graph_emb.detach().numpy(), delimiter='\t')

        # def dump(prefix='dx-', emb):
        #     with open(prefix + output_file ,'w') as fout:
        #         m = emb.detach().cpu().numpy()
        #         for
        #         fout.write()


class ClsHead(nn.Module):
    def __init__(self, config: BertConfig, voc_size):
        super(ClsHead, self).__init__()
        # 全连接输出层，输出为voc_size
        self.cls = nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size), nn.ReLU(
        ), nn.Linear(config.hidden_size, voc_size))

    def forward(self, input):
        return self.cls(input)

# 输出层，输出loss值
class SelfSupervisedHead(nn.Module):
    def __init__(self, config: BertConfig, age_voc_size,body_tem_voc_size, pulse_rate_voc_size, breathing_rate_voc_size, lsbp_voc_size,ldbp_voc_size,rsbp_voc_size,rdbp_voc_size,
                 height_voc_size, weight_voc_size,waist_voc_size,bmi_voc_size,exercise_freq_voc_size,smoking_status_voc_size,drinking_freq_voc_size,
                 heart_rate_voc_size,tcho_voc_size,tg_voc_size,ldlc_voc_size,hdlc_voc_size,is_hyper_voc_size):
        super(SelfSupervisedHead, self).__init__()
        self.multi_cls = nn.ModuleList(
            [ClsHead(config, age_voc_size),ClsHead(config, body_tem_voc_size),ClsHead(config, pulse_rate_voc_size),ClsHead(config, breathing_rate_voc_size), ClsHead(config, lsbp_voc_size),
            ClsHead(config, ldbp_voc_size),ClsHead(config, rsbp_voc_size), ClsHead(config, rdbp_voc_size),
             ClsHead(config, height_voc_size), ClsHead(config, weight_voc_size), ClsHead(config, waist_voc_size),
             ClsHead(config, bmi_voc_size), ClsHead(config, exercise_freq_voc_size),ClsHead(config, smoking_status_voc_size),
             ClsHead(config, drinking_freq_voc_size), ClsHead(config, heart_rate_voc_size),ClsHead(config, tcho_voc_size),ClsHead(config, tg_voc_size),ClsHead(config, ldlc_voc_size),ClsHead(config, hdlc_voc_size),ClsHead(config, is_hyper_voc_size)])

             # ClsHead(config, rdbp_voc_size), ClsHead(config, rdbp_voc_size), ClsHead(config, rdbp_voc_size),
             # ClsHead(config, height_voc_size), ClsHead(config, height_voc_size),ClsHead(config, height_voc_size),
             # ClsHead(config, weight_voc_size), ClsHead(config, weight_voc_size),ClsHead(config, weight_voc_size),
             # ClsHead(config, waist_voc_size), ClsHead(config, waist_voc_size), ClsHead(config, waist_voc_size),
             # ClsHead(config, bmi_voc_size), ClsHead(config, bmi_voc_size),ClsHead(config, bmi_voc_size),
             # ClsHead(config, exercise_freq_voc_size), ClsHead(config, exercise_freq_voc_size),ClsHead(config, exercise_freq_voc_size),
             # ClsHead(config, smoking_status_voc_size), ClsHead(config, smoking_status_voc_size), ClsHead(config, smoking_status_voc_size),
             # ClsHead(config, drinking_freq_voc_size), ClsHead(config, drinking_freq_voc_size),ClsHead(config, drinking_freq_voc_size),
             # ClsHead(config, heart_rate_voc_size), ClsHead(config, heart_rate_voc_size),ClsHead(config, heart_rate_voc_size),
             # ClsHead(config, is_hyper_voc_size), ClsHead(config, is_hyper_voc_size),ClsHead(config, is_hyper_voc_size)])

    def forward(self, age_inputs,body_tem_inputs, pulse_rate_inputs, breathing_rate_inputs, lsbp_inputs,ldbp_inputs,rsbp_inputs,rdbp_inputs,
                 height_inputs, weight_inputs,waist_inputs,bmi_inputs,exercise_freq_inputs,smoking_status_inputs,drinking_freq_inputs,heart_rate_inputs,tcho_inputs,tg_inputs,ldlc_inputs,hdlc_inputs,is_hyper_inputs):
        # inputs (B, hidden)
        # output logits
        return self.multi_cls[0](age_inputs),self.multi_cls[1](body_tem_inputs), self.multi_cls[2](pulse_rate_inputs), self.multi_cls[3](breathing_rate_inputs),\
               self.multi_cls[4](lsbp_inputs), self.multi_cls[5](ldbp_inputs), self.multi_cls[6](rsbp_inputs), \
               self.multi_cls[7](rdbp_inputs), self.multi_cls[8](height_inputs), self.multi_cls[9](weight_inputs),\
               self.multi_cls[10](waist_inputs), self.multi_cls[11](bmi_inputs), self.multi_cls[12](exercise_freq_inputs),\
               self.multi_cls[13](smoking_status_inputs), self.multi_cls[14](drinking_freq_inputs), self.multi_cls[15](heart_rate_inputs),self.multi_cls[16](tcho_inputs),self.multi_cls[17](tg_inputs),self.multi_cls[18](ldlc_inputs),self.multi_cls[19](hdlc_inputs),\
               self.multi_cls[20](is_hyper_inputs), \
               self.multi_cls[20](age_inputs),self.multi_cls[20](body_tem_inputs), self.multi_cls[20](pulse_rate_inputs), self.multi_cls[20](breathing_rate_inputs),\
               self.multi_cls[20](lsbp_inputs), self.multi_cls[20](ldbp_inputs), self.multi_cls[20](rsbp_inputs), \
               self.multi_cls[20](rdbp_inputs), self.multi_cls[20](height_inputs), self.multi_cls[20](weight_inputs),\
               self.multi_cls[20](waist_inputs), self.multi_cls[20](bmi_inputs), self.multi_cls[20](exercise_freq_inputs),\
               self.multi_cls[20](smoking_status_inputs), self.multi_cls[20](drinking_freq_inputs), self.multi_cls[20](heart_rate_inputs),self.multi_cls[20](tcho_inputs),self.multi_cls[20](tg_inputs),self.multi_cls[20](ldlc_inputs),self.multi_cls[20](hdlc_inputs)


class GBERT_Pretrain_train(PreTrainedBertModel):
    def __init__(self, config: BertConfig,tokenizer, age_voc=None,body_tem_voc=None, pulse_rate_voc=None, breathing_rate_voc=None, lsbp_voc=None,ldbp_voc=None,rsbp_voc=None,rdbp_voc=None,
                 height_voc=None, weight_voc=None,waist_voc=None,bmi_voc=None,exercise_freq_voc=None,smoking_status_voc=None,drinking_freq_voc=None,
                 heart_rate_voc=None,tcho_voc=None,tg_voc=None,ldlc_voc=None,hdlc_voc=None,is_hyper_voc=None):
        super(GBERT_Pretrain_train, self).__init__(config)
        self.age_voc_size = len(age_voc.word2idx)
        self.body_tem_voc_size = len(body_tem_voc.word2idx)
        self.pulse_rate_voc_size = len(pulse_rate_voc.word2idx)
        self.breathing_rate_voc_size = len(breathing_rate_voc.word2idx)
        self.lsbp_voc_size = len(lsbp_voc.word2idx)
        self.ldbp_voc_size = len(ldbp_voc.word2idx)
        self.rsbp_voc_size = len(rsbp_voc.word2idx)
        self.rdbp_voc_size = len(rdbp_voc.word2idx)
        self.height_voc_size = len(height_voc.word2idx)
        self.weight_voc_size = len(weight_voc.word2idx)
        self.waist_voc_size = len(waist_voc.word2idx)
        self.bmi_voc_size = len(bmi_voc.word2idx)
        self.exercise_freq_voc_size = len(exercise_freq_voc.word2idx)
        self.smoking_status_voc_size = len(smoking_status_voc.word2idx)
        self.drinking_freq_voc_size = len(drinking_freq_voc.word2idx)
        self.heart_rate_voc_size = len(heart_rate_voc.word2idx)
        self.tcho_voc_size = len(tcho_voc.word2idx)
        self.tg_voc_size = len(tg_voc.word2idx)
        self.ldlc_voc_size = len(ldlc_voc.word2idx)
        self.hdlc_voc_size = len(hdlc_voc.word2idx)
        self.is_hyper_voc_size = len(is_hyper_voc.word2idx)

        # self.body_tem_voc_size= len(body_tem_voc.word2idx)
        # self.pulse_rate_voc_size=len(pulse_rate_voc.word2idx)
        # self.breathing_rate_voc_size= len(breathing_rate_voc.word2idx)
        # self.lsbp_voc_size= len(lsbp_voc.word2idx)
        # self.ldbp_voc_size= len(ldbp_voc.word2idx)
        # self.rsbp_voc_size= len(rsbp_voc.word2idx)
        # self.rdbp_voc_size= len(rdbp_voc.word2idx)
        # self.height_voc_size= len(height_voc.word2idx)
        # self.weight_voc_size= len(weight_voc.word2idx)
        # self.waist_voc_size= len(waist_voc.word2idx)
        # self.bmi_voc_size= len(bmi_voc.word2idx)
        # self.exercise_freq_voc_size= len(exercise_freq_voc.word2idx)
        # self.smoking_status_voc_size= len(smoking_status_voc.word2idx)
        # self.drinking_freq_voc_size= len(drinking_freq_voc.word2idx)
        # self.heart_rate_voc_size= len(heart_rate_voc.word2idx)
        # self.is_hyper_voc_size= len(is_hyper_voc.word2idx)


        self.bert = BERT(config,body_tem_voc, pulse_rate_voc, breathing_rate_voc, lsbp_voc,ldbp_voc)
        self.cls_pretrain = SelfSupervisedHead(
            config,self.age_voc_size,self.body_tem_voc_size, self.pulse_rate_voc_size, self.breathing_rate_voc_size, self.lsbp_voc_size,self.ldbp_voc_size,self.rsbp_voc_size,self.rdbp_voc_size,
                 self.height_voc_size, self.weight_voc_size,self.waist_voc_size,self.bmi_voc_size,self.exercise_freq_voc_size,self.smoking_status_voc_size,self.drinking_freq_voc_size,
                 self.heart_rate_voc_size,self.tcho_voc_size,self.tg_voc_size,self.ldlc_voc_size,self.hdlc_voc_size,self.is_hyper_voc_size)
        self.dense = nn.ModuleList(
            [MappingHead(config),MappingHead(config),MappingHead(config),MappingHead(config),MappingHead(config),MappingHead(config), MappingHead(config), MappingHead(config), MappingHead(config), MappingHead(config),
             MappingHead(config), MappingHead(config), MappingHead(config), MappingHead(config), MappingHead(config),
             MappingHead(config), MappingHead(config), MappingHead(config), MappingHead(config), MappingHead(config),
             MappingHead(config)])
        self.cls_predict = nn.Sequential(nn.Linear(40 * config.hidden_size, 2 * config.hidden_size), nn.ReLU(),
                                 nn.Linear(2 * config.hidden_size, len(tokenizer.is_hyper_voc_multi.word2idx)))

        self.apply(self.init_bert_weights)

    def forward(self, inputs, age_labels=None,body_tem_labels=None, pulse_rate_labels=None,breathing_rate_labels=None,
                                        lsbp_labels=None, ldbp_labels=None,rsbp_labels=None,rdbp_labels=None,
                                        height_labels=None, weight_labels=None,waist_labels=None,bmi_labels=None,
                                        exercise_freq_labels=None, smoking_status_labels=None,drinking_freq_labels=None,
                                        heart_rate_labels=None,tcho_labels=None,tg_labels=None,ldlc_labels=None,hdlc_labels=None,is_hyper_labels=None,epoch=None,flag=1):
        # print(drinking_freq_labels)
        # inputs (B, 2, max_len)
        # bert_pool (B, hidden)
        # inputs -> torch.Size([4, 2, 55]) dx_labels -> torch.Size([4, 1997])
        # rx_labels -> torch.Size([4, 468]) dx_bert_pool ->torch.Size([4, 300]) rx_bert_pool ->rx_bert_pool
        #############预训练#################
        if flag==0:
            _, age_bert_pool = self.bert(inputs[:, 0, :], torch.zeros(
                (inputs.size(0), inputs.size(2))).long().to(inputs.device))
            _, body_tem_bert_pool = self.bert(inputs[:, 1, :], torch.zeros(
                (inputs.size(0), inputs.size(2))).long().to(inputs.device))
            _, pulse_rate_bert_pool = self.bert(inputs[:, 2, :], torch.zeros(
                (inputs.size(0), inputs.size(2))).long().to(inputs.device))
            _, breathing_rate_bert_pool = self.bert(inputs[:, 3, :], torch.zeros(
                (inputs.size(0), inputs.size(2))).long().to(inputs.device))
            _, lsbp_bert_pool = self.bert(inputs[:, 4, :], torch.zeros(
                (inputs.size(0), inputs.size(2))).long().to(inputs.device))
            _, ldbp_bert_pool = self.bert(inputs[:, 5, :], torch.zeros(
                (inputs.size(0), inputs.size(2))).long().to(inputs.device))
            _, rsbp_bert_pool = self.bert(inputs[:, 6, :], torch.zeros(
                (inputs.size(0), inputs.size(2))).long().to(inputs.device))
            _, rdbp_bert_pool = self.bert(inputs[:, 7, :], torch.zeros(
                (inputs.size(0), inputs.size(2))).long().to(inputs.device))
            _, height_bert_pool = self.bert(inputs[:, 8, :], torch.zeros(
                (inputs.size(0), inputs.size(2))).long().to(inputs.device))
            _, weight_bert_pool = self.bert(inputs[:, 9, :], torch.zeros(
                (inputs.size(0), inputs.size(2))).long().to(inputs.device))
            _, waist_bert_pool = self.bert(inputs[:, 10, :], torch.zeros(
                (inputs.size(0), inputs.size(2))).long().to(inputs.device))
            _, bmi_bert_pool = self.bert(inputs[:, 11, :], torch.zeros(
                (inputs.size(0), inputs.size(2))).long().to(inputs.device))
            _, exercise_freq_bert_pool = self.bert(inputs[:, 12, :], torch.zeros(
                (inputs.size(0), inputs.size(2))).long().to(inputs.device))
            _, smoking_status_bert_pool = self.bert(inputs[:, 13, :], torch.zeros(
                (inputs.size(0), inputs.size(2))).long().to(inputs.device))
            _, drinking_freq_bert_pool = self.bert(inputs[:, 14, :], torch.zeros(
                (inputs.size(0), inputs.size(2))).long().to(inputs.device))
            _, heart_rate_bert_pool = self.bert(inputs[:, 15, :], torch.zeros(
                (inputs.size(0), inputs.size(2))).long().to(inputs.device))
            _, tcho_bert_pool = self.bert(inputs[:, 16, :], torch.zeros(
                (inputs.size(0), inputs.size(2))).long().to(inputs.device))
            _, tg_bert_pool = self.bert(inputs[:, 17, :], torch.zeros(
                (inputs.size(0), inputs.size(2))).long().to(inputs.device))
            _, ldlc_bert_pool = self.bert(inputs[:, 18, :], torch.zeros(
                (inputs.size(0), inputs.size(2))).long().to(inputs.device))
            _, hdlc_bert_pool = self.bert(inputs[:, 19, :], torch.zeros(
                (inputs.size(0), inputs.size(2))).long().to(inputs.device))
            _, is_hyper_bert_pool = self.bert(inputs[:, 20, :], torch.zeros(
                (inputs.size(0), inputs.size(2))).long().to(inputs.device))

            age2age,body_tem2body_tem,pulse_rate2pulse_rate, breathing_rate2breathing_rate,lsbp2lsbp,ldbp2ldbp,rsbp2rsbp,rdbp2rdbp,height2height,weight2weight,\
            waist2waist,bmi2bmi,exercise_freq2exercise_freq,smoking_status2smoking_status,drinking_freq2drinking_freq,heart_rate2heart_rate,tcho2tcho,tg2tg,ldlc2ldlc,hdlc2hdlc,is_hyper2is_hyper,\
                    age2is_hyper,body_tem2is_hyper, pulse_rate2is_hyper, breathing_rate2is_hyper, lsbp2is_hyper, ldbp2is_hyper, rsbp2is_hyper, rdbp2is_hyper, height2is_hyper, weight2is_hyper, \
                    waist2is_hyper, bmi2is_hyper, exercise_freq2is_hyper, smoking_status2is_hyper, drinking_freq2is_hyper, heart_rate2is_hyper,tcho2is_hyper,tg2is_hyper,ldlc2is_hyper,hdlc2is_hyper\
                        = self.cls_pretrain(age_bert_pool,body_tem_bert_pool, pulse_rate_bert_pool, breathing_rate_bert_pool,lsbp_bert_pool,ldbp_bert_pool,
                                           rsbp_bert_pool,rdbp_bert_pool,height_bert_pool,weight_bert_pool,waist_bert_pool,bmi_bert_pool,
                                           exercise_freq_bert_pool,smoking_status_bert_pool,drinking_freq_bert_pool,heart_rate_bert_pool,tcho_bert_pool,tg_bert_pool,ldlc_bert_pool,hdlc_bert_pool,is_hyper_bert_pool)
            # output logits
            # if is_hyper_labels is None:
                # print("hhh")
            if age_labels is None or body_tem_labels is None or pulse_rate_labels is None or breathing_rate_labels is None or lsbp_labels is None or \
                    ldbp_labels is None or rsbp_labels is None or rdbp_labels is None or height_labels is None or\
                    weight_labels is None or waist_labels is None or bmi_labels is None or exercise_freq_labels is None or smoking_status_labels is None or\
                    drinking_freq_labels is None or heart_rate_labels is None or tcho_labels is None or tg_labels is None or ldlc_labels is None or hdlc_labels is None:
                # print("label=0")
                # print(age_labels)
                return F.sigmoid(age2age),F.sigmoid(body_tem2body_tem), F.sigmoid(pulse_rate2pulse_rate), F.sigmoid(breathing_rate2breathing_rate), \
                       F.sigmoid(lsbp2lsbp), F.sigmoid(ldbp2ldbp), F.sigmoid(rsbp2rsbp),F.sigmoid(rdbp2rdbp), F.sigmoid(height2height), \
                       F.sigmoid(weight2weight),F.sigmoid(waist2waist),F.sigmoid(bmi2bmi),F.sigmoid(exercise_freq2exercise_freq),F.sigmoid(smoking_status2smoking_status),\
                       F.sigmoid(drinking_freq2drinking_freq),F.sigmoid(heart_rate2heart_rate),F.sigmoid(tcho2tcho),F.sigmoid(tg2tg),F.sigmoid(ldlc2ldlc),F.sigmoid(hdlc2hdlc),F.sigmoid(is_hyper2is_hyper),F.sigmoid(age2is_hyper),F.sigmoid(body_tem2is_hyper),F.sigmoid(pulse_rate2is_hyper),\
                       F.sigmoid(breathing_rate2is_hyper),F.sigmoid(lsbp2is_hyper),F.sigmoid(ldbp2is_hyper),F.sigmoid(rsbp2is_hyper),F.sigmoid(rdbp2is_hyper),F.sigmoid(height2is_hyper),\
                       F.sigmoid(weight2is_hyper),F.sigmoid(waist2is_hyper),F.sigmoid(bmi2is_hyper),F.sigmoid(exercise_freq2is_hyper),F.sigmoid(smoking_status2is_hyper),F.sigmoid(drinking_freq2is_hyper),\
                       F.sigmoid(heart_rate2is_hyper),F.sigmoid(tcho2is_hyper),F.sigmoid(tg2is_hyper),F.sigmoid(ldlc2is_hyper),F.sigmoid(hdlc2is_hyper)
            else:
                loss = F.binary_cross_entropy_with_logits(age2age,age_labels) + \
                    F.binary_cross_entropy_with_logits(body_tem2body_tem, body_tem_labels) + \
                    F.binary_cross_entropy_with_logits(pulse_rate2pulse_rate, pulse_rate_labels) + \
                    F.binary_cross_entropy_with_logits(breathing_rate2breathing_rate, breathing_rate_labels) + \
                    F.binary_cross_entropy_with_logits(lsbp2lsbp, lsbp_labels) + \
                    F.binary_cross_entropy_with_logits(ldbp2ldbp, ldbp_labels) + \
                    F.binary_cross_entropy_with_logits(rsbp2rsbp, rsbp_labels) + \
                    F.binary_cross_entropy_with_logits(rdbp2rdbp,rdbp_labels) + \
                    F.binary_cross_entropy_with_logits(height2height, height_labels) + \
                    F.binary_cross_entropy_with_logits(weight2weight, weight_labels)+\
                    F.binary_cross_entropy_with_logits(waist2waist, waist_labels) + \
                    F.binary_cross_entropy_with_logits(bmi2bmi, bmi_labels) + \
                    F.binary_cross_entropy_with_logits(exercise_freq2exercise_freq, exercise_freq_labels) + \
                    F.binary_cross_entropy_with_logits(smoking_status2smoking_status, smoking_status_labels) + \
                    F.binary_cross_entropy_with_logits(drinking_freq2drinking_freq, drinking_freq_labels) + \
                    F.binary_cross_entropy_with_logits(heart_rate2heart_rate, heart_rate_labels) + \
                    F.binary_cross_entropy_with_logits(tcho2tcho, tcho_labels) + \
                    F.binary_cross_entropy_with_logits(tg2tg, tg_labels) + \
                    F.binary_cross_entropy_with_logits(ldlc2ldlc, ldlc_labels) + \
                    F.binary_cross_entropy_with_logits(hdlc2hdlc, hdlc_labels) + \
                    F.binary_cross_entropy_with_logits(is_hyper2is_hyper, is_hyper_labels) + \
                    F.binary_cross_entropy_with_logits(age2is_hyper, is_hyper_labels) + \
                    F.binary_cross_entropy_with_logits(body_tem2is_hyper, is_hyper_labels) + \
                    F.binary_cross_entropy_with_logits(pulse_rate2is_hyper, is_hyper_labels) + \
                    F.binary_cross_entropy_with_logits(breathing_rate2is_hyper, is_hyper_labels) + \
                    F.binary_cross_entropy_with_logits(lsbp2is_hyper, is_hyper_labels) + \
                    F.binary_cross_entropy_with_logits(ldbp2is_hyper, is_hyper_labels) + \
                    F.binary_cross_entropy_with_logits(rsbp2is_hyper, is_hyper_labels) + \
                    F.binary_cross_entropy_with_logits(rdbp2is_hyper, is_hyper_labels) + \
                    F.binary_cross_entropy_with_logits(height2is_hyper, is_hyper_labels) + \
                    F.binary_cross_entropy_with_logits(weight2is_hyper, is_hyper_labels) + \
                    F.binary_cross_entropy_with_logits(waist2is_hyper, is_hyper_labels) + \
                    F.binary_cross_entropy_with_logits(bmi2is_hyper, is_hyper_labels) + \
                    F.binary_cross_entropy_with_logits(exercise_freq2is_hyper, is_hyper_labels) + \
                    F.binary_cross_entropy_with_logits(smoking_status2is_hyper, is_hyper_labels) + \
                    F.binary_cross_entropy_with_logits(drinking_freq2is_hyper, is_hyper_labels)+\
                    F.binary_cross_entropy_with_logits(heart_rate2is_hyper, is_hyper_labels)+\
                    F.binary_cross_entropy_with_logits(tcho2is_hyper, is_hyper_labels)+\
                    F.binary_cross_entropy_with_logits(tg2is_hyper, is_hyper_labels)+\
                    F.binary_cross_entropy_with_logits(ldlc2is_hyper, is_hyper_labels)+\
                    F.binary_cross_entropy_with_logits(hdlc2is_hyper, is_hyper_labels)

                return loss, F.sigmoid(age2age),F.sigmoid(body_tem2body_tem), F.sigmoid(pulse_rate2pulse_rate), F.sigmoid(breathing_rate2breathing_rate), \
                       F.sigmoid(lsbp2lsbp), F.sigmoid(ldbp2ldbp), F.sigmoid(rsbp2rsbp),F.sigmoid(rdbp2rdbp), F.sigmoid(height2height), \
                       F.sigmoid(weight2weight),F.sigmoid(waist2waist),F.sigmoid(bmi2bmi),F.sigmoid(exercise_freq2exercise_freq),F.sigmoid(smoking_status2smoking_status),\
                       F.sigmoid(drinking_freq2drinking_freq),F.sigmoid(heart_rate2heart_rate),F.sigmoid(tcho2tcho),F.sigmoid(tg2tg),F.sigmoid(ldlc2ldlc),F.sigmoid(hdlc2hdlc),F.sigmoid(is_hyper2is_hyper),F.sigmoid(age2is_hyper),F.sigmoid(body_tem2is_hyper),F.sigmoid(pulse_rate2is_hyper),\
                       F.sigmoid(breathing_rate2is_hyper),F.sigmoid(lsbp2is_hyper),F.sigmoid(ldbp2is_hyper),F.sigmoid(rsbp2is_hyper),F.sigmoid(rdbp2is_hyper),F.sigmoid(height2is_hyper),\
                       F.sigmoid(weight2is_hyper),F.sigmoid(waist2is_hyper),F.sigmoid(bmi2is_hyper),F.sigmoid(exercise_freq2is_hyper),F.sigmoid(smoking_status2is_hyper),F.sigmoid(drinking_freq2is_hyper),\
                       F.sigmoid(heart_rate2is_hyper),F.sigmoid(tcho2is_hyper),F.sigmoid(tg2is_hyper),F.sigmoid(ldlc2is_hyper),F.sigmoid(hdlc2is_hyper)
        elif flag==1:
            token_types_ids = torch.cat([torch.zeros((1, inputs.size(1))), torch.zeros((1, inputs.size(1))),
                                         torch.zeros((1, inputs.size(1))), torch.zeros((1, inputs.size(1))),
                                         torch.zeros((1, inputs.size(1))), torch.zeros((1, inputs.size(1))),
                                         torch.zeros((1, inputs.size(1))), torch.zeros((1, inputs.size(1))),
                                         torch.zeros((1, inputs.size(1))), torch.zeros((1, inputs.size(1))),
                                         torch.zeros((1, inputs.size(1))), torch.zeros((1, inputs.size(1))),
                                         torch.zeros((1, inputs.size(1))), torch.zeros((1, inputs.size(1))),
                                         torch.zeros((1, inputs.size(1))), torch.zeros((1, inputs.size(1))),
                                         torch.zeros((1, inputs.size(1))), torch.zeros((1, inputs.size(1))),
                                         torch.zeros((1, inputs.size(1))), torch.zeros((1, inputs.size(1))), torch.ones(
                    (1, inputs.size(1)))], dim=0).long().to(inputs.device)
            token_types_ids = token_types_ids.repeat(
                1 if inputs.size(0) // 21 == 0 else inputs.size(0) // 21, 1)
            _, bert_pool = self.bert(inputs, token_types_ids)
            loss = 0
            bert_pool = bert_pool.view(21, -1, bert_pool.size(1))  # (3, adm, H)#bert_pool(16,1,300)
            # dx_bert_pool = self.dense[0](bert_pool[0])  # (adm, H)
            # rx_bert_pool = self.dense[1](bert_pool[1])  # (adm, H)
            # sx_bert_pool = self.dense[2](bert_pool[2])  # (adm, H)
            age_bert_pool = self.dense[0](bert_pool[0])
            body_tem_bert_pool = self.dense[1](bert_pool[1])#size(1,300)
            pulse_rate_bert_pool = self.dense[2](bert_pool[2])
            breathing_rate_bert_pool = self.dense[3](bert_pool[3])
            lsbp_bert_pool = self.dense[4](bert_pool[4])
            ldbp_bert_pool = self.dense[5](bert_pool[5])
            rsbp_bert_pool = self.dense[6](bert_pool[6])
            rdbp_bert_pool = self.dense[7](bert_pool[7])
            height_bert_pool = self.dense[8](bert_pool[8])
            weight_bert_pool = self.dense[9](bert_pool[9])
            waist_bert_pool = self.dense[10](bert_pool[10])
            bmi_bert_pool = self.dense[11](bert_pool[11])
            exercise_freq_bert_pool = self.dense[12](bert_pool[1])
            smoking_status_bert_pool = self.dense[13](bert_pool[13])
            drinking_freq_bert_pool = self.dense[14](bert_pool[14])
            heart_rate_bert_pool = self.dense[15](bert_pool[15])
            tcho_bert_pool = self.dense[16](bert_pool[16])
            tg_bert_pool = self.dense[17](bert_pool[17])
            ldlc_bert_pool = self.dense[18](bert_pool[18])
            hdlc_bert_pool = self.dense[19](bert_pool[19])
            # is_hyper_bert_pool = self.dense[15](bert_pool[15])

            # mean and concat for rx prediction task
            is_hyper_logits = []
            for i in range(is_hyper_labels.size(0)):
                # mean
                age_mean = torch.mean(age_bert_pool[0:i + 1, :], dim=0, keepdim=True)
                body_tem_mean = torch.mean(body_tem_bert_pool[0:i + 1, :], dim=0, keepdim=True)
                pulse_rate_mean = torch.mean(pulse_rate_bert_pool[0:i + 1, :], dim=0, keepdim=True)
                breathing_rate_mean = torch.mean(breathing_rate_bert_pool[0:i + 1, :], dim=0, keepdim=True)
                lsbp_mean = torch.mean(lsbp_bert_pool[0:i + 1, :], dim=0, keepdim=True)
                ldbp_mean = torch.mean(ldbp_bert_pool[0:i + 1, :], dim=0, keepdim=True)
                rsbp_mean = torch.mean(rsbp_bert_pool[0:i + 1, :], dim=0, keepdim=True)
                rdbp_mean = torch.mean(rdbp_bert_pool[0:i + 1, :], dim=0, keepdim=True)
                height_mean = torch.mean(height_bert_pool[0:i + 1, :], dim=0, keepdim=True)
                weight_mean = torch.mean(weight_bert_pool[0:i + 1, :], dim=0, keepdim=True)
                waist_mean = torch.mean(waist_bert_pool[0:i + 1, :], dim=0, keepdim=True)
                bmi_mean = torch.mean(bmi_bert_pool[0:i + 1, :], dim=0, keepdim=True)
                exercise_freq_mean = torch.mean(exercise_freq_bert_pool[0:i + 1, :], dim=0, keepdim=True)
                smoking_status_mean = torch.mean(smoking_status_bert_pool[0:i + 1, :], dim=0, keepdim=True)
                drinking_freq_mean = torch.mean(drinking_freq_bert_pool[0:i + 1, :], dim=0, keepdim=True)
                heart_rate_mean = torch.mean(heart_rate_bert_pool[0:i + 1, :], dim=0, keepdim=True)
                tcho_mean = torch.mean(tcho_bert_pool[0:i + 1, :], dim=0, keepdim=True)
                tg_mean = torch.mean(tg_bert_pool[0:i + 1, :], dim=0, keepdim=True)
                ldlc_mean = torch.mean(ldlc_bert_pool[0:i + 1, :], dim=0, keepdim=True)
                hdlc_mean = torch.mean(hdlc_bert_pool[0:i + 1, :], dim=0, keepdim=True)
                # is_hyper_mean = torch.mean(is_hyper_bert_pool[0:i + 1, :], dim=0, keepdim=True)
                # dx_mean = torch.mean(dx_bert_pool[0:i+1, :], dim=0, keepdim=True)
                # rx_mean = torch.mean(rx_bert_pool[0:i+1, :], dim=0, keepdim=True)
                # sx_mean = torch.mean(sx_bert_pool[0:i + 1, :], dim=0, keepdim=True)
                # concat
                concat = torch.cat(
                    [age_mean,body_tem_mean, pulse_rate_mean, breathing_rate_mean, \
                     lsbp_mean, ldbp_mean, rsbp_mean, rdbp_mean, height_mean, weight_mean, \
                     waist_mean, bmi_mean, exercise_freq_mean, smoking_status_mean, \
                     drinking_freq_mean, heart_rate_mean,tcho_mean,tg_mean,ldlc_mean,hdlc_mean, \
                     age_bert_pool[i + 1, :].unsqueeze(dim=0), \
                     body_tem_bert_pool[i + 1, :].unsqueeze(dim=0), \
                     pulse_rate_bert_pool[i+1, :].unsqueeze(dim=0),\
                     breathing_rate_bert_pool[i+1, :].unsqueeze(dim=0), \
                     lsbp_bert_pool[i+1, :].unsqueeze(dim=0), ldbp_bert_pool[i+1, :].unsqueeze(dim=0),\
                     rsbp_bert_pool[i+1, :].unsqueeze(dim=0), rdbp_bert_pool[i+1, :].unsqueeze(dim=0),\
                     height_bert_pool[i+1, :].unsqueeze(dim=0), weight_bert_pool[i+1, :].unsqueeze(dim=0), \
                     waist_bert_pool[i+1, :].unsqueeze(dim=0), bmi_bert_pool[i+1, :].unsqueeze(dim=0),\
                     exercise_freq_bert_pool[i+1, :].unsqueeze(dim=0),\
                     smoking_status_bert_pool[i+1, :].unsqueeze(dim=0), \
                     drinking_freq_bert_pool[i+1, :].unsqueeze(dim=0),\
                     heart_rate_bert_pool[i+1, :].unsqueeze(dim=0), \
                     tcho_bert_pool[i + 1, :].unsqueeze(dim=0), \
                     tg_bert_pool[i + 1, :].unsqueeze(dim=0), \
                     ldlc_bert_pool[i + 1, :].unsqueeze(dim=0), \
                     hdlc_bert_pool[i + 1, :].unsqueeze(dim=0)],dim=-1)
                is_hyper_logits.append(self.cls_predict(concat))

            is_hyper_logits = torch.cat(is_hyper_logits, dim=0)
            loss = F.binary_cross_entropy_with_logits(is_hyper_logits, is_hyper_labels)
            return loss, is_hyper_logits
class GBERT_Predict_med(PreTrainedBertModel):
    def __init__(self, config: BertConfig, tokenizer):
        super(GBERT_Predict_med, self).__init__(config)
        self.bert = BERT(config, tokenizer.dx_voc, tokenizer.rx_voc)
        self.dense = nn.ModuleList([MappingHead(config), MappingHead(config), MappingHead(config)])
        self.cls = nn.Sequential(nn.Linear(4*config.hidden_size, 2*config.hidden_size),
                                 nn.ReLU(), nn.Linear(2*config.hidden_size, len(tokenizer.rx_voc_multi.word2idx)))

        self.apply(self.init_bert_weights)
        # self.input_ids=input_ids

    def forward(self, input_ids, dx_labels=None, rx_labels=None,sx_labels=None, epoch=None):
        """
        :param input_ids: [B, max_seq_len] where B = 3*adm
        :param rx_labels: [adm-1, rx_size]
        :param dx_labels: [adm-1, dx_size]
        :return:
        """
        # token_types_ids = torch.ones(input_ids.size(0), input_ids.size(1)).long().to(input_ids.device)
        # # token_types_ids = torch.cat([torch.zeros((1, input_ids.size(1))), torch.ones(
        # #     (1, input_ids.size(1)))], dim=0).long().to(input_ids.device)
        # token_types_ids = token_types_ids.repeat(
        #     1 if input_ids.size(0)//3 == 0 else input_ids.size(0)//3, 1)
        # bert_pool: (2*adm, H)
        # _, bert_pool = self.bert(input_ids, token_types_ids)
        # loss = 0
        # bert_pool = bert_pool.view(3, -1, bert_pool.size(1))  # (2, adm, H)
        # dx_bert_pool = self.dense[0](bert_pool[0])  # (adm, H)
        # rx_bert_pool = self.dense[1](bert_pool[1])  # (adm, H)
        # sx_bert_pool = self.dense[2](bert_pool[2])

        token_types_ids = torch.ones(input_ids.size(0),input_ids.size(1)).long().to(input_ids.device)
        # token_types_ids = torch.cat([torch.zeros((1, input_ids.size(1))), torch.ones(
        #     (1, input_ids.size(1))),torch.zeros((1, input_ids.size(1)))], dim=0).long().to(input_ids.device)
        # token_types_ids = token_types_ids.repeat(
        #     1 if input_ids.size(0) // 3 == 0 else input_ids.size(0) // 3, 1)
        # bert_pool: (3*adm, H)
        # _, dx_bert_pool = self.bert(input_ids[:, 0, :], torch.zeros(
        #     (input_ids.size(0), inputs.size(2))).long().to(inputs.device), True)
        # _, rx_bert_pool = self.bert(inputs[:, 1, :], torch.zeros(
        #     (inputs.size(0), inputs.size(2))).long().to(inputs.device), True)
        # _, sx_bert_pool = self.bert(inputs[:, 2, :], torch.zeros(
        #     (inputs.size(0), inputs.size(2))).long().to(inputs.device), False)

        adm = int(input_ids.size(0)/3)
        _, dx_bert_pool = self.bert(input_ids[0:adm, :], token_types_ids[0:adm,:])
        _, rx_bert_pool = self.bert(input_ids[adm:adm*2, :], token_types_ids[adm:adm*2, :])
        _, sx_bert_pool = self.bert(input_ids[adm*2:adm*3, :], token_types_ids[adm*2:adm*3, :],False)
        bert_pool = torch.cat([dx_bert_pool,rx_bert_pool,sx_bert_pool],dim=0)
        _, bert_pool = self.bert(input_ids, token_types_ids)
        loss = 0
        bert_pool = bert_pool.view(3, -1, bert_pool.size(1))  # (3, adm, H)
        dx_bert_pool = self.dense[0](bert_pool[0])  # (adm, H)
        rx_bert_pool = self.dense[1](bert_pool[1])  # (adm, H)
        sx_bert_pool = self.dense[2](bert_pool[2])  # (adm, H)

        # mean and concat for rx prediction task
        rx_logits = []
        for i in range(rx_labels.size(0)):
            # mean
            dx_mean = torch.mean(dx_bert_pool[0:i+1, :], dim=0, keepdim=True)
            rx_mean = torch.mean(rx_bert_pool[0:i+1, :], dim=0, keepdim=True)
            sx_mean = torch.mean(sx_bert_pool[0:i+1, :], dim=0, keepdim=True)
            # concat
            concat = torch.cat(
                [dx_mean,dx_mean, sx_mean,sx_mean,], dim=-1)
            rx_logits.append(self.cls(concat))

        rx_logits = torch.cat(rx_logits, dim=0)
        loss = F.binary_cross_entropy_with_logits(rx_logits, rx_labels)
        return rx_logits
class GBERT_Predict_test(PreTrainedBertModel):
    def __init__(self, config: BertConfig,tokenizer,age_voc=None, body_tem_voc=None, pulse_rate_voc=None, breathing_rate_voc=None, lsbp_voc=None,ldbp_voc=None,rsbp_voc=None,rdbp_voc=None,
                 height_voc=None, weight_voc=None,waist_voc=None,bmi_voc=None,exercise_freq_voc=None,smoking_status_voc=None,drinking_freq_voc=None,
                 heart_rate_voc=None,tcho_voc=None,tg_voc=None,ldlc_voc=None,hdlc_voc=None,is_hyper_voc=None):
        super(GBERT_Predict_test, self).__init__(config)

        self.age_voc_size = len(age_voc.word2idx)
        self.body_tem_voc_size = len(body_tem_voc.word2idx)
        self.pulse_rate_voc_size = len(pulse_rate_voc.word2idx)
        self.breathing_rate_voc_size = len(breathing_rate_voc.word2idx)
        self.lsbp_voc_size = len(lsbp_voc.word2idx)
        self.ldbp_voc_size = len(ldbp_voc.word2idx)
        self.rsbp_voc_size = len(rsbp_voc.word2idx)
        self.rdbp_voc_size = len(rdbp_voc.word2idx)
        self.height_voc_size = len(height_voc.word2idx)
        self.weight_voc_size = len(weight_voc.word2idx)
        self.waist_voc_size = len(waist_voc.word2idx)
        self.bmi_voc_size = len(bmi_voc.word2idx)
        self.exercise_freq_voc_size = len(exercise_freq_voc.word2idx)
        self.smoking_status_voc_size = len(smoking_status_voc.word2idx)
        self.drinking_freq_voc_size = len(drinking_freq_voc.word2idx)
        self.heart_rate_voc_size = len(heart_rate_voc.word2idx)
        self.tcho_voc_size = len(tcho_voc.word2idx)
        self.tg_voc_size = len(tg_voc.word2idx)
        self.ldlc_voc_size = len(ldlc_voc.word2idx)
        self.hdlc_voc_size = len(hdlc_voc.word2idx)

        self.is_hyper_voc_size = len(is_hyper_voc.word2idx)

        # self.body_tem_voc_size= len(body_tem_voc.word2idx)
        # self.pulse_rate_voc_size=len(pulse_rate_voc.word2idx)
        # self.breathing_rate_voc_size= len(breathing_rate_voc.word2idx)
        # self.lsbp_voc_size= len(lsbp_voc.word2idx)
        # self.ldbp_voc_size= len(ldbp_voc.word2idx)
        # self.rsbp_voc_size= len(rsbp_voc.word2idx)
        # self.rdbp_voc_size= len(rdbp_voc.word2idx)
        # self.height_voc_size= len(height_voc.word2idx)
        # self.weight_voc_size= len(weight_voc.word2idx)
        # self.waist_voc_size= len(waist_voc.word2idx)
        # self.bmi_voc_size= len(bmi_voc.word2idx)
        # self.exercise_freq_voc_size= len(exercise_freq_voc.word2idx)
        # self.smoking_status_voc_size= len(smoking_status_voc.word2idx)
        # self.drinking_freq_voc_size= len(drinking_freq_voc.word2idx)
        # self.heart_rate_voc_size= len(heart_rate_voc.word2idx)
        # self.is_hyper_voc_size= len(is_hyper_voc.word2idx)


        self.bert = BERT(config,body_tem_voc, pulse_rate_voc, breathing_rate_voc, lsbp_voc,ldbp_voc)
        self.cls_pretrain = SelfSupervisedHead(
            config,self.age_voc_size,self.body_tem_voc_size, self.pulse_rate_voc_size, self.breathing_rate_voc_size, self.lsbp_voc_size,self.ldbp_voc_size,self.rsbp_voc_size,self.rdbp_voc_size,
                 self.height_voc_size, self.weight_voc_size,self.waist_voc_size,self.bmi_voc_size,self.exercise_freq_voc_size,self.smoking_status_voc_size,self.drinking_freq_voc_size,
                 self.heart_rate_voc_size,self.tcho_voc_size,self.tg_voc_size,self.ldlc_voc_size,self.hdlc_voc_size,self.is_hyper_voc_size)
        self.dense = nn.ModuleList(
            [MappingHead(config), MappingHead(config), MappingHead(config), MappingHead(config), MappingHead(config),
             MappingHead(config), MappingHead(config), MappingHead(config), MappingHead(config), MappingHead(config),
             MappingHead(config), MappingHead(config), MappingHead(config), MappingHead(config), MappingHead(config),
             MappingHead(config), MappingHead(config), MappingHead(config), MappingHead(config), MappingHead(config),
             MappingHead(config)])
        self.cls_predict = nn.Sequential(nn.Linear(40 * config.hidden_size, 2 * config.hidden_size), nn.ReLU(),
                                 nn.Linear(2 * config.hidden_size, len(tokenizer.is_hyper_voc_multi.word2idx)))

        self.apply(self.init_bert_weights)

    def forward(self, inputs, age_labels=None,body_tem_labels=None, pulse_rate_labels=None,breathing_rate_labels=None,
                                        lsbp_labels=None, ldbp_labels=None,rsbp_labels=None,rdbp_labels=None,
                                        height_labels=None, weight_labels=None,waist_labels=None,bmi_labels=None,
                                        exercise_freq_labels=None, smoking_status_labels=None,drinking_freq_labels=None,
                                        heart_rate_labels=None,tcho_labels=None,tg_labels=None,ldlc_labels=None,hdlc_labels=None,is_hyper_labels=None,epoch=None,flag=1):
        # inputs (B, 2, max_len)
        # bert_pool (B, hidden)
        # inputs -> torch.Size([4, 2, 55]) dx_labels -> torch.Size([4, 1997])
        # rx_labels -> torch.Size([4, 468]) dx_bert_pool ->torch.Size([4, 300]) rx_bert_pool ->rx_bert_pool
        #############预训练#################
        if flag==0:
            _, body_tem_bert_pool = self.bert(inputs[:, 0, :], torch.zeros(
                (inputs.size(0), inputs.size(2))).long().to(inputs.device))
            _, pulse_rate_bert_pool = self.bert(inputs[:, 1, :], torch.zeros(
                (inputs.size(0), inputs.size(2))).long().to(inputs.device))
            _, breathing_rate_bert_pool = self.bert(inputs[:, 2, :], torch.zeros(
                (inputs.size(0), inputs.size(2))).long().to(inputs.device))
            _, lsbp_bert_pool = self.bert(inputs[:, 3, :], torch.zeros(
                (inputs.size(0), inputs.size(2))).long().to(inputs.device))
            _, ldbp_bert_pool = self.bert(inputs[:, 4, :], torch.zeros(
                (inputs.size(0), inputs.size(2))).long().to(inputs.device))
            _, rsbp_bert_pool = self.bert(inputs[:, 5, :], torch.zeros(
                (inputs.size(0), inputs.size(2))).long().to(inputs.device))
            _, rdbp_bert_pool = self.bert(inputs[:, 6, :], torch.zeros(
                (inputs.size(0), inputs.size(2))).long().to(inputs.device))
            _, height_bert_pool = self.bert(inputs[:, 7, :], torch.zeros(
                (inputs.size(0), inputs.size(2))).long().to(inputs.device))
            _, weight_bert_pool = self.bert(inputs[:, 8, :], torch.zeros(
                (inputs.size(0), inputs.size(2))).long().to(inputs.device))
            _, waist_bert_pool = self.bert(inputs[:, 9, :], torch.zeros(
                (inputs.size(0), inputs.size(2))).long().to(inputs.device))
            _, bmi_bert_pool = self.bert(inputs[:, 10, :], torch.zeros(
                (inputs.size(0), inputs.size(2))).long().to(inputs.device))
            _, exercise_freq_bert_pool = self.bert(inputs[:, 11, :], torch.zeros(
                (inputs.size(0), inputs.size(2))).long().to(inputs.device))
            _, smoking_status_bert_pool = self.bert(inputs[:, 12, :], torch.zeros(
                (inputs.size(0), inputs.size(2))).long().to(inputs.device))
            _, drinking_freq_bert_pool = self.bert(inputs[:, 13, :], torch.zeros(
                (inputs.size(0), inputs.size(2))).long().to(inputs.device))
            _, heart_rate_bert_pool = self.bert(inputs[:, 14, :], torch.zeros(
                (inputs.size(0), inputs.size(2))).long().to(inputs.device))
            _, is_hyper_bert_pool = self.bert(inputs[:, 15, :], torch.zeros(
                (inputs.size(0), inputs.size(2))).long().to(inputs.device))

            body_tem2body_tem,pulse_rate2pulse_rate, breathing_rate2breathing_rate,lsbp2lsbp,ldbp2ldbp,rsbp2rsbp,rdbp2rdbp,height2height,weight2weight,\
            waist2waist,bmi2bmi,exercise_freq2exercise_freq,smoking_status2smoking_status,drinking_freq2drinking_freq,heart_rate2heart_rate,is_hyper2is_hyper,\
                    body_tem2is_hyper, pulse_rate2is_hyper, breathing_rate2is_hyper, lsbp2is_hyper, ldbp2is_hyper, rsbp2is_hyper, rdbp2is_hyper, height2is_hyper, weight2is_hyper, \
                    waist2is_hyper, bmi2is_hyper, exercise_freq2is_hyper, smoking_status2is_hyper, drinking_freq2is_hyper, heart_rate2is_hyper\
                        = self.cls_pretrain(body_tem_bert_pool, pulse_rate_bert_pool, breathing_rate_bert_pool,lsbp_bert_pool,ldbp_bert_pool,
                                           rsbp_bert_pool,rdbp_bert_pool,height_bert_pool,weight_bert_pool,waist_bert_pool,bmi_bert_pool,
                                           exercise_freq_bert_pool,smoking_status_bert_pool,drinking_freq_bert_pool,heart_rate_bert_pool,is_hyper_bert_pool)
            # output logits

            if body_tem_labels is None or pulse_rate_labels is None or breathing_rate_labels is None or lsbp_labels is None or \
                    ldbp_labels is None or rsbp_labels is None or rdbp_labels is None or height_labels is None or\
                    weight_labels is None or waist_labels is None or bmi_labels is None or exercise_freq_labels is None or smoking_status_labels is None or\
                    drinking_freq_labels is None or heart_rate_labels is None or is_hyper_labels is None:
                return F.sigmoid(body_tem2body_tem), F.sigmoid(pulse_rate2pulse_rate), F.sigmoid(breathing_rate2breathing_rate), \
                       F.sigmoid(lsbp2lsbp), F.sigmoid(ldbp2ldbp), F.sigmoid(rsbp2rsbp),F.sigmoid(rdbp2rdbp), F.sigmoid(height2height), \
                       F.sigmoid(weight2weight),F.sigmoid(waist2waist),F.sigmoid(bmi2bmi),F.sigmoid(exercise_freq2exercise_freq),F.sigmoid(smoking_status2smoking_status),\
                       F.sigmoid(drinking_freq2drinking_freq),F.sigmoid(heart_rate2heart_rate),F.sigmoid(is_hyper2is_hyper),F.sigmoid(body_tem2is_hyper),F.sigmoid(pulse_rate2is_hyper),\
                       F.sigmoid(breathing_rate2is_hyper),F.sigmoid(lsbp2is_hyper),F.sigmoid(ldbp2is_hyper),F.sigmoid(rsbp2is_hyper),F.sigmoid(rdbp2is_hyper),F.sigmoid(height2is_hyper),\
                       F.sigmoid(weight2is_hyper),F.sigmoid(waist2is_hyper),F.sigmoid(bmi2is_hyper),F.sigmoid(exercise_freq2is_hyper),F.sigmoid(smoking_status2is_hyper),F.sigmoid(drinking_freq2is_hyper),\
                       F.sigmoid(heart_rate2is_hyper)
            else:
                loss = F.binary_cross_entropy_with_logits(body_tem2body_tem, body_tem_labels) + \
                    F.binary_cross_entropy_with_logits(pulse_rate2pulse_rate, pulse_rate_labels) + \
                    F.binary_cross_entropy_with_logits(breathing_rate2breathing_rate, breathing_rate_labels) + \
                    F.binary_cross_entropy_with_logits(lsbp2lsbp, lsbp_labels) + \
                    F.binary_cross_entropy_with_logits(ldbp2ldbp, ldbp_labels) + \
                    F.binary_cross_entropy_with_logits(rsbp2rsbp, rsbp_labels) + \
                    F.binary_cross_entropy_with_logits(rdbp2rdbp,rdbp_labels) + \
                    F.binary_cross_entropy_with_logits(height2height, height_labels) + \
                    F.binary_cross_entropy_with_logits(weight2weight, weight_labels)+\
                    F.binary_cross_entropy_with_logits(waist2waist, waist_labels) + \
                    F.binary_cross_entropy_with_logits(bmi2bmi, bmi_labels) + \
                    F.binary_cross_entropy_with_logits(exercise_freq2exercise_freq, exercise_freq_labels) + \
                    F.binary_cross_entropy_with_logits(smoking_status2smoking_status, smoking_status_labels) + \
                    F.binary_cross_entropy_with_logits(drinking_freq2drinking_freq, drinking_freq_labels) + \
                    F.binary_cross_entropy_with_logits(heart_rate2heart_rate, heart_rate_labels) + \
                    F.binary_cross_entropy_with_logits(is_hyper2is_hyper, is_hyper_labels) + \
                    F.binary_cross_entropy_with_logits(body_tem2is_hyper, is_hyper_labels) + \
                    F.binary_cross_entropy_with_logits(pulse_rate2is_hyper, is_hyper_labels) + \
                    F.binary_cross_entropy_with_logits(breathing_rate2is_hyper, is_hyper_labels) + \
                    F.binary_cross_entropy_with_logits(lsbp2is_hyper, is_hyper_labels) + \
                    F.binary_cross_entropy_with_logits(ldbp2is_hyper, is_hyper_labels) + \
                    F.binary_cross_entropy_with_logits(rsbp2is_hyper, is_hyper_labels) + \
                    F.binary_cross_entropy_with_logits(rdbp2is_hyper, is_hyper_labels) + \
                    F.binary_cross_entropy_with_logits(height2is_hyper, is_hyper_labels) + \
                    F.binary_cross_entropy_with_logits(weight2is_hyper, is_hyper_labels) + \
                    F.binary_cross_entropy_with_logits(waist2is_hyper, is_hyper_labels) + \
                    F.binary_cross_entropy_with_logits(bmi2is_hyper, is_hyper_labels) + \
                    F.binary_cross_entropy_with_logits(exercise_freq2is_hyper, is_hyper_labels) + \
                    F.binary_cross_entropy_with_logits(smoking_status2is_hyper, is_hyper_labels) + \
                    F.binary_cross_entropy_with_logits(drinking_freq2is_hyper, is_hyper_labels)

                return loss, F.sigmoid(body_tem2body_tem), F.sigmoid(pulse_rate2pulse_rate), F.sigmoid(breathing_rate2breathing_rate), \
                       F.sigmoid(lsbp2lsbp), F.sigmoid(ldbp2ldbp), F.sigmoid(rsbp2rsbp),F.sigmoid(rdbp2rdbp), F.sigmoid(height2height), \
                       F.sigmoid(weight2weight),F.sigmoid(waist2waist),F.sigmoid(bmi2bmi),F.sigmoid(exercise_freq2exercise_freq),F.sigmoid(smoking_status2smoking_status),\
                       F.sigmoid(drinking_freq2drinking_freq),F.sigmoid(heart_rate2heart_rate),F.sigmoid(is_hyper2is_hyper),F.sigmoid(body_tem2is_hyper),F.sigmoid(pulse_rate2is_hyper),\
                       F.sigmoid(breathing_rate2is_hyper),F.sigmoid(lsbp2is_hyper),F.sigmoid(ldbp2is_hyper),F.sigmoid(rsbp2is_hyper),F.sigmoid(rdbp2is_hyper),F.sigmoid(height2is_hyper),\
                       F.sigmoid(weight2is_hyper),F.sigmoid(waist2is_hyper),F.sigmoid(bmi2is_hyper),F.sigmoid(exercise_freq2is_hyper),F.sigmoid(smoking_status2is_hyper),F.sigmoid(drinking_freq2is_hyper),\
                       F.sigmoid(heart_rate2is_hyper)
        elif flag==1:
            token_types_ids = torch.cat([torch.zeros((1, inputs.size(1))), torch.zeros((1, inputs.size(1))),
                                         torch.zeros((1, inputs.size(1))), torch.zeros((1, inputs.size(1))),
                                         torch.zeros((1, inputs.size(1))), torch.zeros((1, inputs.size(1))),
                                         torch.zeros((1, inputs.size(1))), torch.zeros((1, inputs.size(1))),
                                         torch.zeros((1, inputs.size(1))), torch.zeros((1, inputs.size(1))),
                                         torch.zeros((1, inputs.size(1))), torch.zeros((1, inputs.size(1))),
                                         torch.zeros((1, inputs.size(1))), torch.zeros((1, inputs.size(1))),
                                         torch.zeros((1, inputs.size(1))), torch.zeros((1, inputs.size(1))),
                                         torch.zeros((1, inputs.size(1))), torch.zeros((1, inputs.size(1))),
                                         torch.zeros((1, inputs.size(1))), torch.zeros((1, inputs.size(1))), torch.ones(
                    (1, inputs.size(1)))], dim=0).long().to(inputs.device)
            token_types_ids = token_types_ids.repeat(
                1 if inputs.size(0) // 21 == 0 else inputs.size(0) // 21, 1)
            _, bert_pool = self.bert(inputs, token_types_ids)
            loss = 0
            bert_pool = bert_pool.view(21, -1, bert_pool.size(1))  # (3, adm, H)#bert_pool(16,1,300)
            # dx_bert_pool = self.dense[0](bert_pool[0])  # (adm, H)
            # rx_bert_pool = self.dense[1](bert_pool[1])  # (adm, H)
            # sx_bert_pool = self.dense[2](bert_pool[2])  # (adm, H)
            age_bert_pool = self.dense[0](bert_pool[0])
            body_tem_bert_pool = self.dense[1](bert_pool[1])  # size(1,300)
            pulse_rate_bert_pool = self.dense[2](bert_pool[2])
            breathing_rate_bert_pool = self.dense[3](bert_pool[3])
            lsbp_bert_pool = self.dense[4](bert_pool[4])
            ldbp_bert_pool = self.dense[5](bert_pool[5])
            rsbp_bert_pool = self.dense[6](bert_pool[6])
            rdbp_bert_pool = self.dense[7](bert_pool[7])
            height_bert_pool = self.dense[8](bert_pool[8])
            weight_bert_pool = self.dense[9](bert_pool[9])
            waist_bert_pool = self.dense[10](bert_pool[10])
            bmi_bert_pool = self.dense[11](bert_pool[11])
            exercise_freq_bert_pool = self.dense[12](bert_pool[1])
            smoking_status_bert_pool = self.dense[13](bert_pool[13])
            drinking_freq_bert_pool = self.dense[14](bert_pool[14])
            heart_rate_bert_pool = self.dense[15](bert_pool[15])
            tcho_bert_pool = self.dense[16](bert_pool[16])
            tg_bert_pool = self.dense[17](bert_pool[17])
            ldlc_bert_pool = self.dense[18](bert_pool[18])
            hdlc_bert_pool = self.dense[19](bert_pool[19])
            # is_hyper_bert_pool = self.dense[15](bert_pool[15])

            # mean and concat for rx prediction task
            is_hyper_logits = []
            for i in range(1):
                # mean
                age_mean = torch.mean(age_bert_pool[0:i + 1, :], dim=0, keepdim=True)
                body_tem_mean = torch.mean(body_tem_bert_pool[0:i + 1, :], dim=0, keepdim=True)
                pulse_rate_mean = torch.mean(pulse_rate_bert_pool[0:i + 1, :], dim=0, keepdim=True)
                breathing_rate_mean = torch.mean(breathing_rate_bert_pool[0:i + 1, :], dim=0, keepdim=True)
                lsbp_mean = torch.mean(lsbp_bert_pool[0:i + 1, :], dim=0, keepdim=True)
                ldbp_mean = torch.mean(ldbp_bert_pool[0:i + 1, :], dim=0, keepdim=True)
                rsbp_mean = torch.mean(rsbp_bert_pool[0:i + 1, :], dim=0, keepdim=True)
                rdbp_mean = torch.mean(rdbp_bert_pool[0:i + 1, :], dim=0, keepdim=True)
                height_mean = torch.mean(height_bert_pool[0:i + 1, :], dim=0, keepdim=True)
                weight_mean = torch.mean(weight_bert_pool[0:i + 1, :], dim=0, keepdim=True)
                waist_mean = torch.mean(waist_bert_pool[0:i + 1, :], dim=0, keepdim=True)
                bmi_mean = torch.mean(bmi_bert_pool[0:i + 1, :], dim=0, keepdim=True)
                exercise_freq_mean = torch.mean(exercise_freq_bert_pool[0:i + 1, :], dim=0, keepdim=True)
                smoking_status_mean = torch.mean(smoking_status_bert_pool[0:i + 1, :], dim=0, keepdim=True)
                drinking_freq_mean = torch.mean(drinking_freq_bert_pool[0:i + 1, :], dim=0, keepdim=True)
                heart_rate_mean = torch.mean(heart_rate_bert_pool[0:i + 1, :], dim=0, keepdim=True)
                tcho_mean = torch.mean(tcho_bert_pool[0:i + 1, :], dim=0, keepdim=True)
                tg_mean = torch.mean(tg_bert_pool[0:i + 1, :], dim=0, keepdim=True)
                ldlc_mean = torch.mean(ldlc_bert_pool[0:i + 1, :], dim=0, keepdim=True)
                hdlc_mean = torch.mean(hdlc_bert_pool[0:i + 1, :], dim=0, keepdim=True)

                # is_hyper_mean = torch.mean(is_hyper_bert_pool[0:i + 1, :], dim=0, keepdim=True)
                # dx_mean = torch.mean(dx_bert_pool[0:i+1, :], dim=0, keepdim=True)
                # rx_mean = torch.mean(rx_bert_pool[0:i+1, :], dim=0, keepdim=True)
                # sx_mean = torch.mean(sx_bert_pool[0:i + 1, :], dim=0, keepdim=True)
                # concat
                concat = torch.cat(
                    [age_mean,body_tem_mean, pulse_rate_mean, breathing_rate_mean, \
                     lsbp_mean, ldbp_mean, rsbp_mean, rdbp_mean, height_mean, weight_mean, \
                     waist_mean, bmi_mean, exercise_freq_mean, smoking_status_mean, \
                     drinking_freq_mean, heart_rate_mean,tcho_mean,tg_mean,ldlc_mean,hdlc_mean,\
                     age_bert_pool[i, :].unsqueeze(dim=0),\
                     body_tem_bert_pool[i, :].unsqueeze(dim=0),\
                     pulse_rate_bert_pool[i, :].unsqueeze(dim=0),\
                     breathing_rate_bert_pool[i, :].unsqueeze(dim=0), \
                     lsbp_bert_pool[i, :].unsqueeze(dim=0), ldbp_bert_pool[i, :].unsqueeze(dim=0),\
                     rsbp_bert_pool[i, :].unsqueeze(dim=0), rdbp_bert_pool[i, :].unsqueeze(dim=0),\
                     height_bert_pool[i, :].unsqueeze(dim=0), weight_bert_pool[i, :].unsqueeze(dim=0), \
                     waist_bert_pool[i, :].unsqueeze(dim=0), bmi_bert_pool[i, :].unsqueeze(dim=0),\
                     exercise_freq_bert_pool[i, :].unsqueeze(dim=0),\
                     smoking_status_bert_pool[i, :].unsqueeze(dim=0), \
                     drinking_freq_bert_pool[i, :].unsqueeze(dim=0),\
                     heart_rate_bert_pool[i, :].unsqueeze(dim=0),\
                     tcho_bert_pool[i, :].unsqueeze(dim=0),\
                     tg_bert_pool[i, :].unsqueeze(dim=0),\
                     ldlc_bert_pool[i, :].unsqueeze(dim=0),\
                     hdlc_bert_pool[i, :].unsqueeze(dim=0)],dim=-1)

                is_hyper_logits.append(self.cls_predict(concat))

            is_hyper_logits = torch.cat(is_hyper_logits, dim=0)
            # loss = F.binary_cross_entropy_with_logits(is_hyper_logits, is_hyper_labels)
            return is_hyper_logits

class MappingHead(nn.Module):
    def __init__(self, config: BertConfig):
        super(MappingHead, self).__init__()
        self.dense = nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size),
                                   nn.ReLU())

    def forward(self, input):
        return self.dense(input)

class PGD():
    def __init__(self, model):
        self.model = model
        self.emb_backup = {}
        self.grad_backup = {}

    def attack(self, epsilon=1., alpha=0.3, emb_name='input_ids', is_first_attack=False):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                if is_first_attack:
                    self.emb_backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = alpha * param.grad / norm
                    param.data.add_(r_at)
                    param.data = self.project(name, param.data, epsilon)

    def restore(self, emb_name='input_ids'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.emb_backup
                param.data = self.emb_backup[name]
        self.emb_backup = {}

    def project(self, param_name, param_data, epsilon):
        r = param_data - self.emb_backup[param_name]
        if torch.norm(r) > epsilon:
            r = epsilon * r / torch.norm(r)
        return self.emb_backup[param_name] + r

    def backup_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.grad_backup[name] = param.grad.clone()

    def restore_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.grad = self.grad_backup[name]

rx_embedding="bert.embedding.ontology_embedding.rx_embedding.embedding"
class FGM():
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=1.0, rx_embedding='bert.embedding.ontology_embedding.rx_embedding.embedding',dx_embedding='bert.embedding.ontology_embedding.dx_embedding.embedding'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and (rx_embedding in name or dx_embedding in name):
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, rx_embedding='bert.embedding.ontology_embedding.rx_embedding.embedding',dx_embedding='bert.embedding.ontology_embedding.dx_embedding.embedding'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and (rx_embedding in name or dx_embedding in name):
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}
class GBERT_Predict(PreTrainedBertModel):
    def __init__(self, config: BertConfig, tokenizer):
        super(GBERT_Predict, self).__init__(config)
        self.bert = BERT(config, tokenizer.body_tem_voc,tokenizer.pulse_rate_voc,tokenizer.breathing_rate_voc,tokenizer.lsbp_voc,tokenizer.ldbp_voc)
        self.dense = nn.ModuleList([MappingHead(config), MappingHead(config), MappingHead(config),MappingHead(config), MappingHead(config), MappingHead(config),MappingHead(config), MappingHead(config), MappingHead(config),MappingHead(config), MappingHead(config), MappingHead(config),MappingHead(config), MappingHead(config), MappingHead(config),MappingHead(config)])
        self.cls = nn.Sequential(nn.Linear(31*config.hidden_size, 2*config.hidden_size),nn.ReLU(), nn.Linear(2*config.hidden_size, len(tokenizer.is_hyper_voc_multi.word2idx)))

        self.apply(self.init_bert_weights)
        # self.input_ids=input_ids

    def forward(self, input_ids, body_tem_labels=None, pulse_rate_labels=None,breathing_rate_labels=None,
                                        lsbp_labels=None, ldbp_labels=None,rsbp_labels=None,rdbp_labels=None,
                                        height_labels=None, weight_labels=None,waist_labels=None,bmi_labels=None,
                                        exercise_freq_labels=None, smoking_status_labels=None,drinking_freq_labels=None,
                                        heart_rate_labels=None,is_hyper_labels=None,epoch=None):
        """
        :param input_ids: [B, max_seq_len] where B = 3*adm
        :param rx_labels: [adm-1, rx_size]
        :param dx_labels: [adm-1, dx_size]
        :return:
        """
        # token_types_ids = torch.ones(input_ids.size(0), input_ids.size(1)).long().to(input_ids.device)
        # # token_types_ids = torch.cat([torch.zeros((1, input_ids.size(1))), torch.ones(
        # #     (1, input_ids.size(1)))], dim=0).long().to(input_ids.device)
        # token_types_ids = token_types_ids.repeat(
        #     1 if input_ids.size(0)//3 == 0 else input_ids.size(0)//3, 1)
        # bert_pool: (2*adm, H)
        # _, bert_pool = self.bert(input_ids, token_types_ids)
        # loss = 0
        # bert_pool = bert_pool.view(3, -1, bert_pool.size(1))  # (2, adm, H)
        # dx_bert_pool = self.dense[0](bert_pool[0])  # (adm, H)
        # rx_bert_pool = self.dense[1](bert_pool[1])  # (adm, H)
        # sx_bert_pool = self.dense[2](bert_pool[2])
        # token_types_ids = torch.cat([torch.zeros((1, input_ids.size(1))), torch.ones(
        #     (1, input_ids.size(1)))], dim=0).long().to(input_ids.device)
        # token_types_ids = token_types_ids.repeat(
        #     1 if input_ids.size(0) // 16 == 0 else input_ids.size(0) // 16, 1)
        # token_types_ids = torch.ones(input_ids.size(0),input_ids.size(1)).long().to(input_ids.device)
        token_types_ids = torch.cat([torch.zeros((1, input_ids.size(1))),torch.zeros((1, input_ids.size(1))),torch.zeros((1, input_ids.size(1))),torch.zeros((1, input_ids.size(1))),torch.zeros((1, input_ids.size(1))),torch.zeros((1, input_ids.size(1))),torch.zeros((1, input_ids.size(1))),torch.zeros((1, input_ids.size(1))),torch.zeros((1, input_ids.size(1))),torch.zeros((1, input_ids.size(1))),torch.zeros((1, input_ids.size(1))),torch.zeros((1, input_ids.size(1))),torch.zeros((1, input_ids.size(1))),torch.zeros((1, input_ids.size(1))),torch.zeros((1, input_ids.size(1))), torch.ones(
            (1, input_ids.size(1)))], dim=0).long().to(input_ids.device)
        token_types_ids = token_types_ids.repeat(
            1 if input_ids.size(0) // 16 == 0 else input_ids.size(0) // 16, 1)
        # token_types_ids = torch.cat([torch.zeros((1, input_ids.size(1))), torch.ones(
        #     (1, input_ids.size(1))),torch.zeros((1, input_ids.size(1)))], dim=0).long().to(input_ids.device)
        # token_types_ids = token_types_ids.repeat(
        #     1 if input_ids.size(0) // 3 == 0 else input_ids.size(0) // 3, 1)
        # bert_pool: (3*adm, H)
        # _, dx_bert_pool = self.bert(input_ids[:, 0, :], torch.zeros(
        #     (input_ids.size(0), inputs.size(2))).long().to(inputs.device), True)
        # _, rx_bert_pool = self.bert(inputs[:, 1, :], torch.zeros(
        #     (inputs.size(0), inputs.size(2))).long().to(inputs.device), True)
        # _, sx_bert_pool = self.bert(inputs[:, 2, :], torch.zeros(
        #     (inputs.size(0), inputs.size(2))).long().to(inputs.device), False)

        # adm = int(input_ids.size(0)/16)
        # _, body_tem_bert_pool = self.bert(input_ids[0:adm, :], token_types_ids[0:adm,:])
        # _, pulse_rate_bert_pool = self.bert(input_ids[adm:adm*2, :], token_types_ids[adm:adm*2, :])
        # _, breathing_rate_bert_pool = self.bert(input_ids[adm*2:adm*3, :], token_types_ids[adm*2:adm*3, :],False)
        # _, lsbp_bert_pool = self.bert(input_ids[adm*3:adm*4, :], token_types_ids[adm*3:adm*4, :])
        # _, ldbp_bert_pool = self.bert(input_ids[adm*4:adm*5, :], token_types_ids[adm*4:adm*5, :])
        # _, rsbp_bert_pool = self.bert(input_ids[adm * 5:adm * 6, :], token_types_ids[adm * 5:adm * 6, :])
        # _, rdbp_bert_pool = self.bert(input_ids[adm * 6:adm * 7, :], token_types_ids[adm * 6:adm * 7, :])
        # _, height_bert_pool = self.bert(input_ids[adm * 7:adm * 8, :], token_types_ids[adm * 7:adm * 8, :])
        # _, weight_bert_pool = self.bert(input_ids[adm * 8:adm * 9, :], token_types_ids[adm * 8:adm * 9, :])
        # _, waist_bert_pool = self.bert(input_ids[adm * 9:adm * 10, :], token_types_ids[adm * 9:adm * 10, :])
        # _, bmi_bert_pool = self.bert(input_ids[adm * 10:adm * 11, :], token_types_ids[adm * 10:adm * 11, :])
        # _, exercise_freq_bert_pool = self.bert(input_ids[adm * 11:adm * 12, :], token_types_ids[adm * 11:adm * 12, :])
        # _, smoking_status_bert_pool = self.bert(input_ids[adm * 12:adm * 13, :], token_types_ids[adm * 12:adm * 13, :])
        # _, drinking_freq_bert_pool = self.bert(input_ids[adm * 13:adm * 14, :], token_types_ids[adm * 13:adm * 14, :])
        # _, heart_rate_bert_pool = self.bert(input_ids[adm * 14:adm * 15, :], token_types_ids[adm * 14:adm * 15, :])
        # _, is_hyper_bert_pool = self.bert(input_ids[adm * 15:adm * 16, :], token_types_ids[adm * 15:adm * 16, :])
        # bert_pool = torch.cat([body_tem_bert_pool, pulse_rate_bert_pool,breathing_rate_bert_pool,\
        #         lsbp_bert_pool, ldbp_bert_pool,rsbp_bert_pool,rdbp_bert_pool, height_bert_pool,weight_bert_pool,\
        #         waist_bert_pool, bmi_bert_pool,exercise_freq_bert_pool, smoking_status_bert_pool,\
        #         drinking_freq_bert_pool,heart_rate_bert_pool,is_hyper_bert_pool],dim=0)
        _, bert_pool = self.bert(input_ids, token_types_ids)
        loss = 0
        bert_pool = bert_pool.view(16, -1, bert_pool.size(1))  # (3, adm, H)
        # dx_bert_pool = self.dense[0](bert_pool[0])  # (adm, H)
        # rx_bert_pool = self.dense[1](bert_pool[1])  # (adm, H)
        # sx_bert_pool = self.dense[2](bert_pool[2])  # (adm, H)
        body_tem_bert_pool = self.dense[0](bert_pool[0])
        pulse_rate_bert_pool =self.dense[1](bert_pool[1])
        breathing_rate_bert_pool = self.dense[2](bert_pool[2])
        lsbp_bert_pool = self.dense[3](bert_pool[3])
        ldbp_bert_pool = self.dense[4](bert_pool[4])
        rsbp_bert_pool = self.dense[5](bert_pool[5])
        rdbp_bert_pool = self.dense[6](bert_pool[6])
        height_bert_pool = self.dense[7](bert_pool[7])
        weight_bert_pool = self.dense[8](bert_pool[8])
        waist_bert_pool = self.dense[9](bert_pool[9])
        bmi_bert_pool = self.dense[10](bert_pool[10])
        exercise_freq_bert_pool = self.dense[11](bert_pool[11])
        smoking_status_bert_pool = self.dense[12](bert_pool[12])
        drinking_freq_bert_pool = self.dense[13](bert_pool[13])
        heart_rate_bert_pool = self.dense[14](bert_pool[14])
        is_hyper_bert_pool = self.dense[15](bert_pool[15])


        # mean and concat for rx prediction task
        is_hyper_logits = []
        for i in range(is_hyper_labels.size(0)):
            # mean
            body_tem_mean = torch.mean(body_tem_bert_pool[0:i+1, :], dim=0, keepdim=True)
            pulse_rate_mean = torch.mean(pulse_rate_bert_pool[0:i+1, :], dim=0, keepdim=True)
            breathing_rate_mean = torch.mean(breathing_rate_bert_pool[0:i+1, :], dim=0, keepdim=True)
            lsbp_mean =torch.mean(lsbp_bert_pool[0:i+1, :], dim=0, keepdim=True)
            ldbp_mean = torch.mean(ldbp_bert_pool[0:i+1, :], dim=0, keepdim=True)
            rsbp_mean = torch.mean(rsbp_bert_pool[0:i+1, :], dim=0, keepdim=True)
            rdbp_mean = torch.mean(rdbp_bert_pool[0:i+1, :], dim=0, keepdim=True)
            height_mean = torch.mean(height_bert_pool[0:i+1, :], dim=0, keepdim=True)
            weight_mean = torch.mean(weight_bert_pool[0:i+1, :], dim=0, keepdim=True)
            waist_mean = torch.mean(waist_bert_pool[0:i+1, :], dim=0, keepdim=True)
            bmi_mean = torch.mean(bmi_bert_pool[0:i+1, :], dim=0, keepdim=True)
            exercise_freq_mean = torch.mean(exercise_freq_bert_pool[0:i+1, :], dim=0, keepdim=True)
            smoking_status_mean = torch.mean(smoking_status_bert_pool[0:i+1, :], dim=0, keepdim=True)
            drinking_freq_mean = torch.mean(drinking_freq_bert_pool[0:i+1, :], dim=0, keepdim=True)
            heart_rate_mean = torch.mean(heart_rate_bert_pool[0:i+1, :], dim=0, keepdim=True)
            is_hyper_mean = torch.mean(is_hyper_bert_pool[0:i+1, :], dim=0, keepdim=True)
            # dx_mean = torch.mean(dx_bert_pool[0:i+1, :], dim=0, keepdim=True)
            # rx_mean = torch.mean(rx_bert_pool[0:i+1, :], dim=0, keepdim=True)
            # sx_mean = torch.mean(sx_bert_pool[0:i + 1, :], dim=0, keepdim=True)
            # concat
            concat = torch.cat(
                [body_tem_mean, pulse_rate_mean,breathing_rate_mean,\
                lsbp_mean, ldbp_mean,rsbp_mean,rdbp_mean, height_mean,weight_mean,\
                waist_mean, bmi_mean,exercise_freq_mean, smoking_status_mean,\
                drinking_freq_mean,heart_rate_mean,is_hyper_mean, body_tem_bert_pool[i+1, :].unsqueeze(dim=0), pulse_rate_bert_pool[i+1, :].unsqueeze(dim=0),breathing_rate_bert_pool[i+1, :].unsqueeze(dim=0),\
                lsbp_bert_pool[i+1, :].unsqueeze(dim=0), ldbp_bert_pool[i+1, :].unsqueeze(dim=0),rsbp_bert_pool[i+1, :].unsqueeze(dim=0),rdbp_bert_pool[i+1, :].unsqueeze(dim=0), height_bert_pool[i+1, :].unsqueeze(dim=0),weight_bert_pool[i+1, :].unsqueeze(dim=0),\
                waist_bert_pool[i+1, :].unsqueeze(dim=0), bmi_bert_pool[i+1, :].unsqueeze(dim=0),exercise_freq_bert_pool[i+1, :].unsqueeze(dim=0), smoking_status_bert_pool[i+1, :].unsqueeze(dim=0),\
                drinking_freq_bert_pool[i+1, :].unsqueeze(dim=0),heart_rate_bert_pool[i+1, :].unsqueeze(dim=0)], dim=-1)
            is_hyper_logits.append(self.cls(concat))

        is_hyper_logits = torch.cat(is_hyper_logits, dim=0)
        loss = F.binary_cross_entropy_with_logits(is_hyper_logits, is_hyper_labels)
        return loss, is_hyper_logits

class GBERT_Predict_Side(PreTrainedBertModel):
    def __init__(self, config: BertConfig, tokenizer, side_len):
        super(GBERT_Predict_Side, self).__init__(config)
        self.bert = BERT(config, tokenizer.dx_voc, tokenizer.rx_voc)
        self.dense = nn.ModuleList([MappingHead(config), MappingHead(config)])
        self.cls = nn.Sequential(nn.Linear(3*config.hidden_size, 2*config.hidden_size),
                                 nn.ReLU(), nn.Linear(2*config.hidden_size, len(tokenizer.rx_voc_multi.word2idx)))

        self.side = nn.Sequential(nn.Linear(
            side_len, side_len // 2), nn.ReLU(), nn.Linear(side_len // 2, side_len // 2))
        self.final_cls = nn.Sequential(nn.ReLU(), nn.Linear(len(
            tokenizer.rx_voc_multi.word2idx) + side_len // 2, len(tokenizer.rx_voc_multi.word2idx)))
        # self.cls = nn.Sequential(nn.Linear(3*config.hidden_size, len(tokenizer.rx_voc_multi.word2idx)))
        # self.gru = nn.GRU(config.hidden_size, config.hidden_size, batch_first=True)

        self.apply(self.init_bert_weights)

    def forward(self, input_ids, dx_labels=None, rx_labels=None, epoch=None, input_sides=None):
        """
        :param input_ids: [B, max_seq_len] where B = 2*adm
        :param rx_labels: [adm-1, rx_size]
        :param dx_labels: [adm-1, dx_size]
        :param input_side: [adm-1, side_len]
        :return:
        """
        token_types_ids = torch.cat([torch.zeros((1, input_ids.size(1))), torch.ones(
            (1, input_ids.size(1)))], dim=0).long().to(input_ids.device)
        token_types_ids = token_types_ids.repeat(
            1 if input_ids.size(0)//2 == 0 else input_ids.size(0)//2, 1)
        # bert_pool: (2*adm, H)
        _, bert_pool = self.bert(input_ids, token_types_ids)
        loss = 0
        bert_pool = bert_pool.view(2, -1, bert_pool.size(1))  # (2, adm, H)
        dx_bert_pool = self.dense[0](bert_pool[0])  # (adm, H)
        rx_bert_pool = self.dense[1](bert_pool[1])  # (adm, H)

        # mean and concat for rx prediction task
        visit_vecs = []
        for i in range(rx_labels.size(0)):
            # mean
            dx_mean = torch.mean(dx_bert_pool[0:i+1, :], dim=0, keepdim=True)
            rx_mean = torch.mean(rx_bert_pool[0:i+1, :], dim=0, keepdim=True)
            # concat
            concat = torch.cat(
                [dx_mean, rx_mean, dx_bert_pool[i+1, :].unsqueeze(dim=0)], dim=-1)
            concat_trans = self.cls(concat)
            visit_vecs.append(concat_trans)

        visit_vecs = torch.cat(visit_vecs, dim=0)
        # add side and concat
        side_trans = self.side(input_sides)
        patient_vec = torch.cat([visit_vecs, side_trans], dim=1)

        rx_logits = self.final_cls(patient_vec)
        loss = F.binary_cross_entropy_with_logits(rx_logits, rx_labels)
        return loss, rx_logits

# ------------------------------------------------------------
