from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# import fitlog
import json

import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import csv
import os
import logging
import argparse
import random
from tqdm import tqdm, trange
import dill
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.optim import Adam
from tensorboardX import SummaryWriter

from utils import metric_report, t2n, get_n_params
from config import BertConfig
from predictive_models import GBERT_Predict,FGM,PGD

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

def text_save(filename,list_data):
    file=open(filename,'a')
    for data in list_data:
        file.write(str(data)+'\n')

def text_save_data(filename,data):
    file=open(filename,'a')
    file.write(str(data)+'\n')

f1_eval_list_run=list()
f1_test_list_run=list()
class Voc(object):
    def __init__(self):
        self.idx2word = {}
        self.word2idx = {}

    def add_sentence(self, sentence):
        for word in sentence:
            if word not in self.word2idx:
                self.idx2word[len(self.word2idx)] = word
                self.word2idx[word] = len(self.word2idx)


class EHRTokenizer(object):
    """Runs end-to-end tokenization"""

    def __init__(self, data_dir, special_tokens=("[PAD]", "[CLS]", "[MASK]")):

        self.vocab = Voc()

        # special tokens
        self.vocab.add_sentence(special_tokens)

        self.rx_voc = self.add_vocab(os.path.join(data_dir, 'rx-vocab.txt'))
        self.dx_voc = self.add_vocab(os.path.join(data_dir, 'dx-vocab.txt'))
        self.sx_voc = self.add_vocab(os.path.join(data_dir, 'sx-vocab.txt'))

        # code1 only in multi-visit data
        self.rx_voc_multi = Voc()
        self.dx_voc_multi = Voc()
        self.sx_voc_multi = Voc()
        with open(os.path.join(data_dir, 'rx-vocab-multi.txt'), 'r') as fin:
            for code in fin:
                self.rx_voc_multi.add_sentence([code.rstrip('\n')])
        with open(os.path.join(data_dir, 'dx-vocab-multi.txt'), 'r') as fin:
            for code in fin:
                self.dx_voc_multi.add_sentence([code.rstrip('\n')])
        with open(os.path.join(data_dir, 'sx-vocab-multi.txt'), 'r', encoding="utf-8") as fin:
            for code in fin:
                self.sx_voc_multi.add_sentence([code.rstrip('\n')])

    def add_vocab(self, vocab_file):
        voc = self.vocab
        specific_voc = Voc()
        with open(vocab_file, 'r', encoding="utf-8") as fin:
            for code in fin:
                voc.add_sentence([code.rstrip('\n')])
                specific_voc.add_sentence([code.rstrip('\n')])
        return specific_voc

    def convert_tokens_to_ids(self, tokens):
        """Converts a sequence of tokens into ids using the vocab."""
        ids = []
        for token in tokens:
            ids.append(self.vocab.word2idx[token])
            # print(ids)
        return ids

    def convert_ids_to_tokens(self, ids):
        """Converts a sequence of ids in wordpiece tokens using the vocab."""
        tokens = []
        for i in ids:
            tokens.append(self.vocab.idx2word[i])
        return tokens

def split_str(x):
    return x[0].split(",")

def change_str(x):
    if len(x) > 50:
        return x[:50]
    else:
        return x
# def change_ATC(x):
#     if "110005" in str(x) and len(x)>1:
#         x.remove("110005")
#         return x
#     else:
#         return x

class EHRDataset(Dataset):
    def __init__(self, data_pd, tokenizer: EHRTokenizer, max_seq_len):
        data_pd["CHIEF_COMPLAINTS"] = data_pd["CHIEF_COMPLAINTS"].apply(split_str)
        data_pd["CHIEF_COMPLAINTS"] = data_pd["CHIEF_COMPLAINTS"].apply(change_str)
        data_pd["ATC4"] = data_pd["ATC4"].apply(change_str)
        # data_pd["ATC4"] = data_pd["ATC4"].apply(change_ATC)
        # d_ATC=data_pd["ATC4"]
        # i=0
        # for ATC in d_ATC.values:
        #     if "110005" in str(ATC) and len(ATC)>1:
        #         li=ATC.remove("110005")
        #         d_ATC[i] =li
        #         print(ATC)
        #     i=i+1
        data_pd["ICD9_CODE"] = data_pd["ICD9_CODE"].apply(change_str)
        self.data_pd = data_pd[:100]
        self.tokenizer = tokenizer
        self.seq_len = max_seq_len

        self.sample_counter = 0

        def transform_data(data):
            """
            :param data: raw data form
            :return: {subject_id, [adm, 2, codes]},
            """
            records = {}
            for subject_id in data['SUBJECT_ID'].unique():
                item_df = data[data['SUBJECT_ID'] == subject_id]
                patient = []
                for _, row in item_df.iterrows():
                    admission = [list(row['ICD9_CODE']), list(row['ATC4']), list(row['CHIEF_COMPLAINTS'])]
                    patient.append(admission)
                if len(patient) < 2:
                    continue
                records[subject_id] = patient
            return records

        self.records = transform_data(data_pd)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, item):
        cur_id = self.sample_counter
        self.sample_counter += 1
        subject_id = list(self.records.keys())[item]

        def fill_to_max(l, seq):
            while len(l) < seq:
                l.append('[PAD]')
            return l

        """extract input and output tokens
        """
        input_tokens = []  # (3*max_len*adm)
        output_dx_tokens = []  # (adm-1, l)
        output_rx_tokens = []  # (adm-1, l)
        output_sx_tokens = []  # (adm-1, l)

        for idx, adm in enumerate(self.records[subject_id]):
            input_tokens.extend(
                ['[CLS]'] + fill_to_max(list(adm[0]), self.seq_len - 1))
            input_tokens.extend(
                ['[CLS]'] + fill_to_max(list(adm[1]), self.seq_len - 1))
            input_tokens.extend(
                ['[CLS]'] + fill_to_max(list(adm[2]), self.seq_len - 1))
            # output_rx_tokens.append(list(adm[1]))

            if idx != 0:
                output_rx_tokens.append(list(adm[1]))
                output_dx_tokens.append(list(adm[0]))
                output_sx_tokens.append(list(adm[2]))

        """convert tokens to id
        """
        input_ids = self.tokenizer.convert_tokens_to_ids(input_tokens)
        output_dx_labels = []  # (adm-1, dx_voc_size)
        output_rx_labels = []  # (adm-1, rx_voc_size)
        output_sx_labels = []  # (adm-1, rx_voc_size)

        dx_voc_size = len(self.tokenizer.dx_voc_multi.word2idx)
        rx_voc_size = len(self.tokenizer.rx_voc_multi.word2idx)
        sx_voc_size = len(self.tokenizer.sx_voc_multi.word2idx)
        for tokens in output_dx_tokens:
            tmp_labels = np.zeros(dx_voc_size)
            tmp_labels[list(
                map(lambda x: self.tokenizer.dx_voc_multi.word2idx[x], tokens))] = 1
            output_dx_labels.append(tmp_labels)

        for tokens in output_rx_tokens:
            tmp_labels = np.zeros(rx_voc_size)
            tmp_labels[list(
                map(lambda x: self.tokenizer.rx_voc_multi.word2idx[x], tokens))] = 1
            output_rx_labels.append(tmp_labels)
        for tokens in output_sx_tokens:
            tmp_labels = np.zeros(sx_voc_size)
            # tep = map(lambda x: self.tokenizer.sx_voc_multi.word2idx[x], tokens)
            tmp_labels[list(
                map(lambda x: self.tokenizer.sx_voc_multi.word2idx[x], tokens))] = 1
            output_sx_labels.append(tmp_labels)
        if cur_id < 5:
            logger.info("*** Example ***")
            logger.info("subject_id: %s" % subject_id)
            logger.info("input tokens: %s" % " ".join(
                [str(x) for x in input_tokens]))
            logger.info("input_ids: %s" %
                        " ".join([str(x) for x in input_ids]))

        assert len(input_ids) == (self.seq_len *
                                  3 * len(self.records[subject_id]))
        assert len(output_dx_labels) == (len(self.records[subject_id]) - 1)
        # assert len(output_rx_labels) == len(self.records[subject_id])-1

        cur_tensors = (torch.tensor(input_ids).view(-1, self.seq_len),
                       torch.tensor(output_dx_labels, dtype=torch.float),
                       torch.tensor(output_rx_labels, dtype=torch.float),
                       torch.tensor(output_sx_labels, dtype=torch.float))

        return cur_tensors


def load_dataset(args):
    data_dir = args.data_dir
    max_seq_len = args.max_seq_length

    # load tokenizer
    tokenizer = EHRTokenizer(data_dir)

    # load data
    data = pd.read_pickle(os.path.join(data_dir, 'data-multi-visit.pkl'))

    # load trian, eval, test data
    ids_file = [os.path.join(data_dir, 'train-id.txt'),
                os.path.join(data_dir, 'eval-id.txt'),
                os.path.join(data_dir, 'test-id.txt')]

    def load_ids(data, file_name):
        """
        :param data: multi-visit data
        :param file_name:
        :return: raw data form
        """
        ids = []
        with open(file_name, 'r') as f:
            for line in f:
                ids.append(line.rstrip('\n'))
        tmp = data[data['SUBJECT_ID'].isin(ids)].reset_index(drop=True)
        return tmp

    # tokenizer, tuple(map(lambda x: EHRDataset(load_ids(data, x), tokenizer, max_seq_len), ids_file))
    # return tokenizer, tuple(map(lambda x: EHRDataset(load_ids(data, x), tokenizer, max_seq_len), ids_file))
    return tokenizer, \
           (EHRDataset(load_ids(data, ids_file[0]), tokenizer, max_seq_len), \
           EHRDataset(load_ids(data, ids_file[1]), tokenizer, max_seq_len), \
           EHRDataset(load_ids(data, ids_file[2]), tokenizer, max_seq_len))

def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--model_name", default='GBert-predict-duikang_epoch=10', type=str, required=False,
                        help="model name")
    parser.add_argument("--data_dir",
                        default='../data',
                        type=str,
                        required=False,
                        help="The input data dir.")
    parser.add_argument("--pretrain_dir", default='../saved/GBert-pretraining-duikang_epoch=10', type=str, required=False,
                        help="pretraining model")
    parser.add_argument("--train_file", default='data-multi-visit.pkl', type=str, required=False,
                        help="training data file.")
    parser.add_argument("--output_dir",
                        default='../saved/',
                        type=str,
                        required=False,
                        help="The output directory where the model checkpoints will be written.")

    # Other parameters
    parser.add_argument("--use_pretrain",
                        default=True,
                        action='store_true',
                        help="is use pretrain")
    parser.add_argument("--graph",
                        default=True,
                        action='store_true',
                        help="if use ontology embedding")
    parser.add_argument("--therhold",
                        default=0.3,
                        type=float,
                        help="therhold.")
    parser.add_argument("--max_seq_length",
                        default=60,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        default=False,
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        default=True,
                        action='store_true',
                        help="Whether to run on the dev set.")
    parser.add_argument("--do_test",
                        default=True,
                        action='store_true',
                        help="Whether to run on the test set.")
    parser.add_argument("--train_batch_size",
                        default=16,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--learning_rate",
                        default=5e-4,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=1.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--no_cuda",
                        default=False,
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--seed',
                        type=int,
                        default=1203,
                        help="random seed for initialization")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")

    args = parser.parse_args()
    args.output_dir = os.path.join(args.output_dir, args.model_name)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available()
                          and not args.no_cuda else "cpu")

    if not args.do_train and not args.do_eval:
        raise ValueError(
            "At least one of `do_train` or `do_eval` must be True.")

    # if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
    #     raise ValueError(
    #         "Output directory ({}) already exists and is not empty.".format(args.output_dir))
    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading Dataset")
    tokenizer, (train_dataset, eval_dataset, test_dataset) = load_dataset(args)
    train_dataloader = DataLoader(train_dataset,
                                  sampler=RandomSampler(train_dataset),
                                  batch_size=1)
    eval_dataloader = DataLoader(eval_dataset,
                                 sampler=SequentialSampler(eval_dataset),
                                 batch_size=1)
    test_dataloader = DataLoader(test_dataset,
                                 sampler=SequentialSampler(test_dataset),
                                 batch_size=1)

    print('Loading Model: ' + args.model_name)
    # config = BertConfig(vocab_size_or_config_json_file=len(tokenizer.vocab.word2idx), side_len=train_dataset.side_len)
    # config.graph = args.graph
    # model = SeperateBertTransModel(config, tokenizer.dx_voc, tokenizer.rx_voc)
    if args.use_pretrain:
        logger.info("Use Pretraining model")
        model = GBERT_Predict.from_pretrained(
            args.pretrain_dir, tokenizer=tokenizer)
    else:
        config = BertConfig(
            vocab_size_or_config_json_file=len(tokenizer.vocab.word2idx))
        config.graph = args.graph
        model = GBERT_Predict(config, tokenizer)
    logger.info('# of model parameters: ' + str(get_n_params(model)))

    model.to(device)

    model_to_save = model.module if hasattr(
        model, 'module') else model  # Only save the model it-self
    rx_output_model_file = os.path.join(
        args.output_dir, "pytorch_model.bin")

    # Prepare optimizer
    # num_train_optimization_steps = int(
    #     len(train_dataset) / args.train_batch_size) * args.num_train_epochs
    # param_optimizer = list(model.named_parameters())
    # no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    # optimizer_grouped_parameters = [
    #     {'params': [p for n, p in param_optimizer if not any(
    #         nd in n for nd in no_decay)], 'weight_decay': 0.01},
    #     {'params': [p for n, p in param_optimizer if any(
    #         nd in n for nd in no_decay)], 'weight_decay': 0.0}
    # ]

    # optimizer = BertAdam(optimizer_grouped_parameters,
    #                      lr=args.learning_rate,
    #                      warmup=args.warmup_proportion,
    #                      t_total=num_train_optimization_steps)
    optimizer = Adam(model.parameters(), lr=args.learning_rate)

    global_step = 0
    if args.do_train:
        writer = SummaryWriter(args.output_dir)

        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_dataset))
        logger.info("  Batch size = %d", 1)

        dx_acc_best, rx_acc_best = 0, 0
        acc_name = 'prauc'
        dx_history = {'prauc': []}
        rx_history = {'prauc': []}

        fgm = FGM(model)
        loss_list=list()
        loss_list_adv = list()
        for _ in trange(int(args.num_train_epochs), desc="Epoch"):
            print('')
            tr_loss = 0
            tr_loss_adv=0
            nb_tr_examples, nb_tr_steps = 0, 0
            prog_iter = tqdm(train_dataloader, leave=False, desc='Training')
            model.train()
            for _, batch in enumerate(prog_iter):
                batch = tuple(t.to(device) for t in batch)
                input_ids, dx_labels, rx_labels,sx_labels = batch
                input_ids, dx_labels, rx_labels,sx_labels = input_ids.squeeze(
                    dim=0), dx_labels.squeeze(dim=0), rx_labels.squeeze(dim=0), sx_labels.squeeze(dim=0)

                loss, rx_logits = model(input_ids, dx_labels=dx_labels, rx_labels=rx_labels,sx_labels=sx_labels,
                                        epoch=global_step)
                loss.backward(retain_graph=True)
                # fgm.attack()  # 在embedding上添加对抗扰动
                #
                # loss_adv, rx_logits = model(input_ids, dx_labels=dx_labels, rx_labels=rx_labels, sx_labels=sx_labels,
                #                         epoch=global_step)
                #
                # loss_adv.backward()  # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
                # fgm.restore()  # 恢复embedding参数

                tr_loss += loss.item()
                # tr_loss_adv += loss_adv.item()
                nb_tr_examples += 1
                nb_tr_steps += 1

                # Display loss
                prog_iter.set_postfix(loss='%.4f' % (tr_loss / nb_tr_steps))
                loss_list.append('%.4f' % (tr_loss / nb_tr_steps))
                # loss_list_adv.append('%.4f' % (tr_loss_adv / nb_tr_steps))

                optimizer.step()
                optimizer.zero_grad()


            writer.add_scalar('train/loss', tr_loss / nb_tr_steps, global_step)
            # plt.plot(list(range(len(loss_list_adv))), loss_list_adv, color='red', label='loss_list_adv')
            # plt.plot(list(range(len(loss_list))), loss_list, color='green', label='loss_list')
            # # plt.plot(loss_list, loss_list_adv, linestyle='-', marker='o')
            # plt.legend()  # 显示图例
            # plt.xlabel('iteration times')
            # plt.ylabel('loss')
            # x_major_locator = MultipleLocator(20)
            # y_major_locator = MultipleLocator(0.01)
            # ax = plt.gca()
            # ax.xaxis.set_major_locator(x_major_locator)
            # ax.yaxis.set_major_locator(y_major_locator)
            # plt.xlim(0,1000)
            # plt.ylim(0, 1)
            # plt.show()
            global_step += 1

            text_save_data('duikang/train_loss_epoch=10_no-p.txt', ('%.4f' % (tr_loss / nb_tr_steps)))
            # text_save_data('wuduikang/txt_loss_wuduikang_epoch_nograph.txt',('%.4f' % (tr_loss / nb_tr_steps)))
            # text_save('duikang/txt_loss_adv_duikang.txt', loss_list_adv)

            if args.do_eval:
                print('')
                logger.info("***** Running eval *****")
                model.eval()
                dx_y_preds = []
                dx_y_trues = []
                rx_y_preds = []
                rx_y_trues = []
                for eval_input in tqdm(eval_dataloader, desc="Evaluating"):
                    eval_input = tuple(t.to(device) for t in eval_input)
                    input_ids, dx_labels, rx_labels,sx_labels = eval_input
                    input_ids, dx_labels, rx_labels,sx_labels = input_ids.squeeze(
                    ), dx_labels.squeeze(), rx_labels.squeeze(dim=0),sx_labels.squeeze(dim=0)
                    with torch.no_grad():
                        loss, rx_logits = model(
                            input_ids, dx_labels=dx_labels, rx_labels=rx_labels,sx_labels=sx_labels)
                        rx_y_preds.append(t2n(torch.sigmoid(rx_logits)))
                        rx_y_trues.append(t2n(rx_labels))
                        # dx_y_preds.append(t2n(torch.sigmoid(dx_logits)))
                        # dx_y_trues.append(
                        #     t2n(dx_labels.view(-1, len(tokenizer.dx_voc.word2idx))))
                        # rx_y_preds.append(t2n(torch.sigmoid(rx_logits))[
                        #                   :, tokenizer.rx_singe2multi])
                        # rx_y_trues.append(
                        #     t2n(rx_labels)[:, tokenizer.rx_singe2multi])

                print('')
                # dx_acc_container = metric_report(np.concatenate(dx_y_preds, axis=0), np.concatenate(dx_y_trues, axis=0),
                #                                  args.therhold)
                rx_acc_container = metric_report(np.concatenate(rx_y_preds, axis=0), np.concatenate(rx_y_trues, axis=0),
                                                 args.therhold)
                # jsObj = json.dumps(rx_acc_container)
                # fileObject = open('jsonFile_eval_1.json', 'a')
                # fileObject.write(jsObj)
                # fileObject.close()
                # f1_eval_list_run.append(rx_acc_container['f1'])
                for k, v in rx_acc_container.items():
                    writer.add_scalar(
                        'eval/{}'.format(k), v, global_step)

                if rx_acc_container[acc_name] > rx_acc_best:
                    rx_acc_best = rx_acc_container[acc_name]
                    # save model
                    torch.save(model_to_save.state_dict(),
                               rx_output_model_file)

        with open(os.path.join(args.output_dir, 'bert_config.json'), 'w', encoding='utf-8') as fout:
            fout.write(model.config.to_json_string())

    if args.do_test:
        logger.info("***** Running test *****")
        logger.info("  Num examples = %d", len(test_dataset))
        logger.info("  Batch size = %d", 1)

        def test(task=0):
            # Load a trained model that you have fine-tuned
            model_state_dict = torch.load(rx_output_model_file)
            model.load_state_dict(model_state_dict)
            model.to(device)

            model.eval()
            y_preds = []
            y_trues = []
            test_loss_1 = 0
            # tr_loss_adv = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            test_loss_list = list()
            for test_input in tqdm(test_dataloader, desc="Testing"):
                test_input = tuple(t.to(device) for t in test_input)
                input_ids, dx_labels, rx_labels, sx_labels = test_input
                input_ids, dx_labels, rx_labels, sx_labels = input_ids.squeeze(
                ), dx_labels.squeeze(), rx_labels.squeeze(dim=0), sx_labels.squeeze(dim=0)
                with torch.no_grad():
                    loss, rx_logits = model(
                        input_ids, dx_labels=dx_labels, rx_labels=rx_labels)
                    y_preds.append(t2n(torch.sigmoid(rx_logits)))
                    y_trues.append(t2n(rx_labels))

                    # test_loss=0
                    test_loss_1 += loss.item()
                    test_loss_2 = loss.item()
                    # print(test_loss)
                    nb_tr_examples += 1
                    nb_tr_steps += 1

                    # Display loss
                    # test_loss_list.append('%.4f' % (test_loss/nb_tr_steps))
                    # loss_list_adv.append(loss.item)

            print('')
            # text_save_data('wuduikang/test_loss_1_wuduikang_1.txt', test_loss_1/nb_tr_steps)
            text_save_data('duikang/test_loss_epoch=10_no-p.txt', test_loss_1/nb_tr_steps)
            # text_save_data('duikang/test_loss_2_duikang.txt', test_loss_2)
            acc_container = metric_report(np.concatenate(y_preds, axis=0), np.concatenate(y_trues, axis=0),
                                          args.therhold)
            # f1_test_list_run.append(rx_acc_container['f1'])
            jsObj = json.dumps(acc_container)
            # fileObject = open('duikang/jsonFile_test_1_duikang_1_nograph.json', 'a')
            fileObject = open('duikang/test_json_epoch=10_no-p.json', 'a')
            fileObject.write(jsObj)
            fileObject.close()
            print(acc_container)
            # _, ax1 = plt.subplots()
            # ax2 = ax1.twinx()
            # x=y_preds[0].tolist()
            # print(x)

            # plt.plot(y_preds[0].tolist(), y_trues[0].tolist(),linestyle='-',marker='o')
            # plt.axis([0,1,0,1])

            # ax2.plot(np.arange(2543), y_trues[0].tolist())
            # plt.set_xlabel('y_preds')
            # plt.set_ylabel('y_trues')
            # plt.show()
            # ax2.set_ylabel('y_trues')

            # fitlog.add_loss(loss, name="Loss")

            # save report
            if args.do_train:
                for k, v in acc_container.items():
                    writer.add_scalar(
                        'test/{}'.format(k), v, 0)

            return acc_container

        test(task=0)


if __name__ == "__main__":
    main()
