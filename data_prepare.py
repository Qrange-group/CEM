#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
File function description.

@Time    : 2020/4/13 3:24 下午
@Author  : Weijia.ljw
@File    : data_loader.py
@Software: PyCharm
"""

import os
import sys
import json
import argparse
import pickle as pkl
import jieba
import random
import copy
import logging
from tqdm import tqdm
import numpy as np

curdir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, curdir)
prodir = '..'
sys.path.insert(0, prodir)
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)
from vocab import Vocab
from utility import get_now_time
from keras.preprocessing.sequence import pad_sequences


class DataPrepare(object):
    def __init__(self, mode='train', data_name='normal', use_pre_train=True, embed_size=200):
        """
        Constant variable declaration and configuration.
        """
        self.dialog_max_len = 64
        self.dialog_max_round = 50

        if data_name == 'clothes':
            dataset_folder_name = '/data/MHCH_SSA' + '/clothes'
            self.train_raw_path = curdir + dataset_folder_name + '/train_aux.json'
            self.val_raw_path = curdir + dataset_folder_name + '/dev_aux.json'
            self.test_raw_path = curdir + dataset_folder_name + '/test_aux.json'

            self.all_raw_path = curdir + dataset_folder_name + '/all_aux.json'
            self.pre_train_embeddings_path = curdir + dataset_folder_name + '/clothes_w2v.cbow.200d'

        elif data_name == 'makeup':
            dataset_folder_name = '/data/MHCH_SSA' + '/makeup'
            self.train_raw_path = curdir + dataset_folder_name + '/train_aux.json'
            self.val_raw_path = curdir + dataset_folder_name + '/dev_aux.json'
            self.test_raw_path = curdir + dataset_folder_name + '/test_aux.json'

            self.all_raw_path = curdir + dataset_folder_name + '/all_aux.json'
            self.pre_train_embeddings_path = curdir + dataset_folder_name + '/makeup_w2v.cbow.200d'

        else:
            raise ValueError("Please confirm the correct data mode you entered.")

        self.vocab_save_path = curdir + dataset_folder_name + '/vocab.pkl'
        self.train_path = curdir + dataset_folder_name + '/train.pkl'
        self.val_path = curdir + dataset_folder_name + '/eval.pkl'
        self.test_path = curdir + dataset_folder_name + '/test.pkl'

        self.use_pre_train = use_pre_train
        self.embed_size = embed_size

        self.dialogues_list = []
        self.role_list = []
        self.contents_list = []
        self.dialogues_ids_list = []
        self.dialogues_len_list = []
        self.dialogues_sent_len_list = []
        self.session_id_list = []

        self.senti_list = []
        self.handoff_list = []
        self.score_list = []

        self.mode = mode

    def _load_raw_data(self, mode):
        """
        Check data, create the directories, prepare for the vocabulary and embeddings.
        """
        if mode == 'vocab':
            data_path = self.all_raw_path
        elif mode == 'train':
            data_path = self.train_raw_path
        elif mode == 'eval':
            data_path = self.val_raw_path
        elif mode == 'test':
            data_path = self.test_raw_path
        else:
            raise ValueError("{} mode not exists, please check it.".format(mode))

        if not os.path.exists(data_path):
            now_time = get_now_time()
            raise ValueError("{}: File {} is not exist.".format(now_time, data_path))

        with open(data_path, 'r', encoding='utf-8', newline='\n') as fin:
            print("Open {} successfully.".format(data_path))

            error_num, count = 0, 0
            while True:
                try:
                    line = fin.readline().replace('\n', '')
                    count += 1
                except IOError as e:
                    print(e)
                    error_num += 1
                    continue
                if not line:
                    print("Load data successfully!")
                    break
                try:
                    json_obj = json.loads(line)

                    self.dialogues_list.append(json_obj["session"])

                    self.score_list.append(int(json_obj["score"]) - 1)
                    self.session_id_list.append(json_obj["sessionID"])

                except ValueError as e:
                    error_num += 1
                    print("error line of json format: {}".format(line))

    def word_iter(self):
        """
        Iterates over all the words in dialogue content.
        :return: a generator
        """

        if self.dialogues_list is not None:
            vocab_used_dialogs = []
            for tmp_dialog in self.dialogues_list:
                vocab_used_dialogs.append(tmp_dialog)

            dialog_list = []
            for dialogue in vocab_used_dialogs:
                tmp_content_list = []
                for one_turn in dialogue:
                    tmp_content_list.append(one_turn["content"])
                dialog_list.append(' '.join(tmp_content_list))
            for tmp_dial in dialog_list:
                for token in jieba.cut(tmp_dial):
                    if token != '':
                        yield token
        else:
            raise ValueError("Get a empty dialogues_list.")

    def _load_vocab(self, vocab_path):
        """
        If we already have preprocessed vocabulary object, load it. Or gen a vocab object using gen_vocab()
        :param vocab_path:
        :return:
        """
        if os.path.exists(vocab_path):
            with open(vocab_path, 'rb') as fin:
                vocab = pkl.load(fin)
                return vocab
        else:
            raise ValueError("Get a empty Vocabulary.")

    def convert2ids(self):
        """
        Convert the utternaces in the original dialogues dataset to ids.
        :return: None
        """
        vocab = self._load_vocab(self.vocab_save_path)
        if self.dialogues_list is not None:
            for dialogue in self.dialogues_list:
                dialogue_ids = []
                content_list = []
                role_ids = []
                tmp_sent_len_list = []
                tmp_handoff_list = []
                tmp_senti_list = []

                for one_turn in dialogue:
                    content_list.append(one_turn["content"])
                    tmp_ids = vocab.convert2ids(jieba.cut(one_turn["content"]))

                    tmp_role = 0 if one_turn["role"] == "b" else 1

                    dialogue_ids.append(tmp_ids)
                    role_ids.append(tmp_role)
                    if len(tmp_ids) < self.dialog_max_len:
                        tmp_sent_len_list.append(len(tmp_ids))
                    else:
                        tmp_sent_len_list.append(self.dialog_max_len)
                    tmp_handoff_list.append(int(one_turn["label"]))
                    if int(one_turn["senti"]) == 3:
                        tmp_senti_list.append(1)
                    elif int(one_turn["senti"]) < 3:
                        tmp_senti_list.append(0)
                    else:
                        tmp_senti_list.append(2)

                self.contents_list.append(content_list)
                self.dialogues_sent_len_list.append(tmp_sent_len_list)
                if len(tmp_sent_len_list) < self.dialog_max_round:
                    self.dialogues_len_list.append(len(tmp_sent_len_list))
                else:
                    self.dialogues_len_list.append(self.dialog_max_round)
                self.dialogues_ids_list.append(
                    pad_sequences(dialogue_ids, maxlen=self.dialog_max_len, padding='post', truncating='post'))

                self.role_list.append(role_ids)
                self.handoff_list.append(tmp_handoff_list)
                self.senti_list.append(tmp_senti_list)

        print("Transform all data {} to id successfully!".format(len(self.dialogues_len_list)))

    def gen_vocab(self, min_cnt=2):
        """
        Utilizing the corpus to gen vocabulary and save to pickle.
        :return: None
        """
        self._load_raw_data(mode='vocab')
        vocab = Vocab(lower=True)
        for word in self.word_iter():
            vocab.add(word)

        # filter by cnt
        unfiltered_vocab_size = vocab.size()
        vocab.filter_tokens_by_cnt(min_cnt=min_cnt)
        filtered_num = unfiltered_vocab_size - vocab.size()
        print('After filter {} tokens, the final vocab size is {}'.format(filtered_num, vocab.size()))

        print('Assigning embedding ...')
        if self.use_pre_train:
            print('Pre trained')
            vocab.load_pretrained_embeddings(self.pre_train_embeddings_path)
        else:
            print('Random')
            vocab.randomly_init_embeddings(self.embed_size)

        print('Saving vocab ...')
        print('vocab size is: {}'.format(vocab.size()))
        with open(self.vocab_save_path, 'wb') as fout:
            pkl.dump(vocab, fout)
        print('Done with vocab!')
        return vocab

    def desensitization(self):
        with open(self.vocab_save_path, 'rb') as fin:
            vocab = pkl.load(fin)
        vocab.desensitization()
        with open(self.vocab_save_path, 'wb') as fout:
            pkl.dump(vocab, fout)
        print('Done with desensitization!')

    def save_data(self, mode='train'):
        """
        Save the transformed data to pickle.
        :param mode: str  train/val/test
        :return: None
        """
        if mode == 'train':
            load_path = self.train_path
        elif mode == 'eval':
            load_path = self.val_path
        elif mode == 'test':
            load_path = self.test_path
        else:
            raise ValueError("{} mode not exists, please check it.".format(mode))
        self._load_raw_data(mode=mode)
        self.convert2ids()

        with open(load_path, 'wb') as fout:
            # x
            pkl.dump(self.dialogues_ids_list, fout)
            pkl.dump(self.dialogues_sent_len_list, fout)
            pkl.dump(self.dialogues_len_list, fout)
            pkl.dump(self.session_id_list, fout)
            pkl.dump(self.role_list, fout)

            # main y
            pkl.dump(self.handoff_list, fout)
            # auxiliary y
            pkl.dump(self.senti_list, fout)
            pkl.dump(self.score_list, fout)

        print("Save variable into {}".format(load_path))

    def load_pkl_data(self, mode='train'):
        if mode == 'train':
            load_path = self.train_path
        elif mode == 'eval':
            load_path = self.val_path
        elif mode == 'test':
            load_path = self.test_path
        elif mode == 'predict':
            load_path = self.predict_path
        else:
            raise ValueError("{} mode not exists, please check it.".format(mode))

        if not os.path.exists(load_path):
            raise ValueError("{} not exists, please generate it firstly.".format(load_path))
        else:
            with open(load_path, 'rb') as fin:
                # X
                self.dialogues_ids_list = pkl.load(fin)
                self.dialogues_sent_len_list = pkl.load(fin)
                self.dialogues_len_list = pkl.load(fin)
                self.session_id_list = pkl.load(fin)
                self.role_list = pkl.load(fin)

                # main y
                self.handoff_list = pkl.load(fin)
                # auxiliary y
                self.senti_list = pkl.load(fin)
                self.score_list = pkl.load(fin)

            print("Load variable from {} successfully!".format(load_path))

    def load_config(self, config_path):
        with open(config_path, 'r') as fp:
            return json.load(fp)


if __name__ == "__main__":
    mode_list = ['train', 'eval', 'test']
    data_name_list = ['clothes', 'makeup']

    for data_name in data_name_list:
        data_prepare = DataPrepare(mode='vocab',
                                   data_name=data_name)
        data_prepare.gen_vocab(min_cnt=2)

    for data_name in data_name_list:
        for mode in mode_list:
            data_prepare = DataPrepare(mode=mode,
                                       data_name=data_name)
            data_prepare.save_data(mode=mode)

        data_prepare.desensitization()