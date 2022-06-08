#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os
import sys
import json
import argparse
import pickle as pkl
import jieba
import random
import copy
from tqdm import tqdm
import numpy as np

curdir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, curdir)
prodir = ".."
sys.path.insert(0, prodir)
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
from utility import get_now_time
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical


class DataLoader(object):
    def __init__(
        self, task="train", data="normal", embed_size=200
    ):
        """
        Constant variable declaration and configuration.
        """
        self.dialog_max_len = 64
        self.dialog_max_round = 50

        if data == "clothes":
            dataset_folder_name = "/data" + "/clothes"

        elif data == "makeup":
            dataset_folder_name = "/data" + "/makeup"

        else:
            raise ValueError("Please confirm the correct data task you entered.")

        self.vocab_save_path = curdir + dataset_folder_name + "/vocab.pkl"
        self.train_path = curdir + dataset_folder_name + "/train.pkl"
        self.val_path = curdir + dataset_folder_name + "/eval.pkl"
        self.test_path = curdir + dataset_folder_name + "/test.pkl"
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

        self.task = task

    def load_pkl_data(self, task="train"):
        if task == "train" or task == "counterfactual":
            load_path = self.train_path
        elif task == "eval":
            load_path = self.val_path
        elif task == "test":
            load_path = self.test_path
        else:
            raise ValueError("{} task not exists, please check it.".format(task))

        if not os.path.exists(load_path):
            raise ValueError(
                "{} not exists, please generate it firstly.".format(load_path)
            )
        else:
            with open(load_path, "rb") as fin:
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

                # counterfactual sampling
                if task == "counterfactual":
                    import pandas as pd

                    data = pd.DataFrame()
                    data["dialogues_len_list"] = self.dialogues_len_list
                    data["dialogues_ids_list"] = self.dialogues_ids_list
                    data["dialogues_sent_len_list"] = self.dialogues_sent_len_list
                    data["session_id_list"] = self.session_id_list
                    data["role_list"] = self.role_list
                    data["handoff_list"] = self.handoff_list
                    data["senti_list"] = self.senti_list
                    data["score_list"] = self.score_list

                    sample_num = int(len(self.dialogues_len_list) * 0.8)
                    # select top 20% of dialogue length and score is negative
                    data = data.sort_values("dialogues_len_list")[
                        sample_num:
                    ].reset_index(drop=True)
                    data = data[data["score_list"] == 0]

                    self.dialogues_ids_list = []
                    self.dialogues_len_list = []
                    self.dialogues_sent_len_list = []
                    self.session_id_list = []
                    self.senti_list = []
                    self.handoff_list = []
                    self.score_list = []
                    self.role_list = []

                    self.dialogues_ids_list.extend(
                        data["dialogues_ids_list"].values.tolist()
                    )
                    self.dialogues_sent_len_list.extend(
                        data["dialogues_sent_len_list"].values.tolist()
                    )
                    self.dialogues_len_list.extend(
                        data["dialogues_len_list"].values.tolist()
                    )
                    self.session_id_list.extend(data["session_id_list"].values.tolist())
                    self.role_list.extend(data["role_list"].values.tolist())
                    self.handoff_list.extend(data["handoff_list"].values.tolist())
                    self.senti_list.extend(data["senti_list"].values.tolist())
                    self.score_list.extend(data["score_list"].values.tolist())

                    # make this part's label be chatbot
                    for i in range(len(self.handoff_list)):
                        list_pack = list(
                            zip(
                                self.dialogues_ids_list[i],
                                self.dialogues_sent_len_list[i],
                                self.role_list[i],
                                self.handoff_list[i],
                                self.senti_list[i],
                            )
                        )
                        random.shuffle(list_pack)
                        (
                            self.dialogues_ids_list[i],
                            self.dialogues_sent_len_list[i],
                            self.role_list[i],
                            self.handoff_list[i],
                            self.senti_list[i],
                        ) = zip(*list_pack)

            print("Load variable from {} successfully!".format(load_path))

    @staticmethod
    def load_config(config_path):
        with open(config_path, "r") as fp:
            return json.load(fp)

    def data_generator(
        self, task="train", batch_size=32, shuffle=True, nb_classes=3, epoch=0
    ):
        print("Using data_generator")
        self.load_pkl_data(task=task)
        # X
        dialog_ids_list = self.dialogues_ids_list
        sent_len_list = self.dialogues_sent_len_list
        dial_len_list = self.dialogues_len_list
        sess_id_list = self.session_id_list
        role_list = self.role_list

        # main y
        handoff_list = self.handoff_list
        # aux y
        senti_list = self.senti_list
        score_list = self.score_list

        if shuffle:
            list_pack = list(
                zip(
                    dialog_ids_list,
                    sent_len_list,
                    dial_len_list,
                    sess_id_list,
                    role_list,
                    handoff_list,
                    senti_list,
                    score_list,
                )
            )
            random.seed(epoch + 7)
            random.shuffle(list_pack)
            (
                dialog_ids_list[:],
                sent_len_list[:],
                dial_len_list[:],
                sess_id_list[:],
                role_list[:],
                handoff_list[:],
                senti_list[:],
                score_list[:],
            ) = zip(*list_pack)

        for i in tqdm(range(0, len(score_list), batch_size), desc="Processing:"):
            batch_dialog_ids = pad_sequences(
                dialog_ids_list[i : i + batch_size],
                maxlen=self.dialog_max_round,
                padding="post",
                truncating="post",
                dtype="float32",
            )
            batch_sent_len = pad_sequences(
                sent_len_list[i : i + batch_size],
                maxlen=self.dialog_max_round,
                padding="post",
                truncating="post",
                dtype="int32",
            )
            batch_dia_len = dial_len_list[i : i + batch_size]
            batch_ids = sess_id_list[i : i + batch_size]
            batch_role_ids = pad_sequences(
                role_list[i : i + batch_size],
                maxlen=self.dialog_max_round,
                padding="post",
                truncating="post",
                dtype="int32",
            )

            handoff_padded = pad_sequences(
                handoff_list[i : i + batch_size],
                maxlen=self.dialog_max_round,
                padding="post",
                truncating="post",
                dtype="int32",
                value=0,
            )
            batch_handoff = to_categorical(handoff_padded, 2, dtype="int32")
            senti_padded = pad_sequences(
                senti_list[i : i + batch_size],
                maxlen=self.dialog_max_round,
                padding="post",
                truncating="post",
                dtype="int32",
                value=0,
            )
            batch_senti = to_categorical(senti_padded, 3, dtype="int32")
            batch_score = to_categorical(
                score_list[i : i + batch_size], 3, dtype="int32"
            )

            yield batch_dialog_ids, batch_sent_len, batch_dia_len, batch_ids, batch_role_ids, batch_handoff, batch_senti, batch_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Params")
    parser.add_argument(
        "--phase",
        default="load_data",
        type=str,
        help="phase: What preprocessing action you want take.",
    )
    parser.add_argument(
        "--min_cnt",
        default=2,
        type=int,
        help="min_cnt: filter the word which frequency less than min_cnt.",
    )
    parser.add_argument(
        "--task",
        default="eval",
        type=str,
        help="task: select the train/eval/test task to process.",
    )
    parser.add_argument(
        "--data",
        default="clothes",
        type=str,
        help="data: using which dataset.",
    )
    parser.add_argument(
        "--form",
        default="pkl",
        type=str,
        help="form: save the split data into what file type.",
    )
    args = parser.parse_args()

    data_loader = DataLoader(
        task=args.task, data=args.data
    )

    if args.phase == "test_load":
        pass
    elif args.phase == "load_data":
        data_loader.load_pkl_data(task=args.task)
        count = 0
        for (
            batch_dialog_ids,
            batch_sent_len,
            batch_dia_len,
            batch_ids,
            batch_role_ids,
            batch_handoff,
            batch_senti,
            batch_score,
        ) in data_loader.data_generator(task=args.task):
            # print(batch_dialog_ids.shape)  # (32, 50, 64)
            # print(batch_sent_len.shape)    # (32, 50)
            # print(batch_dia_len)           # list, length is 32, content is int
            # print(batch_ids)               # list, length is 32, content is string, for example 'dev_359'
            # print(batch_role_ids.shape)    # (32, 50)
            # print(batch_handoff.shape)     # (32, 50, 2)
            # print(batch_senti.shape)       # (32, 50, 3)
            # print(batch_score.shape)       # (32, 3)

            if count > 2:
                break
    else:
        now_time = get_now_time()
        raise ValueError(
            "{}: Please check whether '{}' is the action you want take.".format(
                now_time, args.phase
            )
        )
