import pickle as pkl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def read(data_name, mode):
    """
    clothes train
    长度都是8000
    """
    if data_name == "clothes":
        dataset_folder_name = "./MHCH_SSA" + "/clothes"
    elif data_name == "makeup":
        dataset_folder_name = "./MHCH_SSA" + "/makeup"

    vocab_save_path = dataset_folder_name + "/vocab.pkl"
    train_path = dataset_folder_name + "/train.pkl"
    val_path = dataset_folder_name + "/eval.pkl"
    test_path = dataset_folder_name + "/test.pkl"

    if mode == "train":
        load_path = train_path
    elif mode == "eval":
        load_path = val_path
    elif mode == "test":
        load_path = test_path

    with open(load_path, "rb") as fin:
        # X
        dialogues_ids_list = pkl.load(
            fin
        )  # (?, dia_len, sent_len) (?, ?, 64) 对话长度最长为128. 每一行表示一个对话，一个对话用多条list表示，每一条list表示对话中的一句话
        dialogues_sent_len_list = pkl.load(
            fin
        )  # 记录一个对话的每句话长度，一个对话用一条list表示，list长度与dialogues_ids_list的list数量对应，每一个数值表示对应的dialogues_ids_list的list的非0数目
        dialogues_len_list = pkl.load(fin)  # 一个对话用一个数值表示，表示对话的总对话数
        session_id_list = pkl.load(fin)  # 一个对话用一个字符串表示，如train_0
        role_list = pkl.load(fin)  # 表示对话对应的角色，用0/1表示

        # main y
        handoff_list = pkl.load(fin)  # 表示是否更换人工，用0/1表示
        # auxiliary y
        senti_list = pkl.load(fin)  # 当前语句对应的情绪，用0/1/2的list表示
        score_list = pkl.load(fin)  # 当前对话的总体满意度，用0/1/2表示

        # print
        # info = [dialogues_ids_list, dialogues_sent_len_list, dialogues_len_list, session_id_list, role_list,
        #     handoff_list, senti_list, score_list
        # ]
        # for i in info:
        #     print(i[0])


def create_cf_data(data_name, mode):

    if data_name == "clothes":
        dataset_folder_name = "./MHCH_SSA" + "/clothes"
    elif data_name == "makeup":
        dataset_folder_name = "./MHCH_SSA" + "/makeup"

    vocab_save_path = dataset_folder_name + "/vocab.pkl"
    train_path = dataset_folder_name + "/train.pkl"
    val_path = dataset_folder_name + "/eval.pkl"
    test_path = dataset_folder_name + "/test.pkl"

    if mode == "train":
        load_path = train_path
    elif mode == "eval":
        load_path = val_path
    elif mode == "test":
        load_path = test_path
    with open(load_path, "rb") as fin:
        # X
        dialogues_ids_list = pkl.load(fin)  # 每一行表示一个对话，一个对话用多条list表示，每一条list表示对话中的一句话
        dialogues_sent_len_list = pkl.load(
            fin
        )  # 记录一个对话的每句话长度，一个对话用一条list表示，list长度与dialogues_ids_list的list数量对应，每一个数值表示对应的dialogues_ids_list的list的非0数目
        dialogues_len_list = pkl.load(fin)  # 一个对话用一个数值表示，表示对话的总对话数
        session_id_list = pkl.load(fin)  # 一个对话用一个字符串表示，如train_0
        role_list = pkl.load(fin)  # 表示对话对应的角色，用0/1表示

        # main y
        handoff_list = pkl.load(fin)  # 表示是否更换人工，用0/1表示
        # auxiliary y
        senti_list = pkl.load(fin)  # 当前语句对应的情绪，用0/1/2的list表示
        score_list = pkl.load(fin)  # 当前对话的总体满意度，用0/1/2表示

        # info = [dialogues_ids_list, dialogues_sent_len_list, dialogues_len_list, session_id_list, role_list,
        #     handoff_list, senti_list, score_list
        # ]
        # for i in info:
        #     print(i[0])
        # 时间：dialogues_len_list大的 20%
        # 成本：handoff_list为1的
        # 情感：senti_list为0的
        data = pd.DataFrame()
        data["handoff_list"] = handoff_list

        data["dialogues_len_list"] = dialogues_len_list
        data["dialogues_ids_list"] = dialogues_ids_list
        data["dialogues_sent_len_list"] = dialogues_sent_len_list
        data["session_id_list"] = session_id_list
        data["role_list"] = role_list

        data["senti_list"] = senti_list
        data["score_list"] = score_list

        data = data.sort_values("dialogues_len_list")[
            int(len(dialogues_len_list) * 0.8) :
        ].reset_index(drop=True)
        data = data[data["score_list"] == 0]

        dialogues_ids_list.extend(data["dialogues_ids_list"].values.tolist())
        dialogues_sent_len_list.extend(data["dialogues_sent_len_list"].values.tolist())
        dialogues_len_list.extend(data["dialogues_len_list"].values.tolist())
        session_id_list.extend(data["session_id_list"].values.tolist())
        role_list.extend(data["role_list"].values.tolist())
        handoff_list.extend(data["handoff_list"].values.tolist())
        senti_list.extend(data["senti_list"].values.tolist())
        score_list.extend(data["score_list"].values.tolist())


def analysis(data_name="clothes", mode="train"):

    if data_name == "clothes":
        dataset_folder_name = "./MHCH_SSA" + "/clothes"
    elif data_name == "makeup":
        dataset_folder_name = "./MHCH_SSA" + "/makeup"

    vocab_save_path = dataset_folder_name + "/vocab.pkl"
    train_path = dataset_folder_name + "/train.pkl"
    val_path = dataset_folder_name + "/eval.pkl"
    test_path = dataset_folder_name + "/test.pkl"

    if mode == "train":
        load_path = train_path
    elif mode == "eval":
        load_path = val_path
    elif mode == "test":
        load_path = test_path
    with open(load_path, "rb") as fin:
        # X
        dialogues_ids_list = pkl.load(fin)  # 每一行表示一个对话，一个对话用多条list表示，每一条list表示对话中的一句话
        dialogues_sent_len_list = pkl.load(
            fin
        )  # 记录一个对话的每句话长度，一个对话用一条list表示，list长度与dialogues_ids_list的list数量对应，每一个数值表示对应的dialogues_ids_list的list的非0数目
        dialogues_len_list = pkl.load(fin)  # 一个对话用一个数值表示，表示对话的总对话数
        session_id_list = pkl.load(fin)  # 一个对话用一个字符串表示，如train_0
        role_list = pkl.load(fin)  # 表示对话对应的角色，用0/1表示。查阅代码认为1是用户

        # main y
        handoff_list = pkl.load(fin)  # 表示是否更换人工，用0/1表示
        # auxiliary y
        senti_list = pkl.load(fin)  # 当前语句对应的情绪，用0/1/2的list表示
        score_list = pkl.load(fin)  # 当前对话的总体满意度，用0/1/2表示

        dialogues_sent_len = []
        handoff = []

        for i in range(len(dialogues_sent_len_list)):
            dialogues_sent_len.extend(dialogues_sent_len_list[i])
            handoff.extend(handoff_list[i])

        """
        分析senti_list与handoff_list是否相关
        [0, 0, 2, 2, 0, 2, 0, 0, 2, 0, 0, 2]
        [1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0]
        senti_list和handoff_list为list，元素类型也为list，元素不等长，最长为128
        """
        plt.figure(figsize=(10, 6))

        sentences_handoff = []
        sentences_senti = []
        sentences_role = []
        sentences_score = []
        sentences_location = []
        sentences_len = []
        minLen = 2
        maxLen = 128
        # for i in range(len(senti_list)):
        #     # if 1 in handoff_list[i]:
        #         sentences_score.append(score_list[i])
        # df = pd.DataFrame()
        # df['score'] = sentences_score
        # df['scor1e'] = sentences_score

        # print(df.groupby('score').count())
        #         index = handoff_list[i].index(1)
        #         length = len(senti_list[i]) - index - 1
        #         sentences_senti.extend(senti_list[i][index+1:])
        #         sentences_handoff.extend(handoff_list[i][index+1:])
        #         sentences_role.extend(role_list[i][index+1:])
        #         sentences_score.extend([score_list[i]] * length)
        #         sentences_location.extend(list(range(length)))
        #         sentences_len.extend([length] * length)

        df = pd.DataFrame()
        # df['handoff'] = sentences_handoff
        # df['senti'] = sentences_senti
        # df['role'] = sentences_role
        # df['score'] = sentences_score
        # df['location'] = sentences_location
        # df['len'] = sentences_len
        # df['loc_rate'] = df['location'] / df['len']

        df["dialogues_sent_len"] = dialogues_sent_len
        df["handoff"] = handoff
        df = df.sort_values("dialogues_sent_len")
        length = int(df.shape[0] * 0.3)
        print(df[:length]["handoff"].sum() / length)
        print(df[length:]["handoff"].sum() / (df.shape[0] - length))
        # length = len(df.groupby('dialogues_len_list'))
        # for j, i in df.groupby('dialogues_len_list'):
        #     i = i.groupby('score_list').count()
        #     i = i/i.sum()
        #     if (j * 2 > length):
        #         plt.plot(i, 'r', label='length: '+str(j))
        #     else:
        #         plt.plot(i, 'b', label='length: '+str(j))

        # plt.plot(df[1], '*', label='senti')
        # plt.legend()
        # plt.xlabel('全局满意度')
        # plt.ylabel('在当前对话长度中的占比')
        # plt.savefig('./pts/tmp.png')


# read(data_name='makeup', mode='test')
analysis(data_name="clothes", mode="train")
