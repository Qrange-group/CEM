import os
import pickle as pkl
import sys
from tkinter import dialog
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

curdir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, curdir)
prodir = '..'
sys.path.insert(0, prodir)


class DataEnv:
 
    def __init__(self, mode='train', data_name='normal', use_pre_train=False, embed_size=200):
        """
        Constant variable declaration and configuration.
        """
        self.dialog_max_len = 64
        self.dialog_max_round = 50

        if data_name == 'clothes':
            dataset_folder_name = '/MHCH_SSA' + '/clothes'

        elif data_name == 'makeup':
            dataset_folder_name = '/MHCH_SSA' + '/makeup'

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

    def load_pkl_data(self, mode='train'):
        if mode == 'train' or mode == 'counterfactual':
            load_path = self.train_path
        elif mode == 'eval':
            load_path = self.val_path
        elif mode == 'test':
            load_path = self.test_path
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

                # counterfactual sampling
                if mode == 'counterfactual':
                    import pandas as pd
                    data = pd.DataFrame()
                    data['dialogues_len_list'] = self.dialogues_len_list
                    data['dialogues_ids_list'] = self.dialogues_ids_list
                    data['dialogues_sent_len_list'] = self.dialogues_sent_len_list
                    data['session_id_list'] = self.session_id_list
                    data['role_list'] = self.role_list
                    data['handoff_list'] = self.handoff_list
                    data['senti_list'] = self.senti_list
                    data['score_list'] = self.score_list

                    sample_num = int(len(self.dialogues_len_list)*0.8)
                    data = data.sort_values('dialogues_len_list')[sample_num:].reset_index(drop=True)
                    data = data[data['score_list'] == 0]

                    self.dialogues_ids_list = []
                    self.dialogues_len_list = []
                    self.dialogues_sent_len_list = []
                    self.session_id_list = []
                    self.senti_list = []
                    self.handoff_list = []
                    self.score_list = []
                    self.role_list = []
                    
                    self.dialogues_ids_list.extend(data['dialogues_ids_list'].values.tolist())
                    self.dialogues_sent_len_list.extend(data['dialogues_sent_len_list'].values.tolist())
                    self.dialogues_len_list.extend(data['dialogues_len_list'].values.tolist())
                    self.session_id_list.extend(data['session_id_list'].values.tolist())
                    self.role_list.extend(data['role_list'].values.tolist())
                    self.handoff_list.extend(data['handoff_list'].values.tolist())
                    self.senti_list.extend(data['senti_list'].values.tolist())
                    self.score_list.extend(data['score_list'].values.tolist())

                    for i in range(len(self.handoff_list)-sample_num, len(self.handoff_list)):
                        self.handoff_list[i] = self.handoff_list[i] * 0

            print("Load variable from {} successfully!".format(load_path))

    def reset(self, mode='train'):
        '''
        初始化数据
        '''
        self.load_pkl_data(mode=mode)
        data = {}
        data['dialogues_sent_len_list'] = []
        data['dialogues_len_list'] = self.dialogues_len_list
        data['s_lst'] = []        # 当前环境
        data['a_lst'] = []        # 在当前环境采取的action
        data['r_lst'] = []        # 当前环境采取的action带来的reward
        data['s_prime_lst'] = []  # 采取action之后下一步的环境变动
        data['done_lst'] = []     # 当前对话是否结束

        data_add = {}
        data_add['dialogues_sent_len_list'] = []
        data_add['dialogues_len_list'] = self.dialogues_len_list
        data_add['s_lst'] = []        # 当前环境
        data_add['a_lst'] = []        # 在当前环境采取的action
        data_add['r_lst'] = []        # 当前环境采取的action带来的reward
        data_add['s_prime_lst'] = []  # 采取action之后下一步的环境变动
        data_add['done_lst'] = []     # 当前对话是否结束
        s_len = 1 * len(self.dialogues_ids_list[0][0])

        for i in range(len(self.handoff_list)):
            # data['s_lst'].extend(self.dialogues_ids_list[i])
            # data['s_prime_lst'].extend(self.dialogues_ids_list[i][:-1])
            # data['s_prime_lst'].append(self.dialogues_ids_list[i][0])

            dlLen = len(self.handoff_list[i])
            dialog_bool = [[1]] * dlLen 
            dialog_bool[-1] = [0]
            dialog_bool_add = [[0]] * dlLen 
            s_all = []

            for j in range(dlLen):
                s_all.extend(self.dialogues_ids_list[i][j])
                tmp = s_all.copy()
                if s_len > len(s_all):
                    tmp = [0] * (s_len - len(s_all))
                    tmp.extend(s_all)

                data['s_lst'].append(tmp[-s_len:])
                data_add['s_lst'].append(tmp[-s_len:])
                if self.dialogues_sent_len_list[i][j] == 0:
                    self.dialogues_sent_len_list[i][j] = 1
                data['dialogues_sent_len_list'].append(self.dialogues_sent_len_list[i][j])
                data_add['dialogues_sent_len_list'].append(self.dialogues_sent_len_list[i][j])
                data['a_lst'].append([self.handoff_list[i][j]])
                

                if self.handoff_list[i][j] == 1:
                    dialog_bool[j] = [0]
                    data_add['a_lst'].append([0])
                    data_add['r_lst'].append([-1])
                else:
                    data_add['a_lst'].append([1])
                    data_add['r_lst'].append([-4])

                data['r_lst'].append([self.senti_list[i][j] + self.score_list[i] + dlLen/(j+1) + (1 if self.handoff_list[i][j]==0 else 0)])
            data['done_lst'].extend(dialog_bool)
            data_add['done_lst'].extend(dialog_bool_add)
        data['s_prime_lst'].extend(data['s_lst'][1:])
        data['s_prime_lst'].extend(data['s_lst'][-1:])
        data_add['s_prime_lst'].extend(data['s_lst'])

        if mode == 'train':
            for i in ['dialogues_sent_len_list', 'dialogues_len_list', 's_lst', 'a_lst', 'r_lst', 's_prime_lst', 'done_lst']:
                data[i].extend(data_add[i])


        print('Prepare %s data successfully!' % mode)

        return data



    