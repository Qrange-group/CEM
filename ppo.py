from cgi import print_environ
import itertools
from data_env import DataEnv
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch.autograd import Variable
from sklearn.metrics import f1_score, auc, roc_curve, confusion_matrix, classification_report, f1_score, precision_score, recall_score, accuracy_score
import numpy as np
import math
from torch.utils.tensorboard import SummaryWriter
import time
import pickle as pkl


log_dir = "./train_log/"+str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())).replace(' ', '_')
writer = SummaryWriter(log_dir=log_dir, comment='_scalars', filename_suffix="12345678")

#Hyperparameters
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
learning_rate = 0.0005
gamma         = 0.98
lmbda         = 0.95
eps_clip      = 0.1
K_epoch       = 3
T_horizon     = 20

class PPO(nn.Module):
    def __init__(self):
        super(PPO, self).__init__()
        self.data = []
        self.data = {}
        self.lamb = 0.
        self.time_step = 1
        self.batch_size = 1024
        self.vocab_size = 10054
        self.embedding_dim = 200
        self.sent_max_len = 64
        self.fc1   = nn.Linear(self.time_step * self.sent_max_len * self.embedding_dim, 256)
        self.fc_pi = nn.Linear(256, 2)
        self.fc_v  = nn.Linear(256, 1)
        self.word_embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        # nn.Softmax()
        self.vocab_save_path = '/home/user02/zss/robot/RSSN_CF/MHCH_SSA/makeup/vocab.pkl'
        with open(self.vocab_save_path, 'rb') as fin:
            self.word_embedding.weight.data.copy_(torch.from_numpy(pkl.load(fin).embeddings))

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def pi(self, x, softmax_dim = 0):
        prob = F.softmax(self.fc_pi(F.relu(self.fc1(x))), dim=softmax_dim)
        return prob
    
    def v(self, x):
        x = F.relu(self.fc1(x))
        v = self.fc_v(x)
        return v
      
    def put_train_data(self, data_env):
        self.train_data = data_env

    def put_test_data(self, data_env):
        self.test_data = data_env 

    def make_batch(self, mode='train'):
        if mode == 'train':
            self.data = self.train_data
        elif mode == 'test':
            self.data = self.test_data

        s_batch, a_batch, r_batch, s_prime_batch, done_batch = torch.tensor(self.data['s_lst'], dtype=torch.long), \
                                                               torch.tensor(self.data['a_lst']), \
                                                               torch.tensor(self.data['r_lst'], dtype=torch.float) / 100, \
                                                               torch.tensor(self.data['s_prime_lst'], dtype=torch.long), \
                                                               torch.tensor(self.data['done_lst'])
        s_batch = s_batch.to(device)
        s_prime_batch = s_prime_batch.to(device)
        done_batch = done_batch.to(device)
        a_batch = a_batch.to(device)
        r_batch = r_batch.to(device)
        # print((r_batch.data > 0 ).sum())
        # print((r_batch.data < 0).sum())
        # print(ii)

        return s_batch, a_batch, r_batch, s_prime_batch, done_batch
    
    def simple_evaluate_batch(self, mode='test'):
        s, a, r, s_prime, done = self.make_batch(mode=mode)
        data_size = len(s)
        batch_num = int(data_size / self.batch_size) + 1
        total_label = []
        total_pre_label = []

        for i in range(batch_num):
            start = self.batch_size * i
            end = self.batch_size * (i+1)
            r_batch = r[start: end]
            index = (r_batch >= 0).nonzero()
            if index.shape[0] != 0:
                s_batch = self.word_embedding(s[start: end])
                s_batch = torch.reshape(s_batch, shape=[-1, self.time_step * self.sent_max_len * self.embedding_dim])
                a_batch = a[start: end]
                index = index[:,0]
                s_batch = torch.index_select(s_batch, 0, index)
                a_batch = torch.index_select(a_batch, 0, index)
                pi = self.pi(s_batch, softmax_dim=1)
                _, pre_a = torch.max(pi, 1)

                total_label.append(a_batch.detach().cpu().numpy())
                total_pre_label.append(pre_a.detach().cpu().numpy())

        target = torch.tensor(np.concatenate(total_label)).squeeze()
        pred_choice = torch.tensor(np.concatenate(total_pre_label))

        TP = ((pred_choice == 1) & (target.data == 1)).sum()
        TN = ((pred_choice == 0) & (target.data == 0)).sum()
        FN = ((pred_choice == 0) & (target.data == 1)).sum()
        FP = ((pred_choice == 1) & (target.data == 0)).sum()
        p = TP / (TP + FP)
        r = TP / (TP + FN)
        F1 = 2 * r * p / (r + p)
        acc = (TP + TN) / (TP + TN + FP + FN)

        print(confusion_matrix(target, pred_choice))
        print("%s\tprecision: %f; recall: %f; F1: %f; accuracy: %f." % (mode, p, r, F1, acc))

    def train_net(self, ifTestTrain=False, ifTest=False):
        s, a, r, s_prime, done_mask = self.make_batch()
        
        data_size = len(s)
        batch_num = int(data_size / self.batch_size) + 1

        loss_sum = 0
        
        for j in range(batch_num):
            start = self.batch_size * j
            end = self.batch_size * (j+1)
            for i in range(K_epoch):
                self.optimizer.zero_grad()
                s_batch = self.word_embedding(s[start: end])
                s_batch = torch.reshape(s_batch, shape=[-1, self.time_step * self.sent_max_len * self.embedding_dim])

                s_prime_batch = self.word_embedding(s_prime[start: end])
                s_prime_batch = torch.reshape(s_prime_batch, shape=[-1, self.time_step * self.sent_max_len * self.embedding_dim])

                a_batch = a[start: end]
                r_batch = r[start: end]
                done_mask_batch = done_mask[start: end]
                # batch_packed = nn.utils.rnn.pack_padded_sequence(s_batch, self.data['dialogues_sent_len_list'][start: end], enforce_sorted=False)
                # output, hidden = self.gru(batch_packed, None)
                # output, lens = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
                # s_batch = torch.reshape(output, shape=[-1, self.sent_max_len * self.embedding_dim])
                
                pi = self.pi(s_batch, softmax_dim=1)
                prob_a = pi.gather(1, a[start: end])

                td_target = r_batch + gamma * self.v(s_prime_batch) * done_mask_batch
                delta = td_target - self.v(s_batch)
                delta = delta.detach().cpu().numpy()

                advantage_lst = []
                advantage = 0.0
                for delta_t in delta[::-1]:
                    advantage = gamma * lmbda * advantage + delta_t[0]
                    advantage_lst.append([advantage])
                advantage_lst.reverse()
                advantage = torch.tensor(advantage_lst, dtype=torch.float).to(device)
                pi = self.pi(s_batch, softmax_dim=1)
                pi_a = pi.gather(1, a_batch)
                
                ratio = torch.exp(torch.log(pi_a) - torch.log(prob_a))  # a/b == exp(log(a)-log(b))

                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1-eps_clip, 1+eps_clip) * advantage
                loss = F.smooth_l1_loss(self.v(s_batch), td_target.detach()) - torch.min(surr1, surr2) 

                # index = (r_batch >= 0).nonzero()
                # index = index[:,0]
                # s_batch = torch.index_select(s_batch, 0, index)
                # a_batch = torch.index_select(a_batch, 0, index)
                # pi = self.pi(s_batch, softmax_dim=1)
                # pi_a = pi.gather(1, a_batch)
                # # m = Categorical(pi)
                # # pre_a = m.sample()
                # a_batch = a_batch.float()
                # loss_classify = nn.BCELoss()(pi_a, a_batch)
                # loss += loss_classify

                loss.mean().backward()
                self.optimizer.step()

                loss_sum += loss.mean().item()
        
        if ifTestTrain:
            self.evaluate_batch(mode='train')     
        if ifTest:
            self.evaluate_batch()

        return loss_sum
    

    def evaluate_batch(self, mode='test'):
        s, a, r, s_prime, done = self.make_batch(mode=mode)
        data_size = len(s)
        batch_num = int(data_size / self.batch_size) + 1
        counter, total_loss = 0, 0.0
        total_label = []
        total_pre_label = []
        total_pre_scores = []

        for i in range(batch_num):
            start = self.batch_size * i
            end = self.batch_size * (i+1)

            s_batch = self.word_embedding(s[start: end])
            s_batch = torch.reshape(s_batch, shape=[-1, self.time_step * self.sent_max_len * self.embedding_dim])

            pi = self.pi(s_batch, softmax_dim=1)
            m = Categorical(pi)
            
            pre_a = m.sample()
            true_a = a[start: end]
            total_label.append(true_a.detach().cpu().numpy())
            total_pre_label.append(pre_a.detach().cpu().numpy())
            pre_s = pi.gather(1, a[start: end])
            total_pre_scores.append(pre_s.detach().cpu().numpy())
            counter += 1

        total_label_flat = np.concatenate(total_label)
        total_pre_label_flat = np.concatenate(total_pre_label)
        total_pre_scores_flat = np.concatenate(total_pre_scores)

        acc, p, r, f1, macro = self.get_a_p_r_f_sara(target=total_label_flat, predict=total_pre_label_flat)
        print(confusion_matrix(total_label_flat, total_pre_label_flat))
        gtt_1, gtt_2, gtt_3 = self.get_gtt_score(total_label, total_pre_label, lamb=self.lamb)

        # calc AUC score
        fpr, tpr, thresholds = roc_curve(total_label_flat, total_pre_scores_flat, pos_label=1)
        auc_score = auc(fpr, tpr)

        print(
            "F1Score:%.3f\tMacro_F1Score:%.3f\tAUC:%.3f\tGS-I:%.3f\tGS-II:%.3f\tGS-III:%.3f"
            % (f1, macro, auc_score, gtt_1, gtt_2, gtt_3))
    
    def get_a_p_r_f_sara(self, target, predict, category=1):
        # sklearn version
        accuracy = accuracy_score(target, predict)
        precision = precision_score(target, predict, average='macro')
        recall = recall_score(target, predict, average='macro')
        f1 = f1_score(target, predict)
        macro_f1_score = f1_score(target, predict, average='macro')
        return accuracy, precision, recall, f1, macro_f1_score
    
    def get_gtt_score(self, label_list, pre_list, lamb=0.):
        gtt_score_list_1 = []
        gtt_score_list_2 = []
        gtt_score_list_3 = []
        for pres, labels in zip(pre_list, label_list):
            gtt_score_list_1.append(self.golden_transfer_within_tolerance_exp(pres, labels, t=1, lamb=lamb))
            gtt_score_list_2.append(self.golden_transfer_within_tolerance_exp(pres, labels, t=2, lamb=lamb))
            gtt_score_list_3.append(self.golden_transfer_within_tolerance_exp(pres, labels, t=3, lamb=lamb))

        return np.mean(gtt_score_list_1), np.mean(gtt_score_list_2), np.mean(gtt_score_list_3)

    def golden_transfer_within_tolerance_exp(self, pre_labels, true_labels, t=1, eps=1e-7, lamb=0):
        if t <= 0:
            raise ValueError("Tolerance must be positive!!!")
        if not isinstance(t, int):
            raise TypeError("Tolerance must be Integer!!!")

        gtt_score = 0
        suggest_indices = []
        for idx, label in enumerate(true_labels):
            if label == 1:
                suggest_indices.append(idx)
        
        pre_indices = []
        for idx, label in enumerate(pre_labels):
            if label == 1:
                pre_indices.append(idx)

        if len(suggest_indices) == 0:
            if len(pre_indices) == 0:
                gtt_score = 1
            else:
                gtt_score = 0
        else:
            if len(pre_indices) == 0:
                gtt_score = 0
            else:
                GST_score_list = []
                for pre_idx in pre_indices:
                    tmp_score_list = []
                    for suggest_idx in suggest_indices:
                        # suggest_idx is q_i
                        # pre_idx is p_i
                        pre_bias = pre_idx -suggest_idx
                        adjustment_cofficient = 1. / (1 - lamb * (np.sign(pre_bias)))
                        tmp_score = math.exp(- (adjustment_cofficient) * math.pow(pre_bias, 2)/ (2 * math.pow( (t + eps), 2)))
                        tmp_score_list.append(tmp_score)
                    GST_score_list.append(np.max(tmp_score_list))
                # print(punishment_ratio)
                gtt_score = np.mean(GST_score_list)
        return gtt_score


def main():
    model = PPO().to(device)
    data = DataEnv(data_name='makeup').reset(mode='train')
    model.put_train_data(data)
    data = DataEnv(data_name='makeup').reset(mode='test')
    model.put_test_data(data)
    for n_epi in range(10000):
        loss = model.train_net(ifTestTrain = True, ifTest = True)
        writer.add_scalar('loss', loss, n_epi)
        print("epoch: ", n_epi, '\tloss: ', loss)

if __name__ == '__main__':
    main()
