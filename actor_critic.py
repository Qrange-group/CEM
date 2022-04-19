from data_env import DataEnv
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from sklearn.metrics import f1_score, auc, roc_curve, confusion_matrix, classification_report, f1_score, precision_score, recall_score, accuracy_score
import numpy as np
import math

#Hyperparameters
learning_rate = 0.0002
gamma         = 0.98
n_rollout     = 10

class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()
        self.data = {}
        self.lamb = 0.
        self.batch_size = 1024
        self.fc1 = nn.Linear(64,256)
        self.fc_pi = nn.Linear(256,2)
        self.fc_v = nn.Linear(256,1)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        
    def pi(self, x, softmax_dim = 0):
        x = F.relu(self.fc1(x))
        x = self.fc_pi(x)
        prob = F.softmax(x, dim=softmax_dim)
        return prob
    
    def v(self, x):
        x = F.relu(self.fc1(x))
        v = self.fc_v(x)
        return v
    
    def put_data(self, data_env):
        self.data = data_env
        
    def make_batch(self):
        s_batch, a_batch, r_batch, s_prime_batch, done_batch = torch.tensor(self.data['s_lst'], dtype=torch.float) / 17409, \
                                                               torch.tensor(self.data['a_lst']), \
                                                               torch.tensor(self.data['r_lst'], dtype=torch.float), \
                                                               torch.tensor(self.data['s_prime_lst'], dtype=torch.float) / 17409, \
                                                               torch.tensor(self.data['done_lst'], dtype=torch.float)
        return s_batch, a_batch, r_batch, s_prime_batch, done_batch
  
    def train_net(self, ifTest=False):
        s, a, r, s_prime, done = self.make_batch()
        data_size = len(s)
        batch_num = int(data_size / self.batch_size) + 1

        for i in range(batch_num):
            start = self.batch_size * i
            end = self.batch_size * (i+1)
            td_target = r[start: end] + gamma * self.v(s_prime[start: end]) * done[start: end]
            delta = td_target - self.v(s[start: end])
            pi = self.pi(s[start: end], softmax_dim=1)
            pi_a = pi.gather(1, a[start: end])
            loss = -torch.log(pi_a) * delta.detach() + F.smooth_l1_loss(self.v(s[start: end]), td_target.detach())

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()   

        if ifTest:
            self.evaluate_batch()

    def evaluate_batch(self):
        s, a, r, s_prime, done = self.make_batch()
        data_size = len(s)
        batch_num = int(data_size / self.batch_size) + 1
        counter, total_loss = 0, 0.0
        total_label = []
        total_pre_label = []
        total_pre_scores = []
        for i in range(batch_num):
            start = self.batch_size * i
            end = self.batch_size * (i+1)
            pi = self.pi(s[start: end], softmax_dim=1)
            m = Categorical(pi)
            pre_a = m.sample()
            true_a = a[start: end]
            total_label.append(true_a.detach().numpy())
            total_pre_label.append(pre_a.detach().numpy())
            pre_s = pi.gather(1, a[start: end])
            total_pre_scores.append(pre_s.detach().numpy())
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
    data = DataEnv(data_name='clothes').reset()
    model = ActorCritic()    
    model.put_data(data)
    for n_epi in range(10000):
        print("epoch: ", n_epi)
        model.train_net(ifTest=True)


            

if __name__ == '__main__':
    main()