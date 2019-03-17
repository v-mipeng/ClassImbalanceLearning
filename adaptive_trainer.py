import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC as SVC
from sklearn.tree import DecisionTreeClassifier as DT

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Categorical

from utils import *


class MLPPolicy(nn.Module):
    '''
    Works when input dimension is low.
    '''
    def __init__(self, input_dim, hidden_dims, activations=None, output_dim=2):
        super(MLPPolicy, self).__init__()
        dims = [input_dim] + hidden_dims + [output_dim]
        self.linears = []
        for i in range(1, len(dims)):
            linear = nn.Linear(dims[i - 1], dims[i])
            setattr(self, 'linear_{}'.format(i), linear)
            self.linears.append(linear)
        if activations is not None:
            self.activations = activations
        else:
            self.activations = [nn.Sigmoid() for _ in range(len(dims)-2)] + [nn.Softmax(dim=1)]

    def forward(self, x):
        x_hat = x
        for linear, activation in zip(self.linears, self.activations):
            x_hat = activation(linear(x_hat))
        return x_hat


class GRUPolicy(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=2, activations=None):
        super(GRUPolicy, self).__init__()
        self.hidden_dim = hidden_dim
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, output_dim)
        self.output_dim = output_dim

    def forward(self, x):
        out, _ = self.gru(x.unsqueeze(0))
        out = out.view(out.size()[1], out.size(2))
        out = nn.Softmax(dim=1)(self.linear(out))
        return out


class MLPTrainer(object):
    def __init__(self):
        self.env = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5), random_state=1)

    def train(self, data_dir='../data/gaussian/', hidden_dims=None, major_ratio=0.05, lr=None):
        if hidden_dims is None:
            hidden_dims = [5]
        train_x, train_y, valid_x, valid_y, test_x, test_y = self.load_data(data_dir)
        self.policy = MLPPolicy(input_dim=train_x.shape[1] + train_y.shape[1], hidden_dims=hidden_dims)
        self.policy.cuda()
        if lr is None:
            self.optimizer = optim.RMSprop(self.policy.parameters())
        else:
            self.optimizer = optim.RMSprop(self.policy.parameters(), lr=lr)
        best_valid_reward = 0.
        best_test_reward = 0.
        i_episode = 10
        epoch = 0
        x = Variable(torch.cat([torch.from_numpy(train_x).float(), torch.from_numpy(train_y).float()], dim=1)).cuda()
        y = np.zeros((len(train_y), 2)).astype('float32')
        idx = np.argmax(train_y, axis=1)
        y[idx == 0] = [1-major_ratio, major_ratio]
        y[idx == 1] = [0., 1.]
        y = Variable(torch.from_numpy(y).cuda())
        self.initialize_policy(self.policy, x, y)
        while True:
            weight_probs = self.policy(x)
            cross_entropy = - torch.mean(torch.sum(weight_probs * torch.log(weight_probs + 1e-20), dim=1))
            self.reg = torch.mean(weight_probs[:, 1]) * 1e-4
            print(weight_probs)
            log_probs = []
            train_rewards = []
            valid_rewards = []
            test_rewards = []
            for i in range(i_episode):
                data_weights, log_prob = self.sample_weight(weight_probs)
                train_reward, valid_reward, test_reward = self.get_reward(train_x, train_y, data_weights,
                                                valid_x, valid_y, test_x, test_y)
                log_probs.append(log_prob)
                train_rewards.append(train_reward)
                valid_rewards.append(valid_reward)
                test_rewards.append(test_reward)
            if best_valid_reward < np.mean(valid_rewards):
                best_valid_reward = np.mean(valid_rewards)
                best_test_reward = np.mean(test_rewards)
            self.update_policy(log_probs, train_rewards)
            print('Train reward: {} in epoch: {} '.format(np.mean(train_rewards), epoch))
            print('Valid reward: {} in epoch: {} '.format(np.mean(valid_rewards), epoch))
            print('Test reward: {} in epoch: {} '.format(np.mean(test_rewards), epoch))
            print('Best valid F1: {}'.format(best_valid_reward))
            print('Best test F1: {}'.format(best_test_reward))
            epoch += 1
        print('Best valid F1: {}'.format(best_valid_reward))
        print('Best test F1: {}'.format(best_test_reward))

    def sample_weight(self, probs):
        if not isinstance(probs, Variable):
            probs = Variable(probs)
        m = Categorical(probs)
        action = m.sample()
        return action.data.cpu().numpy(), m.log_prob(action).mean().cuda()

    def get_reward(self, train_x, train_y, train_weights, valid_x, valid_y, test_x, test_y):
        '''Train the classifier with supervised

        :param train_x:
        :param train_y:
        :param train_weights:
        :param valid_x:
        :param valid_y:
        :return: The reward (F1)
        '''
        idx = train_weights == 1
        x = train_x[idx]
        y = train_y[idx]
        self.env.fit(x, y)
        preds = self.env.predict(train_x)
        _, _, train_reward = evaluate_f1(train_y, preds, pos_label=1)
        preds = self.env.predict(valid_x)
        _, _, valid_reward = evaluate_f1(valid_y, preds, pos_label=1)
        preds = self.env.predict(test_x)
        _, _, test_reward = evaluate_f1(test_y, preds, pos_label=1)
        return train_reward[1], valid_reward[1], test_reward[1]

    def update_policy(self, log_probs, rewards):
        rewards = Variable(torch.Tensor(rewards).cuda())
        policy_loss = []
        rewards = (rewards - rewards.mean()) / (rewards.std() + float(np.finfo(np.float32).eps))
        for log_prob, reward in zip(log_probs, rewards):
            policy_loss.append(-log_prob * reward)
        self.optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        print('policy loss:{}'.format(policy_loss.data.cpu()))
        print('reg:{}'.format(self.reg.data.cpu()))
        (policy_loss+self.reg).backward()
        self.optimizer.step()

    def load_data(self, data_dir):
        return load_imb_Gaussian(data_dir)

    def initialize_policy(self, policy, x, y, epoch=30):
        optimizer = optim.RMSprop(policy.parameters(), lr=0.001)
        for e in range(epoch):
            probs = policy(x)
            print(probs)
            loss = -torch.mean(torch.sum(y * torch.log(probs), dim=1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


class GRUTrainer(MLPTrainer):
    def train(self, data_dir='../data/gaussian/', hidden_dim=50, major_ratio=0.05, lr=None):
        train_x, train_y, valid_x, valid_y, test_x, test_y = self.load_data(data_dir)
        self.policy = GRUPolicy(input_dim=train_x.shape[1] + train_y.shape[1], hidden_dim=hidden_dim)
        self.policy.cuda()
        if lr is None:
            self.optimizer = optim.RMSprop(self.policy.parameters())
        else:
            self.optimizer = optim.RMSprop(self.policy.parameters(), lr=lr)
        best_valid_reward = 0.
        best_test_reward = 0.
        best_train_reward = 0.
        i_episode = 10
        epoch = 0
        x = Variable(torch.cat([torch.from_numpy(train_x).float(), torch.from_numpy(train_y).float()], dim=1)).cuda()
        y = np.zeros((len(train_y), 2)).astype('float32')
        idx = np.argmax(train_y, axis=1)
        y[idx == 0] = [1 - major_ratio, major_ratio]
        y[idx == 1] = [0.1, 0.9]
        y = Variable(torch.from_numpy(y).cuda())
        self.initialize_policy(self.policy, x, y)
        self.epoch = epoch
        while True:
            x = Variable(
                torch.cat([torch.from_numpy(train_x).float(), torch.from_numpy(train_y).float()], dim=1)).cuda()
            weight_probs = self.policy(x)
            self.reg = torch.mean(weight_probs[:, 1]) ** 2 * 1e-3
            print(weight_probs[:5])
            log_probs = []
            train_rewards = []
            valid_rewards = []
            test_rewards = []
            for i in range(i_episode):
                self.epoch += 1
                data_weights, log_prob = self.sample_weight(weight_probs)
                (train_reward, valid_reward, test_reward,
                 train_x, train_y, valid_x, valid_y) = self.get_reward(train_x, train_y, data_weights, weight_probs,
                                                                       valid_x, valid_y, test_x, test_y)
                log_probs.append(log_prob)
                train_rewards.append(train_reward)
                valid_rewards.append(valid_reward)
                test_rewards.append(test_reward)
            if best_train_reward < np.mean(train_rewards):
                best_valid_reward = np.mean(valid_rewards)
                best_test_reward = np.mean(test_rewards)
                best_train_reward = np.mean(train_rewards)
            self.update_policy(log_probs, train_rewards)
            print('Train reward: {} in epoch: {} '.format(np.mean(train_rewards), self.epoch))
            print('Valid reward: {} in epoch: {} '.format(np.mean(valid_rewards), self.epoch))
            print('Test reward: {} in epoch: {} '.format(np.mean(test_rewards), self.epoch))
            print('Best train reward: {}'.format(best_train_reward))
            print('Best valid reward: {}'.format(best_valid_reward))
            print('Best test reward: {}'.format(best_test_reward))


class TaskTrainer(GRUTrainer):
    def __init__(self, task='spam'):
        super(TaskTrainer, self).__init__()
        from sklearn.svm import SVC as SVM
        self.task = task
        if task == 'vehicle':
            self.env = SVM(C=1e2, kernel='rbf', random_state=0) # For vehicle task
        elif task == 'page':
            self.env = SVM(C=1e2, kernel='rbf', random_state=0, gamma=1e-2) # For page blocks
        elif task == 'credit':
            self.env = DT(max_depth=4) # For credit card task
        elif task == 'spam':
            self.env = LogisticRegression(C=1e2, random_state=0) # For spam detection task

    def train(self, data_dir='../data/gaussian/', hidden_dim=50, major_ratio=0.05, lr=None):
        self.x_l, self.y_l, self.x_u, self.y_u, self.x_all, self.y_all = self.load_data(data_dir)
        self.policy = GRUPolicy(input_dim=self.x_l.shape[1] + self.y_l.shape[1], hidden_dim=hidden_dim)
        self.policy.cuda()
        if lr is None:
            self.optimizer = optim.RMSprop(self.policy.parameters())
        else:
            self.optimizer = optim.RMSprop(self.policy.parameters(), lr=lr)
        best_valid_reward = 0.
        best_test_reward = 0.
        best_train_reward = 0.
        i_episode = 10
        epoch = 0
        x = Variable(torch.cat([torch.from_numpy(self.x_l).float(), torch.from_numpy(self.y_l).float()], dim=1)).cuda()
        y = np.zeros((len(self.x_l), 2)).astype('float32')
        idx = np.argmax(self.y_l, axis=1)
        y[idx == 0] = [1 - major_ratio, major_ratio]
        y[idx == 1] = [0.1, 0.9]
        y = Variable(torch.from_numpy(y).cuda())
        self.initialize_policy(self.policy, x, y)
        self.epoch = epoch
        self.data_changed = False
        import time
        t0 = time.time()
        while True:
            if self.data_changed:
                x = Variable(
                torch.cat([torch.from_numpy(self.x_l).float(), torch.from_numpy(self.y_l).float()], dim=1)).cuda()
                self.data_changed = False
            weight_probs = self.policy(x)
            self.reg = torch.mean(weight_probs[:, 1]) **2 * 1e-3
            print(weight_probs[:5])
            log_probs = []
            train_rewards = []
            valid_rewards = []
            test_rewards = []
            for i in range(i_episode):
                self.epoch += 1
                data_weights, log_prob = self.sample_weight(weight_probs)
                (train_reward, valid_reward, test_reward) = self.get_reward(data_weights, weight_probs)
                log_probs.append(log_prob)
                train_rewards.append(train_reward)
                valid_rewards.append(valid_reward)
                test_rewards.append(test_reward)
            if best_train_reward < np.mean(train_rewards):
                best_valid_reward = np.mean(valid_rewards)
                best_test_reward = np.mean(test_rewards)
                best_train_reward = np.mean(train_rewards)
            self.update_policy(log_probs, train_rewards)
            print('Train reward: {} in epoch: {} '.format(np.mean(train_rewards), self.epoch / i_episode))
            print('Valid reward: {} in epoch: {} '.format(np.mean(valid_rewards), self.epoch / i_episode))
            print('Test reward: {} in epoch: {} '.format(np.mean(test_rewards), self.epoch / i_episode))
            print('Best train reward: {}'.format(best_train_reward))
            print('Best valid reward: {}'.format(best_valid_reward))
            print('Best test reward: {}'.format(best_test_reward))
            t1 = time.time()
            print('Epoch:{} Time:{:0.2f}s'.format(self.epoch%i_episode, t1 - t0))

    def get_reward(self, train_weights, weight_probs):
        '''Train the classifier with supervised

        :param train_x:
        :param train_y:
        :param train_weights:
        :param valid_x:
        :param valid_y:
        :return: The reward (F1)
        '''
        from imblearn.metrics import geometric_mean_score
        from sklearn.metrics import matthews_corrcoef
        idx = train_weights == 1
        x = self.x_l[idx]
        y = self.y_l[idx]
        self.env.fit(x, np.argmax(y, axis=1).astype('int32'))
        if task == 'vehicle':
            preds = self.env.predict(self.x_all)
            valid_reward = geometric_mean_score(np.argmax(self.y_all, axis=1).astype('int32'), preds)
            if len(self.y_u) > 0:
                preds = self.env.decision_function(self.x_u)
        elif self.task == 'page':
            preds = self.env.predict(self.x_all)
            valid_reward = matthews_corrcoef(np.argmax(self.y_all, axis=1).astype('int32'), preds)
            if len(self.y_u) > 0:
                preds = self.env.decision_function(self.x_u)
        elif self.task == 'spam':
            preds = self.env.predict(self.x_all)
            valid_reward = evaluate_f2(np.argmax(self.y_all, axis=1).astype('int32'), preds) # for spam
            if len(self.y_u) > 0:
                preds = self.env.predict_proba(self.x_u)
        elif task == 'credit':
            preds = self.env.predict_proba(self.x_all)[:, 1]
            valid_reward = evaluate_auc_prc(np.argmax(self.y_all, axis=1).astype('int32'), preds)
            if len(self.y_u) > 0:
                preds = self.env.predict_proba(self.x_u)
        if self.epoch >= 1000 and self.epoch % 1000 == 0:
            if len(self.y_u) > 0:
                if task == 'vehicle' or task == 'page':
                    preds = self.env.decision_function(self.x_u)
                    probs = np.zeros((len(preds), 2)).astype('float32')
                    idxes = preds < 0
                    probs[idxes, 0] = 0.5 - 0.5 * preds[idxes] / preds[idxes].min()
                    probs[idxes, 1] = 1 - probs[idxes, 0]
                    idxes = preds >= 0
                    probs[idxes, 1] = 0.5 + 0.5 * preds[idxes] / preds[idxes].max()
                    probs[idxes, 0] = 1 - probs[idxes, 1]
                    preds = probs
                elif self.task == 'spam':
                    preds = self.env.predict_proba(self.x_u)
                elif task == 'credit':
                    preds = self.env.predict_proba(self.x_u)
                self.x_l, self.y_l, self.x_u, self.y_u = self.updata_data(weight_probs.data.cpu().numpy()[:, 1],
                                                                       preds, num=10*self.pos_num)
                print('train set shape:{}'.format(self.x_l.shape))
                print('valid set shape:{}'.format(self.x_u.shape))
                self.optimizer = optim.RMSprop(self.policy.parameters(), lr=0.001 )
                self.data_changed = True
            else:
                #print('train set shape:{}'.format(self.x_l.shape))
                #input('Input any character!')
                pass
        return valid_reward, valid_reward, valid_reward

    def updata_data(self, l_weight, u_pred, num):
        x = self.x_l
        y = self.y_l
        # move wrongly classified examples from D_u to D_l
        idxes = np.arange(len(self.y_u))
        true_proba = u_pred[idxes, np.argmax(self.y_u, axis=1)]
        r_idxes = idxes[true_proba < 0.5]
        l_idxes = r_idxes[min(num, len(r_idxes)):]
        r_idxes = r_idxes[:min(num, len(r_idxes))]
        x = np.concatenate([x, self.x_u[r_idxes]])
        y = np.concatenate([y, self.y_u[r_idxes]])
        if len(l_idxes) > 0:
            l_idxes = np.concatenate([idxes[true_proba >= 0.5], l_idxes])
        else:
            l_idxes = idxes[true_proba >= 0.5]
        x_u = self.x_u[l_idxes]
        y_u = self.y_u[l_idxes]
        return x, y, x_u, y_u

    def load_data(self, data_dir):
        train_x, train_y = load_imb_data(data_dir)
        # Initialize policy with smaller data set first.
        y = np.argmax(train_y, axis=1)
        pos_num = np.sum(y==1)
        neg_idxes = np.arange(len(y))[y==0]
        self.pos_num = pos_num
        chosen_idxes = np.random.choice(a=neg_idxes, size=10*pos_num, replace=False)
        left_idxes = np.array(list(set(neg_idxes).difference(set(chosen_idxes))))
        idxes = np.concatenate([chosen_idxes, np.arange(len(y))[y==1]])
        return train_x[idxes], train_y[idxes], train_x[left_idxes], train_y[left_idxes], train_x, train_y


if __name__ == '__main__':
    import sys
    task = sys.argv[1]
    trainer = TaskTrainer(task=task)
    if task == 'vehicle':
        hidden_dim = 25
        major_ratio = 0.2
        lr = 0.001
    elif task == 'page':
        hidden_dim = 25
        major_ratio = 0.1
        lr = 0.001
    elif task == 'credit':
        hidden_dim = 50
        major_ratio = 0.05
        lr = 0.001
    elif task == 'spam':
        hidden_dim = 25
        major_ratio = 0.1
        lr = 0.001
    else:
        print('Undefined task!')
        sys.exit(1)
    trainer.train(data_dir='../data/real/{}/train.pkl'.format(task),
                  hidden_dim=hidden_dim,
                  major_ratio=major_ratio,
                  lr=lr)