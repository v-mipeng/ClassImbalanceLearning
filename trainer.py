import numpy as np

from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC as SVC
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.tree import DecisionTreeClassifier as DT
from sklearn.cross_validation import train_test_split
from sklearn.datasets import make_classification
from sklearn import datasets, neighbors
from sklearn.metrics import auc, roc_curve
from sklearn.model_selection import StratifiedKFold

from costcla.datasets import load_creditscoring2
from costcla.models import CostSensitiveLogisticRegression, ThresholdingOptimization
from costcla.metrics import savings_score, cost_loss


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


class BaseTrainer(object):
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
        policy_loss = sum(policy_loss).cuda()
        print('policy loss:{}'.format(policy_loss.data.cpu()))
        print('reg:{}'.format(self.reg.data.cpu()))
        (policy_loss).backward() #todo: only apply for 0.1.0 version
        self.optimizer.step()

    def load_data(self, data_dir):
        return load_imb_Gaussian(data_dir)

    def initialize_policy(self, policy, x, y, epoch=30):
        optimizer = optim.RMSprop(policy.parameters(), lr=0.001)
        for e in range(epoch):
            probs = policy(x)
            loss = -torch.mean(torch.sum(y * torch.log(probs), dim=1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


class GRUTrainer(BaseTrainer):
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
        import time
        t0 = time.time()
        while True:
            weight_probs = self.policy(x)
            self.reg = torch.mean(weight_probs[:, 1]) ** 2 * 1e-3
            #print(weight_probs[:5])
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
            if best_train_reward < np.mean(train_rewards):
                best_valid_reward = np.mean(valid_rewards)
                best_test_reward = np.mean(test_rewards)
                best_train_reward = np.mean(train_rewards)
            self.update_policy(log_probs, train_rewards)
            print('Train reward: {} in epoch: {} '.format(np.mean(train_rewards), self.epoch))
            #print('Valid reward: {} in epoch: {} '.format(np.mean(valid_rewards), self.epoch))
            #print('Test reward: {} in epoch: {} '.format(np.mean(test_rewards), self.epoch))
            print('Best train reward: {}'.format(best_train_reward))
            #print('Best valid reward: {}'.format(best_valid_reward))
            #print('Best test reward: {}'.format(best_test_reward))
            self.epoch += 1
            t1 = time.time()
            print('Epoch:{} Time:{:0.2f}s'.format(self.epoch, t1-t0))


class SynTrainer(GRUTrainer):
    '''Experiments on synthetic data'''
    def __init__(self):
        super(SynTrainer, self).__init__()
        self.env = LogisticRegression(C=1e1)
        #self.env = KNN(n_neighbors=10)
        #self.env = DT(max_depth=3)
        #self.env = SVC(C=1e3)

    def load_data(self, data_dir):
        return load_imb_Gaussian(data_dir)

    def get_reward(self, train_x, train_y, train_weights, valid_x, valid_y, test_x, test_y):
        idx = train_weights == 1
        x = train_x[idx]
        y = train_y[idx]
        self.env.fit(x, np.argmax(y, axis=1).astype('int32'))
        preds = self.env.predict(train_x)
        _, _, train_reward = evaluate_f1(np.argmax(train_y, axis=1).astype('int32'), preds, pos_label=1)
        preds = self.env.predict(valid_x)
        _, _, valid_reward = evaluate_f1(np.argmax(valid_y, axis=1).astype('int32'), preds, pos_label=1)
        preds = self.env.predict(test_x)
        _, _, test_reward = evaluate_f1(np.argmax(test_y, axis=1).astype('int32'), preds, pos_label=1)
        if self.epoch == 50:
            np.save('gaussian_weight.npy', np.array(train_weights))
        return train_reward[1], valid_reward[1], test_reward[1]


class CheckerBoardTrainer(GRUTrainer):
    def __init__(self, imb_ratio=5):
        super(CheckerBoardTrainer, self).__init__()
        from sklearn.svm import SVC as SVM
        self.imb_ratio = imb_ratio
        self.env = SVM(C=1e4, kernel='rbf', random_state=0)

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
            weight_probs = self.policy(x)
            self.reg = torch.mean(weight_probs[:, 1]) ** 2 * 1e-3
            # print(weight_probs[:5])
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
            if best_train_reward < np.mean(train_rewards):
                best_valid_reward = np.mean(valid_rewards)
                best_test_reward = np.mean(test_rewards)
                best_train_reward = np.mean(train_rewards)
                np.save('weight_{}.npy'.format(self.imb_ratio), data_weights)
            self.update_policy(log_probs, train_rewards)
            print('Train reward: {} in epoch: {} '.format(np.mean(train_rewards), self.epoch))
            print('Valid reward: {} in epoch: {} '.format(np.mean(valid_rewards), self.epoch))
            print('Test reward: {} in epoch: {} '.format(np.mean(test_rewards), self.epoch))
            print('Best train reward: {}'.format(best_train_reward))
            print('Best valid reward: {}'.format(best_valid_reward))
            print('Best test reward: {}'.format(best_test_reward))
            self.epoch += 1

    def load_data(self, data_dir):
        train_x, train_y = load_checker_board(data_dir)
        return train_x, train_y, train_x, train_y, train_x, train_y

    def get_reward(self, train_x, train_y, train_weights, valid_x, valid_y, test_x, test_y):
        idx = train_weights == 1
        x = train_x[idx]
        y = train_y[idx]
        self.env.fit(x, np.argmax(y, axis=1).astype('int32'))
        preds = self.env.predict(train_x)
        train_reward = evaluate_macro_f1(np.argmax(train_y, axis=1).astype('int32'), preds, pos_label=1)
        return train_reward, train_reward, train_reward


class TaskTrainer(GRUTrainer):
    def __init__(self, task):
        super(TaskTrainer, self).__init__()
        from sklearn.svm import SVC as SVM
        self.task = task
        if task == 'vehicle':
            self.env = SVM(C=1e2, kernel='rbf', random_state=0)  # For vehicle task
        elif task == 'page':
            self.env = SVM(C=1e2, kernel='rbf', random_state=0, gamma=1e-2)  # For page blocks
        elif task == 'credit':
            self.env = DT(max_depth=4)  # For credit card task
        elif task == 'spam':
            self.env = LogisticRegression(C=1e2, random_state=0)  # For spam detection task

    def get_reward(self, train_x, train_y, train_weights, valid_x, valid_y, test_x, test_y):
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
        x = train_x[idx]
        y = train_y[idx]
        self.env.fit(x, np.argmax(y, axis=1).astype('int32'))
        if task == 'vehicle':
            preds = self.env.predict(valid_x)
            valid_reward = geometric_mean_score(np.argmax(valid_y, axis=1).astype('int32'), preds)
        elif self.task == 'page':
            preds = self.env.predict(valid_x)
            valid_reward = matthews_corrcoef(np.argmax(valid_y, axis=1).astype('int32'), preds)
        elif self.task == 'spam':
            preds = self.env.predict(valid_x)
            valid_reward = evaluate_f2(np.argmax(valid_y, axis=1).astype('int32'), preds)  # for spam
        elif task == 'credit':
            preds = self.env.predict_proba(valid_x)[:, 1]
            valid_reward = evaluate_auc_prc(np.argmax(valid_y, axis=1).astype('int32'), preds)
        return valid_reward, valid_reward, valid_reward

    def load_data(self, data_dir):
        train_x, train_y = load_imb_data(data_dir)
        return train_x, train_y, train_x, train_y, train_x, train_y



if __name__ == '__main__':
    #trainer = HImbGaussianTrainer()
    #trainer.train(input_hidden_dims=[5], union_hidden_dims=[])
    #trainer = SynTrainer()
    #trainer.train(hidden_dims=[10, 5])
    # trainer = LFWTrainer(1)
    # trainer.train(hidden_dim=25, major_ratio=0.2, lr=0.001)
    # import sys
    # sys.exit(0)
    #trainer = CreditFraudTrainer()
    #trainer.train(data_dir='../data/real/creditcard/', hidden_dim=50, major_ratio=0.1, lr=0.001)
    #trainer = PageTrainer()
    #trainer.train(data_dir='../data/real/page/', hidden_dim=25, major_ratio=0.1, lr=0.001)
    #trainer = SynTrainer()
    #trainer.train(data_dir='../data/gaussian/', hidden_dim=25, major_ratio=0.1, lr=0.0001)
    #trainer = CreditScoreTrainer()
    #trainer.train(data_dir='../data/real/creditcard/', hidden_dim=50, major_ratio=0.1, lr=0.001)
    #trainer = AmazonTrainer()
    #trainer.train(data_dir='../data/real/cmd/amazon.mat', hidden_dim=50, source_domain=2, target_domain=1, lr=0.001)
    #import sys
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


