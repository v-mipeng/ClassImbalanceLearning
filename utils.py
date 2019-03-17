import os
import pickle

import numpy as np
from scipy.io import loadmat
from sklearn.preprocessing import normalize

from sklearn.metrics import (f1_score, accuracy_score, precision_recall_fscore_support,
                             precision_recall_curve, auc, roc_curve)
from imblearn.metrics import classification_report_imbalanced


def evaluate_f1(y, y_pred, pos_label=1):
    precision, recall, f1, support = precision_recall_fscore_support(y, y_pred, pos_label=pos_label)
    return precision, recall, f1


def evaluate_macro_f1(y, y_pred, pos_label=1):
    f1 = f1_score(y, y_pred, pos_label=pos_label, average='macro')
    return f1


def evaluate_auc_prc(y, pred):
    precision, recall, thresholds = precision_recall_curve(y, pred)
    aucprc = auc(recall, precision)
    return aucprc


def evaluate_auc_roc(y, pred):
    fpr, tpr, thresholds = roc_curve(y, pred)
    roc_auc = auc(fpr, tpr)
    return roc_auc


def evaluate_f2(y, y_pred):
    precision, recall, f1, support = precision_recall_fscore_support(y, y_pred, pos_label=1)
    #print classification_report(y, y_pred)
    f2 = (1+0.5**2)*(precision[1]*recall[1])/(0.5**2*precision[1]+recall[1])
    return f2


def load_imb_Gaussian(data_dir):
    train = pickle.load(open(os.path.join(data_dir, 'train.pkl'), 'rb'), encoding='bytes')
    train_x, train_y = train[b'x'], train[b'y']
    valid = pickle.load(open(os.path.join(data_dir, 'valid.pkl'), 'rb'), encoding='bytes')
    valid_x, valid_y = valid[b'x'], valid[b'y']
    test = pickle.load(open(os.path.join(data_dir, 'test.pkl'), 'rb'), encoding='bytes')
    test_x, test_y = test[b'x'], test[b'y']
    return train_x, train_y, valid_x, valid_y, test_x, test_y


def load_imb_Credit_Fraud(data_dir):
    train = pickle.load(open(os.path.join(data_dir, 'train.pkl'), 'rb'), encoding='bytes')
    train_x, train_y = train[b'x'], train[b'y']
    valid = pickle.load(open(os.path.join(data_dir, 'valid.pkl'), 'rb'), encoding='bytes')
    valid_x, valid_y = valid[b'x'], valid[b'y']
    test = pickle.load(open(os.path.join(data_dir, 'test.pkl'), 'rb'), encoding='bytes')
    test_x, test_y = test[b'x'], test[b'y']
    return train_x, train_y, valid_x, valid_y, test_x, test_y


def load_imb_Page(data_dir):
    train = pickle.load(open(os.path.join(data_dir, 'train.pkl'), 'rb'), encoding='bytes')
    train_x, train_y = train[b'x'], train[b'y']
    valid = pickle.load(open(os.path.join(data_dir, 'valid.pkl'), 'rb'), encoding='bytes')
    valid_x, valid_y = valid[b'x'], valid[b'y']
    test = pickle.load(open(os.path.join(data_dir, 'test.pkl'), 'rb'), encoding='bytes')
    test_x, test_y = test[b'x'], test[b'y']
    print(train_x.shape)
    return train_x, train_y, valid_x, valid_y, test_x, test_y


def load_imb_Vehicle(data_dir):
    train = pickle.load(open(data_dir, 'rb'), encoding='bytes')
    train_x, train_y = train[b'x'], train[b'y']
    return train_x, train_y #valid_x, valid_y


def load_checker_board(data_dir):
    train = pickle.load(open(data_dir, 'rb'), encoding='bytes')
    train_x, train_y = train[b'x'], train[b'y']
    return train_x, train_y


def load_imb_data(data_dir):
    train = pickle.load(open(data_dir, 'rb'), encoding='bytes')
    train_x, train_y = train[b'x'], train[b'y']
    return train_x, train_y
