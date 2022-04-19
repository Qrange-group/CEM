#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os
import sys
import resource
import time
import tensorflow as tf
import numpy as np
import math
from sklearn.metrics import confusion_matrix, classification_report, f1_score, precision_score, recall_score, accuracy_score
import itertools
# import torch
import json

PAD_token = 0


def to_categorical(y, num_classes=None, dtype='float32'):
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical


def load_config(config_path):
    with open(config_path, 'r') as fp:
        return json.load(fp)


def get_now_time():
    now_time = '_'.join(time.asctime(time.localtime(time.time())).split(' '))  # [:3])
    return now_time


def print_trainable_variables(output_detail, logger):
    total_parameters = 0
    parameters_string = ""

    for variable in tf.compat.v1.trainable_variables():

        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_parameters += variable_parameters
        if len(shape) == 1:
            parameters_string += ("%s %d\n" % (variable.name, variable_parameters))
        else:
            parameters_string += ("%s %s=%d\n" % (variable.name, str(shape), variable_parameters))

    if logger:
        if output_detail:
            logger.info('\n' + parameters_string)
        logger.info("Total %d variables, %s params" % (len(tf.compat.v1.trainable_variables()), "{:,}".format(total_parameters)))
    else:
        if output_detail:
            print('\n' + parameters_string)
        print("Total %d variables, %s params" % (len(tf.compat.v1.trainable_variables()), "{:,}".format(total_parameters)))


def show_layer_info(layer_name, layer_out, logger=None):
    if logger:
        logger.info('[layer]: %s\t[shape]: %s'
                    % (layer_name, str(layer_out.get_shape().as_list())))
    else:
        print('[layer]: %s\t[shape]: %s'
              % (layer_name, str(layer_out.get_shape().as_list())))


def show_layer_info_with_memory(layer_name, layer_out, logger=None):
    if logger:
        logger.info('[layer]: %s\t[shape]: %s \n%s'
                    % (layer_name, str(layer_out.get_shape().as_list()), show_memory_use()))
    else:
        print('[layer]: %s\t[shape]: %s \n%s'
              % (layer_name, str(layer_out.get_shape().as_list()), show_memory_use()))


def show_memory_use():
    rusage_denom = 1024.
    if sys.platform == 'darwin':
        rusage_denom = rusage_denom * rusage_denom
    ru = resource.getrusage(resource.RUSAGE_SELF)
    total_memory = 1. * (ru.ru_maxrss + ru.ru_ixrss +
                         ru.ru_idrss + ru.ru_isrss) / rusage_denom
    strinfo = "\x1b[33m [Memory] Total Memory Use: %.4f MB \t Resident: %ld Shared: %ld UnshareData: " \
              "%ld UnshareStack: %ld \x1b[0m" % \
              (total_memory, ru.ru_maxrss, ru.ru_ixrss, ru.ru_idrss, ru.ru_isrss)
    return strinfo


def zeroPadding(l, fillvalue=PAD_token):
    return list(itertools.zip_longest(*l, fillvalue=fillvalue))


def binaryMatrix(l):
    m = []
    for i, seq in enumerate(l):
        m.append([])
        for token in seq:
            if token == PAD_token:
                m[i].append(0)
            else:
                m[i].append(1)
    return m


def get_a_p_r_f_sara(target, predict, category=1):
    # sklearn version
    accuracy = accuracy_score(target, predict)
    precision = precision_score(target, predict, average='macro')
    recall = recall_score(target, predict, average='macro')
    f1 = f1_score(target, predict)
    macro_f1_score = f1_score(target, predict, average='macro')
    return accuracy, precision, recall, f1, macro_f1_score


def golden_transfer_within_tolerance_exp(pre_labels, true_labels, t=1, eps=1e-7, lamb=0):
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


def get_gtt_score(label_list, pre_list, lamb=0.):
    gtt_score_list_1 = []
    gtt_score_list_2 = []
    gtt_score_list_3 = []
    for pres, labels in zip(pre_list, label_list):
        gtt_score_list_1.append(golden_transfer_within_tolerance_exp(pres, labels, t=1, lamb=lamb))
        gtt_score_list_2.append(golden_transfer_within_tolerance_exp(pres, labels, t=2, lamb=lamb))
        gtt_score_list_3.append(golden_transfer_within_tolerance_exp(pres, labels, t=3, lamb=lamb))

    return np.mean(gtt_score_list_1), np.mean(gtt_score_list_2), np.mean(gtt_score_list_3)


if __name__ == "__main__":
    pass

