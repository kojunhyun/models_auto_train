# -*- coding:utf-8 -*-

from __future__ import print_function

from tpot import TPOTClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score

import numpy as np
import pandas as pd
import pickle

import os
import sys
import collections
import time
import codecs

class ModelConfig(object):
    """hyper parameter"""
    def __init__(self):
        
        self.population_size = 2        # genetic programming population every generation   
        self.verbosity = 2                # print information and provide a progress bar
        
        self.pattern_size = 4
        
        self.epoch_num = 1              # 에폭 횟수(학습용전체데이터를 몇 회 반복해서 학습할 것인가 입력)

        
class OutputConfig(object):
    """result"""
    def __init__(self):


        self.tr_accuracy = 0.0
        
        self.tr_each_recall = []
        self.tr_each_precision = []
        self.tr_each_f1 = []
        
        self.tr_micro_recall = 0.0
        self.tr_micro_precision = 0.0
        self.tr_micro_f1 = 0.0
        
        self.tr_macro_recall = 0.0
        self.tr_macro_precision = 0.0        
        self.tr_macro_f1 = 0.0
                
        self.va_accuracy = 0
        
        self.va_each_recall = []
        self.va_each_precision = []        
        self.va_each_f1 = []
        
        self.va_micro_recall = 0.0
        self.va_micro_precision = 0.0        
        self.va_micro_f1 = 0.0
        
        self.va_macro_recall = 0.0
        self.va_macro_precision = 0.0        
        self.va_macro_f1 = 0.0
        
        self.processing_time = ''
        
        
def min_max_scaling(x, min, max):
    np_min = np.array(min)
    np_max = np.array(max)
    
    return (x - np_min) / (np_max - np_min + 1e-7) # 1e-7은 0으로 나누는 오류 예방차원  


def remake_data(data, save_path):
    
    dataset_sort = data.sort_values(by=['date','product_no'], axis=0)
    dataset_sort = dataset_sort.reset_index(drop=True)
    
    p_l = list(dataset_sort.product_no.values)
    p_d = collections.defaultdict(lambda: 0)
    for p in p_l:
        p_d[p] += 1
        
    copy_data = dataset_sort.copy()    
    del copy_data['date']
    del copy_data['product_no']
    ###################################################
    # order ouput일 경우, click output 제거
    # click output일 경우, order output 제거
    del copy_data['output_o']
    ###################################################
    
    
    mm_data = copy_data.values.astype(np.float)
    data_min = []
    data_max = []
    for i in range(len(mm_data[0])-1):
        data_min.append(np.min(mm_data[:,i]))
        data_max.append(np.max(mm_data[:,i]))
    
    np.save(save_path + '/data_min', data_min)
    np.save(save_path + '/data_max', data_max)
    
    
    r_x = min_max_scaling(mm_data[:,:-1].astype(np.float), data_min, data_max)
            
    range_y = mm_data[:,-1].astype(np.int)
    #print(range_y)
    r_y = range_y.reshape(len(range_y),1)        
    re_data = np.concatenate((r_x, r_y), axis=1)
       
    input_size = len(re_data[0]) - 1    
    label_ind = len(re_data[0]) - 1
    va_ratio = 0.2
    


    split = StratifiedShuffleSplit(n_splits=1, test_size=va_ratio, random_state=42)
    for train_index, valid_index in split.split(re_data, re_data[:,label_ind]):
        train_set = re_data[train_index]
        valid_set = re_data[valid_index]

            
    
    train_x = train_set[:,:input_size].copy()
    print('train x')
    print(train_x.shape)
    
    train_y = train_set[:,input_size].copy()
    train_y = train_y.reshape(len(train_set),1)
    
    train_y = train_y.flatten()
    print('train y')
    print(train_y.shape)
    
    valid_x = valid_set[:,:input_size].copy()
    print('valid x')
    print(valid_x.shape)
    
    valid_y = valid_set[:,input_size].copy()
    valid_y = valid_y.reshape(len(valid_set),1)
    
    valid_y = valid_y.flatten()
    print('valid y')
    print(valid_y.shape)
    
    
    
    return train_x, train_y, valid_x, valid_y


def score_calculate(matrix_score, o_conf, verbose=False):
    
    real_total = np.sum(matrix_score, axis=0)
    pred_total = np.sum(matrix_score, axis=1)
    
    """
    if len(real_total) == 2:
        recall = (matrix_score[1,1] / (matrix_score[0,1] + matrix_score[1,1]))
        precision = (matrix_score[1,1] / (matrix_score[1,0] + matrix_score[1,1]))
        f1 = (2 * precision * recall) / (precision + recall)
        print('recall : ', recall)
        print('precision : ', recall)
        print('f1 : ', recall)
    #elif len(real_total) > 2:
    """    
    
    each_recall = []
    each_precision= []
    
    for c in range(len(real_total)):
        
        each_recall.append((matrix_score[c,c] / real_total[c]))
        each_precision.append((matrix_score[c,c] / pred_total[c]))
    
    each_recall = np.array(each_recall)
    each_precision = np.array(each_precision)
    each_f1 = (2 * each_precision * each_recall) / (each_precision + each_recall)
    
    
    tmp_tp = 0.0
    tmp_fp = 0.0
    tmp_fn = 0.0
    macro_avg_recall = 0.0
    macro_avg_precision = 0.0
    macro_avg_f1 = 0.0
    for c in range(len(real_total)):
        
        tmp_tp += matrix_score[c,c]
        tmp_fn += real_total[c]
        tmp_fp += pred_total[c]
        
        macro_avg_recall += each_recall[c]
        macro_avg_precision += each_precision[c]
        macro_avg_f1 += each_f1[c]
        
    micro_avg_recall = tmp_tp / tmp_fn
    micro_avg_precision = tmp_tp / tmp_fp
    micro_avg_f1 = (2 * micro_avg_precision * micro_avg_recall) / (micro_avg_precision + micro_avg_recall)
    
    macro_avg_recall = macro_avg_recall / len(each_recall)
    macro_avg_precision = macro_avg_precision / len(each_precision)
    macro_avg_f1 = macro_avg_f1 / len(each_f1)
    
    
    if verbose == True:
        o_conf.tr_each_recall = each_recall
        o_conf.tr_each_precision = each_precision
        o_conf.tr_each_f1 = each_f1
        
        o_conf.tr_micro_recall = micro_avg_recall
        o_conf.tr_micro_precision = micro_avg_precision
        o_conf.tr_micro_f1 = micro_avg_f1
        
        o_conf.tr_macro_recall = macro_avg_recall
        o_conf.tr_macro_precision = macro_avg_precision     
        o_conf.tr_macro_f1 = macro_avg_f1
    else:
        o_conf.va_each_recall = each_recall
        o_conf.va_each_precision = each_precision
        o_conf.va_each_f1 = each_f1
        
        o_conf.va_micro_recall = micro_avg_recall
        o_conf.va_micro_precision = micro_avg_precision
        o_conf.va_micro_f1 = micro_avg_f1
        
        o_conf.va_macro_recall = macro_avg_recall
        o_conf.va_macro_precision = macro_avg_precision     
        o_conf.va_macro_f1 = macro_avg_f1

    
    return o_conf


def TpotClick(data, save_path, config):
    
    start_time = time.time() # 시작시간을 기록한다
    
    info_path = os.path.join(save_path, "tpot_click_info.txt")
    fid = open(info_path, 'w')
    #########################################################################################
    # info
    fid.write("Hyper parameter\n")
    fid.write("*"*50 + "\n")
    
    fid.write("population_size  : %f\n" % config.population_size)
    fid.write("verbosity : %f\n" % config.verbosity) 
    fid.write("epoch_num    : %d\n" % config.epoch_num)
    
    train_x, train_y, valid_x, valid_y = remake_data(data, save_path)
    
    tpot = TPOTClassifier(generations=config.epoch_num, population_size=config.population_size, verbosity=config.verbosity)
    tpot.fit(train_x, train_y)
    
    
    with codecs.open(save_path + '/tpot_click_train_model.txt', 'w', encoding='utf-8') as f:
        pickle.dump(tpot.fitted_pipeline_, f)
    
    
    output_config = OutputConfig()
    
    tr_matrix_scores = np.zeros([config.pattern_size, config.pattern_size], dtype=float)    
    tr_predictions = tpot.fitted_pipeline_.predict(train_x)
    
    total_tagt = np.array(train_y, dtype=int)
    total_pred = np.array(tr_predictions, dtype=int)
    
    for i in range(len(train_y)):
        tr_matrix_scores[total_pred[i], total_tagt[i]] += 1.0
        
    output_config.tr_accuracy = accuracy_score(total_tagt, total_pred)
    
    output_config = score_calculate(tr_matrix_scores, output_config, verbose=True)
 
    
    va_matrix_scores = np.zeros([config.pattern_size, config.pattern_size], dtype=float)
    va_predictions = tpot.fitted_pipeline_.predict(valid_x)
    
    total_tagt = np.array(valid_y, dtype=int)
    total_pred = np.array(va_predictions, dtype=int)
    
    for i in range(len(valid_y)):
        va_matrix_scores[total_pred[i], total_tagt[i]] += 1.0

    output_config.va_accuracy = accuracy_score(total_tagt, total_pred)
    
    output_config = score_calculate(va_matrix_scores, output_config)
    
    
    

    
    end_time = time.time() # 종료시간을 기록한다
    processing_time = end_time - start_time # 경과시간을 구한다
    
    m, s = divmod(processing_time, 60)
    h, m = divmod(m, 60)
    fid.write("Tpot click processing time : %d:%02d:%02f" % (h, m, s))
    fid.close()
    

    
    
    proceessing_time = ('%d:%02d:%02f' % (h,m,s))
    output_config.processing_time = proceessing_time

    
    return output_config


