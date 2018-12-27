# -*- coding:utf-8 -*-

from __future__ import print_function

from datetime import datetime, timedelta

import numpy as np
import tensorflow as tf
import pandas as pd
import os
import sys
import collections
import time
from sklearn.model_selection import StratifiedShuffleSplit


#################################################################################################################################################
# 랜덤에 의해 똑같은 결과를 재현하도록 시드 설정
# 하이퍼파라미터를 튜닝하기 위한 용도(흔들리면 무엇때문에 좋아졌는지 알기 어려움)
tf.set_random_seed(777)

    
class ModelConfig(object):
    """hyper parameter"""
    def __init__(self):
        self.input_data_column_cnt = 17  # 입력데이터의 컬럼 개수(Variable 개수)
        self.output_data_column_cnt = 4  # 결과데이터의 컬럼 개수
        
        self.seq_length = 14             # 1개 시퀀스의 길이(시계열데이터 입력 개수)
        self.rnn_cell_hidden_dim = 100   # 각 셀의 (hidden)출력 크기
        
        self.pattern_size = 4            # 클릭 0 ~ 60: 0, 클릭 61 ~ 120  : 1, 클릭 121 ~ 360 : 2, 클릭 360 ~ : 3
        self.forget_bias = 1.0           # 망각편향(기본값 1.0)
        
        self.num_stacked_layers = 2      # stacked LSTM layers 개수
        self.keep_prob = 1.0             # dropout할 때 keep할 비율    ## train = 0.7 test = 1.0
        
        self.batch_size = 64
        self.epoch_num = 100               # 에폭 횟수(학습용전체데이터를 몇 회 반복해서 학습할 것인가 입력)
        #epoch_num = 100            
        self.learning_rate = 0.01       # 학습률
        
        self.max_grad_norm = 1
        self.init_scale = 0.1


class OutputConfig(object):
    """result"""
    def __init__(self):
        self.best_iteration = 0
        self.tr_loss = 0.0
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
                
        self.va_loss = 0
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
        
        
class LstmOrderModel(object):
    def __init__(self, is_training, config):
        self.is_training = is_training
        self.batch_size = config.batch_size
        self.seq_length = config.seq_length
        
        # 텐서플로우 플레이스홀더 생성
        # 입력 X, 출력 Y를 생성한다
        
        self.X = tf.placeholder(tf.float32, [None, self.seq_length, config.input_data_column_cnt])
        print("X: ", self.X)
        self.targets = tf.placeholder(tf.int32, [None, self.seq_length, 1])
        print("targets: ", self.targets)
        
        # num_stacked_layers개의 층으로 쌓인 Stacked RNNs 생성
        stackedRNNs = [self.lstm_cell(config) for _ in range(config.num_stacked_layers)]
        multi_cells = tf.contrib.rnn.MultiRNNCell(stackedRNNs, state_is_tuple=True) if config.num_stacked_layers > 1 else self.lstm_cell(config)
        
        # RNN Cell(여기서는 LSTM셀임)들을 연결
        hypothesis, _states = tf.nn.dynamic_rnn(multi_cells, self.X, dtype=tf.float32)
        print("hypothesis : ", hypothesis)
        
        hypothesis = tf.reshape(hypothesis, [self.batch_size*self.seq_length, config.rnn_cell_hidden_dim])
        
        softmax_w = tf.get_variable("softmax_w", [config.rnn_cell_hidden_dim, config.pattern_size], dtype=tf.float32)
        softmax_b = tf.get_variable("softmax_b", [config.pattern_size], dtype=tf.float32)
        logits = tf.nn.xw_plus_b(hypothesis, softmax_w, softmax_b)
        
        # Reshape logits to be a 3-D tensor for sequence loss
        self.logits = tf.reshape(logits, [self.batch_size, self.seq_length, config.pattern_size])
        print('logits : ', self.logits)
        
        tgg = tf.reshape(self.targets, [self.batch_size, self.seq_length])

        # Use the contrib sequence loss and average over the batches
        loss = tf.contrib.seq2seq.sequence_loss(
            self.logits,
            tgg, ##
            tf.ones([self.batch_size, self.seq_length], dtype=tf.float32),
            average_across_timesteps=False,
            average_across_batch=True)
        
        self.cost = tf.reduce_sum(loss)
        self.final_state = _states

        self.softmax_out = tf.nn.softmax(tf.reshape(self.logits, [self.batch_size*self.seq_length, config.pattern_size]))
        self.predict = tf.cast(tf.argmax(self.softmax_out, axis=1), tf.int32)
        tgg = tf.reshape(self.targets, [-1])
        
        
        correct_prediction = tf.equal(self.predict, tgg)
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        
        
        if not is_training:
            return
        
        optimizer = tf.train.AdamOptimizer(config.learning_rate)
        
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), config.max_grad_norm)
        self.train_op = optimizer.apply_gradients(
            zip(grads, tvars),
            global_step=tf.train.get_or_create_global_step())
        
        
        
        # 모델(LSTM 네트워크) 생성
    def lstm_cell(self, config):
        # LSTM셀을 생성
        # num_units: 각 Cell 출력 크기
        # forget_bias:  to the biases of the forget gate 
        #              (default: 1)  in order to reduce the scale of forgetting in the beginning of the training.
        # state_is_tuple: True ==> accepted and returned states are 2-tuples of the c_state and m_state.
        # state_is_tuple: False ==> they are concatenated along the column axis.
        cell = tf.contrib.rnn.BasicLSTMCell(num_units=config.rnn_cell_hidden_dim, 
        forget_bias=config.forget_bias, state_is_tuple=True, activation=tf.nn.softsign)

        if config.keep_prob < 1.0:
            cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=config.keep_prob)
        return cell
    

def min_max_scaling(x, min, max):
    np_min = np.array(min)
    np_max = np.array(max)
    
    return (x - np_min) / (np_max - np_min + 1e-7) # 1e-7은 0으로 나누는 오류 예방차원  
    
    
def remake_data(data, seq_length, save_path):
   
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
    
    total_data = []
    
    for prod_d in p_d:
        prod_data = dataset_sort[(dataset_sort.product_no == prod_d)]
        
        del prod_data['date']
        del prod_data['product_no']
        del prod_data['output_o']
                
        np_prod = np.array(prod_data)

        for i in range(len(prod_data) - seq_length):
            #print(len(np_prod))
            #print(np_prod[i:i+seq_length,:-1])
            
            #print(prod_data[i:i+seq_length,:-1].values)
            #print(prod_data[i:i+seq_length,:-1].values.astype(np.float))
    
            r_x = min_max_scaling(np_prod[i:i+seq_length,:-1].astype(np.float), data_min, data_max)
            
            range_y = np_prod[i:i+seq_length,-1].astype(np.int)
            #print(range_y)
            r_y = range_y.reshape(len(range_y),1)
            
            re_data = np.concatenate((r_x, r_y), axis=1)
            
            re_data = re_data.flatten()
            
            total_data.append(re_data)
    
    total_data = np.array(total_data)    
    ######################################################
    
    tmp = collections.Counter(total_data[:,-1]) 
    
    if 1 in tmp.values():
        lower = []
        for c in tmp:
            if tmp[c] == 1:
                lower.append(c)
        click_min = np.array(lower).min()
        total_data[total_data[:,-1] > click_min] = click_min
    ######################################################    
    input_size = len(mm_data[0]) - 1    
    
    va_ratio = 0.2
    label_ind = len(total_data[0]) - 1


    split = StratifiedShuffleSplit(n_splits=1, test_size=va_ratio, random_state=42)
    for train_index, valid_index in split.split(total_data, total_data[:,label_ind]):
        train_set = total_data[train_index]
        valid_set = total_data[valid_index]

    
    train_set = train_set.reshape(len(train_set),seq_length, input_size+1)
    valid_set = valid_set.reshape(len(valid_set),seq_length, input_size+1)
    
    
    train_x = train_set[:,:,:input_size].copy()
    print('train x')
    print(train_x.shape)
    
    train_y = train_set[:,:,input_size].copy()
    train_y = train_y.reshape(len(train_set),seq_length,1)
    
    print('train y')
    print(train_y.shape)
    
    valid_x = valid_set[:,:,:input_size].copy()
    print('valid x')
    print(valid_x.shape)
    
    valid_y = valid_set[:,:,input_size].copy()
    valid_y = valid_y.reshape(len(valid_set),seq_length,1)
    
    print('valid y')
    print(valid_y.shape)
    
    return train_x, train_y, valid_x, valid_y
    
    
def batch_iterator(dataX, dataY, batch_size, num_steps):
    data_len = len(dataY)
    batch_len = int(data_len / batch_size)
    
    if batch_len == 0:
        raise ValueError("epoch_size == 0, decrease batch_size or num_steps")
    
    for i in range(batch_len):
        input_x = dataX[i*batch_size: (i+1)*batch_size]
        input_y = dataY[i*batch_size: (i+1)*batch_size]
        #print(input_x.shape)
        #print(input_y.shape)
        yield (input_x, input_y)
        

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


def run_epoch(session, model, data_x, data_y, model_config, model_outconfig, eval_op=None, verbose=False):
    """Runs the model on the given data."""
    
    costs = 0.0
    iters = 0
    accs = 0.0
    
    matrix_scores = np.zeros([model_config.pattern_size, model_config.pattern_size],dtype=float)
    
    for step, (x, y) in enumerate(batch_iterator(data_x, data_y, model.batch_size, model.seq_length)):
        if eval_op is not None:
            _, _cost, acc, pred = session.run([model.train_op, model.cost, model.accuracy, model.predict], feed_dict={model.X: x, model.targets: y})
        
        else:
            _cost, acc, pred = session.run([model.cost, model.accuracy, model.predict],feed_dict={model.X: x, model.targets: y})
        
        costs += _cost
        iters += 1
        accs += acc

        iter_tagt = np.array(y, dtype=int).flatten()
        iter_pred = np.array(pred).flatten()
        
        for i in range(len(iter_tagt)):
            matrix_scores[iter_pred[i], iter_tagt[i]] += 1.0
        
    batch_loss = costs/iters
    batch_accs = accs/iters
    
    epoch_outconfig = score_calculate(matrix_scores, model_outconfig, verbose)
    
    if verbose == True:
        epoch_outconfig.tr_loss = batch_loss
        epoch_outconfig.tr_accuracy = batch_accs
    else:
        epoch_outconfig.va_loss = batch_loss
        epoch_outconfig.va_accuracy = batch_accs
        
        
    return  epoch_outconfig


    
def LstmClick(data, save_path, config):
    
    start_time = time.time() # 시작시간을 기록한다
    
    train_x, train_y, valid_x, valid_y = remake_data(data, config.seq_length, save_path)
    
    info_path = os.path.join(save_path, "lstm_click_info.txt")
    fid = open(info_path, 'w')
    #########################################################################################
    # info
    fid.write("Hyper parameter\n")
    fid.write("*"*50 + "\n")
    fid.write("input_data_columns  : %d\n" % config.input_data_column_cnt)
    fid.write("output_data_columns : %d\n" % config.output_data_column_cnt)
    fid.write("seq_length          : %d\n" % config.seq_length) 
    fid.write("rnn_cell_hidden_dim : %d\n" % config.rnn_cell_hidden_dim) 
    fid.write("pattern_size        : %d\n" % config.pattern_size) 
    fid.write("forget_bias         : %f\n" % config.forget_bias) 
    fid.write("num_stacked_layers  : %d\n" % config.num_stacked_layers) 
    fid.write("keep_prob           : %f\n" % config.keep_prob) 
    fid.write("batch_size          : %f\n" % config.batch_size) 
    fid.write("epoch_num           : %f\n" % config.epoch_num) 
    fid.write("learning_rate       : %f\n" % config.learning_rate) 
    fid.write("max_grad_norm       : %f\n" % config.max_grad_norm) 
    fid.write("init_scale          : %f\n\n" % config.init_scale) 

    
    with tf.Graph().as_default(), tf.Session() as sess:
        initializer = tf.random_uniform_initializer(-config.init_scale, config.init_scale)
        with tf.variable_scope("model", reuse=None):
            m_tr = LstmOrderModel(is_training=True, config=config)
        with tf.variable_scope("model", reuse=True):
            m_va = LstmOrderModel(is_training=False, config=config)
            
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        
        
        sum_f1 = 0.0
        best_iteration = 0
        for i in range(config.epoch_num):
            output_config = OutputConfig()
            
            epoch_config = run_epoch(sess, m_tr, train_x, train_y, config, output_config, 
                                                                             m_tr.train_op, verbose=True)
            epoch_config = run_epoch(sess, m_va, valid_x, valid_y, config, epoch_config)
            
            print('*'*10, end=' ')
            print('LSTM_click epoch : ', i)
            print('tr_loss : ', epoch_config.tr_loss)
            print('tr_accuracy : ', epoch_config.tr_accuracy)
            print('tr_micro_recall : ', epoch_config.tr_micro_recall)
            print('tr_micro_precision : ', epoch_config.tr_micro_precision)                        
            print('tr_micro_f1 : ', epoch_config.tr_micro_f1)
            
            print('va_loss : ', epoch_config.va_loss)
            print('va_accuracy : ', epoch_config.va_accuracy)
            print('va_micro_recall : ', epoch_config.va_micro_recall)
            print('va_micro_precision : ', epoch_config.va_micro_precision)            
            print('va_micro_f1 : ', epoch_config.va_micro_f1)
            
            if sum_f1 < ((epoch_config.tr_micro_f1 + epoch_config.va_micro_f1)/2):
                sum_f1 = ((epoch_config.tr_micro_f1 + epoch_config.va_micro_f1)/2)
                saver.save(sess, save_path + '/model.ckpt', global_step=i+1)
                
                fid.write("*"*10)
                fid.write("iteration          : %d\n" % i)
                fid.write("tr_loss            : %f\n" % epoch_config.tr_loss)
                fid.write("tr_accuracy        : %f\n" % epoch_config.tr_accuracy)
                fid.write("tr_micro_recall    : %f\n" % epoch_config.tr_micro_recall)
                fid.write("tr_micro_precision : %f\n" % epoch_config.tr_micro_precision)                
                fid.write("tr_micro_f1        : %f\n\n" % epoch_config.tr_micro_f1)
                
                fid.write("va_loss            : %f\n" % epoch_config.va_loss)
                fid.write("va_accuracy        : %f\n" % epoch_config.va_accuracy)
                fid.write("va_micro_recall    : %f\n" % epoch_config.va_micro_recall)
                fid.write("va_micro_precision : %f\n" % epoch_config.va_micro_precision)                
                fid.write("va_micro_f1        : %f\n\n" % epoch_config.va_micro_f1)
                
                epoch_config.best_iteration = i
                
                best_output_config = epoch_config
    
    
    
    end_time = time.time() # 종료시간을 기록한다
    processing_time = end_time - start_time # 경과시간을 구한다
    
    
    m, s = divmod(processing_time, 60)
    h, m = divmod(m, 60)
    fid.write("LSTM click processing time : %d:%02d:%02f" % (h, m, s))
    fid.close()
    
    
    proceessing_time = ('%d:%02d:%02f' % (h,m,s))
    best_output_config.processing_time = proceessing_time
    
    
    return best_output_config
    
    