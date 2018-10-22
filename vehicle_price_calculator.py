# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 17:01:28 2018
@author: Notorious-V
"""

import tensorflow as tf
#import csv
#import pandas as pd
import numpy as np

def input_csv():
    CSV_PATH = './CARS.csv'
    dataset = tf.contrib.data.make_csv_dataset(CSV_PATH, batch_size=100, header=True)
    
    return dataset

#create rnn model
def rnn_model(data, num_hidden, num_labels):
    
    cell_fw = tf.nn.rnn_cell.RNNCell
    cell_bw = tf.nn.rnn_cell.RNNCell
    
    tf.contrib.rnn.stack_bidirectional_dynamic_rnn(cell_fw, cell_bw, data)
    '''outputs, current_state = tf.nn.dynamic_rnn(cell, data, dtype=tf.float32)
    output = outputs[-1]
    
    w_softmax = tf.Variable(tf.truncated_normal([num_hidden, num_labels]))
    b_softmax = tf.Variable(tf.random_normal([num_labels]))
    logit = tf.matmul(output, w_softmax) + b_softmax
    print(output)
    return logit'''
    
def main():
    
    '''#set display options
    pd.set_option('display.max_colwidth', 100)
    
    #read in data file
    data = pd.read_csv('CARS.csv')
    df = pd.DataFrame(data)
    
    #print(df)
    # list of vehicle makes
    vehicle_Make = list(set(df['Make']))
    #print(len(vehicle_Make))
    #print(vehicle_Make)'''
    
    #read csv file, create an input pipeline
    dataset = input_csv()
   
    #print(dataset.batch(100))
    '''x = np.random.sample((100,2))
    
    dataset = tf.data.Dataset.from_tensor_slices(x)
    
    iterator = dataset.make_one_shot_iterator()
    my_data = iterator.get_next()
    
    with tf.Session() as sess:
        print(sess.run(my_data))'''

    # use a placeholder
    x = tf.placeholder(tf.float32, shape=[None,2])
    
    # create random float sampling
    data = np.random.sample((100,2))
    
    #define an interator instance
    iterator = dataset.make_initializable_iterator()
    
    #get the next values
    iterate = iterator.get_next()
    
    with tf.Session() as sess:
        sess.run(iterator.initializer, feed_dict={ x: data})
        print(sess.run(iterate))
        #print(rnn_model(sess.run(iterate), 100,15))
    
main()