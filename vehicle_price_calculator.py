# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 17:01:28 2018

@author: Notorious-V
"""

import tensorflow as tf
import csv
import pandas as pd
import numpy as np

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
    #print(vehicle_Make)
    
    #list of models
    vehicle_Models = list(set(df['Model']))
    #print(vehicle_Models)
    
    # list of vehicle types
    vehicle_Types = list(set(df['Type']))
    #print(vehicle_Types)
    
    # list of vehicle origin
    vehicle_Origin = list(set(df['Origin']))
    #print(vehicle_Origin)
    
    # List of engine sizes
    vehicle_eng = list(set(df['EngineSize']))
    print(vehicle_eng)
    # lets create at tensor of string type (rank 1 tensor)'''
    
    #read csv file, create an input pipeline
    CSV_PATH = './CARS.csv'
    dataset = tf.contrib.data.make_csv_dataset(CSV_PATH, batch_size=100, header= True)
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
    
    
main()