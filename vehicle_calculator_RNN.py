# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 22:14:33 2018

@author: Notorious-V
"""

import tensorflow as tf
import pandas as pd
import re
import matplotlib.pyplot as plt



# convert categorical variables into one hot encoded system
def encode(series):
  return pd.get_dummies(series.astype(str))

# lstm cell factory, returns one cell per function call
def lstmCell():
  cell = tf.nn.rnn_cell.LSTMCell(num_units=5, state_is_tuple=True, activation='selu')
  return cell

# reset the graph and remove previous global variables
tf.reset_default_graph()

# set the random seed for the graph level
tf.set_random_seed(777)

def main():
  data = pd.read_csv('/content/drive/My Drive/app/CARS.csv')
  #print(data)
  train_x = pd.get_dummies(data['Make'])
  
  values = []
  
  #convert to integer?
  for i, string in enumerate(data['Invoice']):
    values.append(float(re.sub('\$|,','',string)))
  
  data['Invoice2'] = values
  
  train_x = pd.concat([encode(data['Model']), encode(data['Type']), encode(data['Origin']), encode(data['DriveTrain']),data['EngineSize'],data['Invoice2'], 
                       encode(data['Cylinders']), encode(data['Horsepower']), encode(data['MPG_City']), encode(data['MPG_Highway']),encode(data['Weight']), encode(data['Wheelbase']), encode(data['Length'])], axis=1)
  
  values.clear()
  
  #convert to integer?
  for i, string in enumerate(data['MSRP']):
    values.append(float(re.sub('\$|,','',string)))
  
  data['MSRP2'] = values
  
#   train_y = encode(data['MSRP2'])
  train_y = data['MSRP2']
  
  # how many columns
  seq_len = train_x.shape[1]
  data_dim = 1
  output_dim = 1
  learning_rate = 0.001
  epochs = 3
#   n_classes = train_y.shape[1]
  
  X = tf.placeholder('float32', [None,seq_len, data_dim])
  Y = tf.placeholder('float32', [None, 1])
  
   # split into training and testing data
  train_size = 0.5 # 50% training data
  
  train_cnt = floor(train_x.shape[0] * train_size) # floor(428 * .9)
  
  # split by rows
  x_train = train_x.iloc[0:train_cnt].values
  x_train = x_train.reshape(-1,seq_len,1)
  
  y_train = train_y.iloc[0:train_cnt].values
  y_train = y_train.reshape(y_train.shape[0],1)
  
#   x_test = train_x.iloc[train_cnt:].values
#   x_test = x_test.reshape(-1,seq_len,1)

  x_test = train_x.iloc[0:].values
  x_test = x_test.reshape(-1,x_test.shape[1],1) # (428,1072,1)
  
  print(x_test.shape)
  
#   y_test =train_y.iloc[train_cnt:].values
#   y_test = y_test.reshape(y_test.shape[0],1)

  y_test =train_y.iloc[0:].values
  y_test = y_test.reshape(y_test.shape[0],1)
  
  print(y_test.shape)
  
  
  multi_cells = tf.nn.rnn_cell.MultiRNNCell([lstmCell() for _ in range(2)], state_is_tuple=True)
  outputs, _states = tf.nn.dynamic_rnn(multi_cells, inputs=X, dtype=tf.float32)
  
  prediction = tf.contrib.layers.fully_connected(outputs[:, -1], output_dim, activation_fn=None)
  
  loss = tf.reduce_sum(tf.square(prediction - Y))
  optimizer = tf.train.AdamOptimizer(learning_rate)
  train = optimizer.minimize(loss)
  
#   targets = tf.placeholder(tf.float32, [None, n_classes])
  targets = tf.placeholder(tf.float32, [None, 1])
  predictions = tf.placeholder(tf.float32, [None,1])
  
  rmse = tf.sqrt(tf.reduce_mean(tf.square(targets - predictions)))
  
  with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    
    for i in range(epochs):
      #for j in range(25):
      _, step_loss = sess.run([_states, loss], feed_dict={X: [x_train[i]], Y: [y_train[i]]})
      if i % 5 == 0:
        print("[step: {}] loss: {}".format(i, step_loss))
      
  # Test step
    predicted = sess.run(prediction, feed_dict={X: x_test})
    #print(test_predict)
    rmse_val = sess.run(rmse, feed_dict={
                    targets: y_test, predictions: predicted})
    print("RMSE: {}".format(rmse_val))
   
    # Plot predictions
    #plt.plot(y_test)
    plt.plot(predicted)
    plt.xlabel("")
    plt.ylabel("Vehicle Price")
    plt.show()
  
main()