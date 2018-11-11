# imports
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
import seaborn as sns
from math import floor, ceil
from pylab import rcParams
​
%matplotlib inline
[ ]

# set styles for plots and graphs
​
sns.set(style='ticks', palette='Spectral', font_scale=1.5)
​
palette = ['#4caf50', '#2196f3', '#9e9e9e', '#ff9800', '#607d8b', '#9c27b0']
sns.set_palette(palette)
rcParams['figure.figsize'] = 16, 8
​
plt.xkcd()
random_state = 42
np.random.seed(random_state)
tf.set_random_seed(random_state)
[ ]

def encode(series):
  return pd.get_dummies(series.astype(str))
[ ]

def multilayer_Rnn(x, weights, biases, keep_prob):
  layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
  layer_1 = tf.nn.relu(layer_1)
  layer_1 = tf.nn.dropout(layer_1, keep_prob)
  out_layer = tf.matmul(layer_1, weights['out']) + biases['out']
  return out_layer
[ ]

def main():
  
  # read in data csv file
  data = pd.read_csv('CARS.csv')
  
  #fill in any missing data
  #print(data.describe())
  #print(data.shape)
  #https://medium.com/@curiousily/tensorflow-for-hackers-part-ii-building-simple-neural-network-2d6779d2f91b
  
  #print(data.shape)
  
  #data['EngineSize'].plot(kind='bar', rot=0)
  
  # one hot encoding
  train_x = pd.get_dummies(data['Make'])
  #train_x['Model'] = data['Model']
  #train_x['Type'] = data['Type']
  #train_x['Origin'] = data['Origin']
  #train_x['DriveTrain'] = data['DriveTrain']
  
  train_x = pd.concat([encode(data['Model']), encode(data['Type']), encode(data['Origin']), encode(data['DriveTrain']),data['EngineSize'], 
                       data['Cylinders'], data['Horsepower'], data['MPG_City'], data['MPG_Highway'],data['Weight'], data['Wheelbase'], data['Length']], axis=1)
  
  #print(train_x)
  
  values = []
  #convert to integer?
  for i, string in enumerate(data['MSRP']):
    values.append(int(re.sub('\$|,','',string)))
  
  data['MSRP2'] = values
  #print(data['MSRP2'].dtype)
  
  train_y = data['MSRP2']
  
  #print(train_y.shape)
  
  n_hidden_1 = 38
  n_classes = train_x.shape[1]
  
  n_input = train_x.shape[1] # 445 column inputs
  n_classes = train_y.shape[0] # 428 rows
  
  weights = { 'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])), 'out': tf.Variable(tf.random_normal([n_hidden_1,n_classes]))}
  biases = {'b1': tf.Variable(tf.random_normal([n_hidden_1])), 'out': tf.Variable(tf.random_normal([n_classes]))}
  
  keep_prob = tf.placeholder('float')
  
  training_epochs = 1 #5000
  display_step = 1000
  batch_size = 10
  
  x = tf.placeholder('float', [None, n_input])
  y = tf.placeholder('float', [None])
  
  # split into training and testing data
  train_size = 0.9
  
  train_cnt = floor(train_x.shape[0] * train_size)
  x_train = train_x.iloc[0:train_cnt].values
  y_train = train_y.iloc[0:train_cnt].values
  
  x_test = train_x.iloc[train_cnt:].values
  y_test =train_y.iloc[train_cnt:].values
  
  predictions = multilayer_Rnn(x, weights, biases, keep_prob)
  
  cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=predictions, labels=y))
  
  optimize = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost)
  
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for epoch in range(training_epochs):
        avg_cost = 0.0
        total_batch = 5 #int(len(x_train) / batch_size)
      
        x_batches = np.array_split(x_train, total_batch)
        print(x_batches[1].shape)
        y_batches = np.array_split(y_train, total_batch)
        print(y_batches[0].shape)
        
        # issue with cost function labels
        for i in range(total_batch):
            batch_x, batch_y = x_batches[i], y_batches[i]
            #print(batch_x.shape)
            _, c = sess.run([optimize, cost], 
                            feed_dict={
                                x: batch_x, 
                                y: batch_y, 
                                keep_prob: 0.8
                            })
            avg_cost += c / total_batch
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", \
                "{:.9f}".format(avg_cost))
    print("Optimization Finished!")
    correct_prediction = tf.equal(tf.argmax(predictions, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("Accuracy:", accuracy.eval({x: x_test, y: y_test, keep_prob: 1.0}))
        
  
  
  #print(x_train.shape)
  #print(y_train.shape)
  #print(train_x.shape)
  #print(train_x)
  #print(train_y.shape)
main()