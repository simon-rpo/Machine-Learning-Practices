import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.nn import rnn_cell # rnn_cell module para redes recurrentes para tensor flow
import random
import collections
import time

start_time = time.time()

# Define a log file to sum up our model
# Conveniently, the log will be stored in our data path 
data_path = ""
#writer = tf.summary.FileWriter(data_path)

# Text file containing words for training
training_file = 'C:\\Users\\PC\\Downloads\\test_Conv\\RNN\\dataset\\input_data_4.txt'

# Reading text file
def read_data(fname):
    with open(fname) as f:
        content = f.readlines()
        content = [x.strip() for x in content]
        content = [word for i in range(len(content)) for word in content[i].split()]
        content = np.array(content)
        return content

training_data = read_data(data_path+training_file)
print(training_data)
print("Training data loaded...")