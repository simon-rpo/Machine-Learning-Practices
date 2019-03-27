import tensorflow as tf
import numpy as np
import mnist_reader

# Load training and eval data

# train_labels = train_labels.astype(np.int32)  # not required

# eval_data = eval_data/np.float32(255)
# eval_labels = eval_labels.astype(np.int32)  # not required

##################################################

X_train, y_train = mnist_reader.load_mnist('data/fashion', kind='train')
X_test, y_test = mnist_reader.load_mnist('data/fashion', kind='t10k')


((train_data, train_labels),
 (eval_data, eval_labels)) = tf.keras.datasets.mnist.load_data()

train_data = train_data/np.float32(255)
train_data2 = X_train/np.float32(255)
# X_test, y_test = mnist_reader.load_mnist('data/fashion', kind='t10k')
