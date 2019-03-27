import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

DATA_DIR = './data/fashion'
#DATA_DIR = 'C:\\Users\\PC\\Downloads\\test_Conv\\data\\fashion'
mnist = input_data.read_data_sets(
    DATA_DIR, one_hot=False, validation_size=0)

train_data = mnist.train.images  # Returns np.array
train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
# train_data, train_labels = train_data, train_labels
# eval_data = mnist.test.images  # Returns np.array
# eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)
# eval_data, eval_labels = eval_data, eval_labels


def display_image(num):
    print(train_labels[num])
    label = train_labels[num].argmax(axis=0)
    image = train_data[num].reshape([28, 28])
    plt.title('Example: %d label: %d' % (num, label))
    plt.imshow(image, cmap=plt.get_cmap('gray_r'))
    plt.show()


for i in range(5):
    display_image(i)
