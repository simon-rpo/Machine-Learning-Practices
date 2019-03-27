from __future__ import absolute_import, division, print_function

import tensorflow as tf
import numpy as np
import mnist_reader
import matplotlib.pyplot as plt
import cv2
from tensorflow.examples.tutorials.mnist import input_data

tf.logging.set_verbosity(tf.logging.INFO)

DATA_DIR = './data/fashion'




def cnn_model_fn(features, labels, mode):
    """Model function for CNN."""
    # Input Layer
    input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])

    # Convolutional Layer #1
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu) 

    # Pooling Layer #1
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    # Convolutional Layer #2 and Pooling Layer #2
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    # Dense Layer
    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
    
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
    
    dropout = tf.layers.dropout(inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Logits Layer
    logits = tf.layers.dense(inputs=dropout, units=10)

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])
    }
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


# Load training and eval data
# ((train_data, train_labels),
#  (eval_data, eval_labels)) = tf.keras.datasets.mnist.load_data()

# train_data = train_data/np.float32(255)
# train_labels = train_labels.astype(np.int32)  # not required

# eval_data = eval_data/np.float32(255)
# eval_labels = eval_labels.astype(np.int32)  # not required


# train_data, train_labels = mnist_reader.load_mnist(
#     'data/fashion', kind='train')
# eval_data, eval_labels = mnist_reader.load_mnist('data/fashion', kind='t10k')

# fashion_mnist = input_data.read_data_sets(
#     'data/fashion', source_url='http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/')
mnist = input_data.read_data_sets(
        DATA_DIR, one_hot=False, validation_size=0)

train = mnist.train
test = mnist.test
train_Labels = mnist.train.labels
test_Labels = mnist.test.labels

train_data = train.images/np.float32(255)
train_data = train_data.reshape(train_data.shape[0], 28, 28)
train_labels = train_Labels.astype(np.int32)  # not required

eval_data = test.images/np.float32(255)
eval_data = eval_data.reshape(eval_data.shape[0], 28, 28)
eval_labels = test_Labels.astype(np.int32)  # not required

# Testing purposes...
plt.imshow(train_data[24], cmap='gist_gray')
plt.show()

# plt.imshow(test_data[28, cmap='gist_gray')
# plt.title(types[str(test.labels[0])])
# plt.show()


# def resize_np(np_array):
#     resized = []
#     for i in list(np_array):
#         larger = cv2.resize(i, (28, 28))
#         resized.append(np.array(larger))
#     return (np.array(resized).astype(np.float32))


# train_data = resize_np(train_data)
# train_labels = train_labels.astype(np.int32)
# eval_data = resize_np(eval_data)
# eval_labels = eval_labels.astype(np.int32)


# # Show images and labels
# plt.imshow(train_data[0].reshape(28,28), cmap='gist_gray')
# #plt.title(types[str(train_labels[0])])
# plt.show()

# plt.imshow(train_data[23].reshape(28,28), cmap='gist_gray')
# #plt.title(types[str(train.labels[23])])
# plt.show()


# Create the Estimator
mnist_classifier = tf.estimator.Estimator(
    model_fn=cnn_model_fn, model_dir="/tmp/testmodel1")


# Set up logging for predictions
tensors_to_log = {"probabilities": "softmax_tensor"}

logging_hook = tf.train.LoggingTensorHook(
    tensors=tensors_to_log, every_n_iter=50)

# Train the model
train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": train_data},
    y=train_labels,
    batch_size=400,
    num_epochs=None,
    shuffle=True)

# train one step and display the probabilties
mnist_classifier.train(
    input_fn=train_input_fn,
    steps=1,
    hooks=[logging_hook])

mnist_classifier.train(input_fn=train_input_fn, steps=2000)


eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": eval_data},
    y=eval_labels,
    num_epochs=1,
    shuffle=False)

eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
print(eval_results)
