# Lastest Version
from __future__ import absolute_import, division, print_function

import cv2
import h5py
import matplotlib.pyplot as plt
import numpy as np
import skimage.color
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

tf.logging.set_verbosity(tf.logging.INFO)

DATA_DIR = 'C:\\Users\\PC\\Downloads\\test_Conv\\data\\animal'


def cnn_model_fn(features, labels, mode):
    """Model function for CNN."""
    # Input Layer
    # Reshape X to 4-D tensor: [batch_size, width, height, channels]
    # MNIST images are 28x28 pixels, and have one color channel
    input_layer = tf.reshape(features["x"], [-1, 56, 56, 1])

    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=16,
        kernel_size=[3, 3],
        padding="same")

    relu1 = tf.nn.relu(conv1)

    conv2 = tf.layers.conv2d(
        inputs=relu1,
        filters=16,
        kernel_size=[3, 3],
        padding="same")

    relu2 = tf.nn.relu(conv2)

    conv3 = tf.layers.conv2d(
        inputs=relu2,
        filters=16,
        kernel_size=[3, 3],
        padding="same")

    relu3 = tf.nn.relu(conv3)

    pool1 = tf.layers.max_pooling2d(inputs=relu3, pool_size=[2, 2], strides=2)

    dropout1 = tf.layers.dropout(
        inputs=pool1, rate=0.25, training=mode == tf.estimator.ModeKeys.TRAIN)

    conv4 = tf.layers.conv2d(
        inputs=dropout1,
        filters=16,
        kernel_size=[3, 3],
        padding="same")

    relu4 = tf.nn.relu(conv4)

    conv5 = tf.layers.conv2d(
        inputs=relu4,
        filters=16,
        kernel_size=[3, 3],
        padding="same")

    relu5 = tf.nn.relu(conv5)

    conv6 = tf.layers.conv2d(
        inputs=relu5,
        filters=16,
        kernel_size=[3, 3],
        padding="same")

    relu6 = tf.nn.relu(conv6)

    pool2 = tf.layers.max_pooling2d(inputs=relu6, pool_size=[2, 2], strides=2)

    dropout2 = tf.layers.dropout(
        inputs=pool2, rate=0.25, training=mode == tf.estimator.ModeKeys.TRAIN)

    flatten = tf.reshape(dropout2, [-1, 14 * 14 * 16])

    dropout3 = tf.layers.dropout(
        inputs=flatten, rate=0.3, training=mode == tf.estimator.ModeKeys.TRAIN)

    logits = tf.layers.dense(
        inputs=dropout3, units=21
    )

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
    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=21)
    loss = tf.losses.softmax_cross_entropy(
        onehot_labels=onehot_labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def resizeImg(data):
    images = []
    for i in data:
        images.append(cv2.resize(i, dsize=(56, 56)))
    return np.asarray(images, dtype=np.float32)


def main(unused_argv):
    # Load training and eval data
    ftrain = h5py.File(
        'C:\\Users\\PC\\Downloads\\test_Conv\\animal_data\\train_animals.h5', 'r')
    ftest = h5py.File(
        'C:\\Users\\PC\\Downloads\\test_Conv\\animal_data\\test_animals.h5', 'r')

    train_data, train_labels = ftest['test_set_x'], ftest['test_set_y']
    eval_data, eval_labels = ftrain['train_set_x'],  ftrain['train_set_y']

    train_data = np.asarray(train_data, dtype=np.float32)
    train_data = train_data/255.
    train_data = resizeImg(train_data)
    train_data = skimage.color.rgb2gray(train_data)
    train_labels = np.asarray(train_labels, dtype=np.int32).reshape(420)

    eval_data = np.asarray(eval_data, dtype=np.float32)
    eval_data = eval_data/255.
    eval_data = resizeImg(eval_data)
    eval_data = skimage.color.rgb2gray(eval_data)
    eval_labels = np.asarray(eval_labels, dtype=np.int32).reshape(1680)

    # Testing purposes...
    plt.imshow(train_data[21])
    plt.show()

    # Create the Estimator
    mnist_classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn,
        model_dir="C:\\Users\\PC\\Downloads\\test_Conv\\animal_data\\tmp")

    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=train_labels,
        batch_size=32,
        num_epochs=50,
        shuffle=True)

    # Evaluate the model and print results
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data},
        y=eval_labels,
        num_epochs=1,
        shuffle=False)

    for _ in range(50):
        mnist_classifier.train(
            input_fn=train_input_fn,
            steps=1000)

        eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
        print(eval_results)


if __name__ == "__main__":
    tf.app.run()
