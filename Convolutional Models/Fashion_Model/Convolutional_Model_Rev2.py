# Lastest Version
from __future__ import absolute_import, division, print_function

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

tf.logging.set_verbosity(tf.logging.INFO)

DATA_DIR = 'C:\\Users\\PC\\Downloads\\test_Conv\\data\\fashion'


def cnn_model_fn(features, labels, mode):
    """Model function for CNN."""
    # Input Layer
    # Reshape X to 4-D tensor: [batch_size, width, height, channels]
    # MNIST images are 28x28 pixels, and have one color channel
    input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])

    # Convolutional Layer #1
    # Computes 32 features using a 3x3 filter with ReLU activation.
    # Padding is added to preserve width and height.
    # Input Tensor Shape: [batch_size, 28, 28, 1]
    # Output Tensor Shape: [batch_size, 28, 28, 32]
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu)

    # Pooling Layer #1
    # First max pooling layer with a 2x2 filter and stride of 2
    # Input Tensor Shape: [batch_size, 28, 28, 32]
    # Output Tensor Shape: [batch_size, 14, 14, 32]
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[3, 3], strides=3)

    # Dropout Layer #1
    # First dropout to prevent overfitting, 0.25 probability that element will be kept.
    dropout1 = tf.layers.dropout(
        inputs=pool1, rate=0.25, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Convolutional Layer #2
    # Computes 64 features using a 3x3 filter with ReLU activation.
    # Padding is added to preserve width and height.
    # Input Tensor Shape: [batch_size, 14, 14, 32]
    # Output Tensor Shape: [batch_size, 14, 14, 64]
    conv2 = tf.layers.conv2d(
        inputs=dropout1,
        filters=64,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu)

    # Pooling Layer #2
    # First max pooling layer with a 2x2 filter and stride of 2
    # Input Tensor Shape: [batch_size, 14, 14, 64]
    # Output Tensor Shape: [batch_size, 7, 7, 64]
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    # Dropout Layer #2
    # Second dropout to prevent overfitting, 0.25 probability that element will be kept
    dropout2 = tf.layers.dropout(
        inputs=pool2, rate=0.25, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Convolutional Layer #3
    # Computes 128 features using a 3x3 filter.
    # Padding is added to preserve width and height.
    # Input Tensor Shape: [batch_size, 7, 7, 64]
    # Output Tensor Shape: [batch_size, 7, 7, 128]
    conv3 = tf.layers.conv2d(
        inputs=dropout2,
        filters=128,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu)

    # Dropout Layer #2
    # First dropout to prevent overfitting, 0.4 probability that element will be kept.
    dropout3 = tf.layers.dropout(
        inputs=conv3, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Flatten tensor into a batch of vectors
    # Input Tensor Shape: [batch_size, 7, 7, 128]
    # Output Tensor Shape: [batch_size, 6272]
    flatten = tf.reshape(dropout3, [-1, 7 * 7 * 128])

    # Dense Layer
    # Densely connected layer with 128 neurons
    # Input Tensor Shape: [batch_size, 7 * 7 * 128]
    # Output Tensor Shape: [batch_size, 128]
    dense = tf.layers.dense(
        inputs=flatten, units=128, activation=tf.nn.relu)

    # Dropout Layer #2
    # First dropout to prevent overfitting
    dropout4 = tf.layers.dropout(
        inputs=dense, rate=0.3, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Logits layer
    # Input Tensor Shape: [batch_size, 128]
    # Output Tensor Shape: [batch_size, 10]
    logits = tf.layers.dense(
        inputs=dropout4, units=10
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
    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=10)
    loss = tf.losses.softmax_cross_entropy(
        onehot_labels=onehot_labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
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


def main(unused_argv):
    # Load training and eval data
    mnist = input_data.read_data_sets(
        DATA_DIR, one_hot=False, validation_size=0)
    train_data = mnist.train.images  # Returns np.array
    train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    train_data, train_labels = train_data, train_labels
    eval_data = mnist.test.images  # Returns np.array
    eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)
    eval_data, eval_labels = eval_data, eval_labels

    # # Testing purposes...
    # plt.imshow(train_data[83].reshape(28, 28), cmap='gist_gray')
    # plt.show()

    # Create the Estimator
    mnist_classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn, model_dir="C:\\Users\\PC\\Downloads\\test_Conv\\tmp")

    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=train_labels,
        batch_size=128,
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
