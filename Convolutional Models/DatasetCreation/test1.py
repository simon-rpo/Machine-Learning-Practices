# Lastest Version
from __future__ import absolute_import, division, print_function

import cv2
import h5py
import matplotlib.pyplot as plt
import numpy as np
import skimage.color
import tensorflow as tf
import tensorflow_hub as hub

tf.logging.set_verbosity(tf.logging.INFO)

DATA_DIR = 'C:\\Users\\PC\\Downloads\\test_Conv\\Convolutional Models\\DatasetCreation\\data_shoes\\new_set\\'


def cnn_model_fn(features, labels, mode):
     # Load Inception-v3 model.
    module = hub.Module(
        "https://tfhub.dev/google/imagenet/inception_v3/feature_vector/1")
    input_layer = adjust_image(features["x"])
    outputs = module(input_layer)

    logits = tf.layers.dense(inputs=outputs, units=10)

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
            labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def adjust_image(data):
    # Reshaped to [batch, height, width, channels].
    imgs = tf.reshape(data, [-1, 28, 28, 1])
    # Adjust image size to that in Inception-v3 input.
    imgs = tf.image.resize_images(imgs, (299, 299))
    # Convert to RGB image.
    imgs = tf.image.grayscale_to_rgb(imgs)
    return imgs


def main(unused_argv):
    with tf.Graph().as_default() as g:
            # Load training and eval data
        ftrain = h5py.File(DATA_DIR + 'train_dataset.h5', 'r')
        ftest = h5py.File(DATA_DIR + 'test_dataset.h5', 'r')

        train_data, train_labels = ftest['test_set_x'], ftest['test_set_y']
        eval_data, eval_labels = ftrain['train_set_x'],  ftrain['train_set_y']

        train_data = np.asarray(train_data, dtype=np.float32)
        # train_data = train_data/255.
        # train_data = skimage.color.rgb2gray(train_data)
        train_labels = np.asarray(
            train_labels, dtype=np.int32).reshape(train_labels.shape[0])

        eval_data = np.asarray(eval_data, dtype=np.float32)
        # eval_data = eval_data/255.
        # eval_data = skimage.color.rgb2gray(eval_data)
        eval_labels = np.asarray(
            eval_labels, dtype=np.int32).reshape(eval_labels.shape[0])

        # Testing purposes...
        plt.imshow(train_data[21])
        plt.show()

        # Create the Estimator
        mnist_classifier = tf.estimator.Estimator(
            model_fn=cnn_model_fn,
            model_dir="C:\\Users\\PC\\Downloads\\test_Conv\\Convolutional Models\\DatasetCreation\\tmp")

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
