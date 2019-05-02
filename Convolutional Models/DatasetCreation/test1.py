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
MODEL_DIR = "C:\\Users\\PC\\Downloads\\test_Conv\\Convolutional Models\\DatasetCreation\\tmp"


def cnn_model_fn(features, labels, mode):
    # Load Inception-v3 model.
    module = hub.Module(
        "https://tfhub.dev/google/imagenet/inception_v3/classification/1")

    input_layer = tf.reshape(features["x"], [-1, 299, 299, 3])

    # outputs = module(input_layer)
    outputs = module(dict(images=input_layer),
                     signature="image_classification",
                     as_dict=True)

    # print(outputs.items())

    middle_output = outputs["InceptionV3/Mixed_7c"]

    avgPool = tf.layers.AveragePooling2D()(middle_output)

    bn1 = tf.layers.batch_normalization(
        avgPool, training=mode == tf.estimator.ModeKeys.TRAIN)

    logits1 = tf.layers.dense(inputs=avgPool, units=128,
                              activation=tf.nn.relu)

    dropout1 = tf.layers.dropout(
        inputs=logits1, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN)

    logits2 = tf.layers.dense(inputs=dropout1, units=64,
                              activation=tf.nn.relu)

    dropout2 = tf.layers.dropout(
        inputs=logits2, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN)

    logits = tf.layers.dense(inputs=dropout2, units=3)

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
    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=3)
    loss = tf.losses.softmax_cross_entropy(
        onehot_labels=onehot_labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())

        summary_hook = tf.train.SummarySaverHook(
            100,
            output_dir=MODEL_DIR,
            summary_op=tf.summary.merge_all())

        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op,
                                          training_hooks=[summary_hook])

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def adjust_image(data):
    # Reshaped to [batch, height, width, channels].
    imgs = tf.reshape(data, [-1, 299, 299, 3])
    # Adjust image size to that in Inception-v3 input.
    imgs = tf.image.resize_images(imgs, (299, 299))
    # Convert to RGB image.
    imgs = tf.image.grayscale_to_rgb(imgs)
    return imgs


def main(unused_argv):
    with tf.Graph().as_default() as g:
            # Load training and eval data
        ftrain = h5py.File(DATA_DIR + 'train_dataset_1_.h5', 'r')
        ftest = h5py.File(DATA_DIR + 'test_dataset_1_.h5', 'r')

        train_data, train_labels = ftrain['train_set_x'],  ftrain['train_set_y']
        eval_data, eval_labels = ftest['test_set_x'], ftest['test_set_y']

        train_data = np.asarray(train_data, dtype=np.float32)
        train_data = train_data/255.
        # train_data = skimage.color.rgb2gray(train_data)
        train_labels = np.asarray(
            train_labels, dtype=np.int32).reshape(train_labels.shape[0])

        eval_data = np.asarray(eval_data, dtype=np.float32)
        eval_data = eval_data/255.
        # eval_data = skimage.color.rgb2gray(eval_data)
        eval_labels = np.asarray(
            eval_labels, dtype=np.int32).reshape(eval_labels.shape[0])

        # Testing purposes...
        plt.imshow(train_data[21])
        plt.show()

        # Create the Estimator
        mnist_classifier = tf.estimator.Estimator(
            model_fn=cnn_model_fn,
            model_dir=MODEL_DIR)

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
