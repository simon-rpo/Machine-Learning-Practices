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


# Variables Globales
DATASET_PATH = 'C:\\Users\\PC\\Downloads\\test_Conv\\Cert\\animal_data\\'
CHECKPOINT_PATH = 'C:\\Users\\PC\\Downloads\\test_Conv\\Cert\\animal_data\\tmp'


def cnn_model_fn(features, labels, mode):
    """Model function for CNN."""
    # Input Layer
    # Escalo el tensor de entrada: [batch_size, width, height, channels]
    input_layer = tf.reshape(features["x"], [-1, 56, 56, 1])

    # Convolucion #1
    # 24 Filtros de 3x3, stride 1
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=24,
        kernel_size=[3, 3],
        strides=(1, 1),
        padding="same")

    # Relu #1
    relu1 = tf.nn.relu(conv1)

    # Convolucion #2
    # 24 Filtros de 3x3, stride 1
    conv2 = tf.layers.conv2d(
        inputs=relu1,
        filters=24,
        kernel_size=[3, 3],
        strides=(1, 1),
        padding="same")

    # Relu #2
    relu2 = tf.nn.relu(conv2)

    # Pool #1, Filtro de 2x2 y Stride 2(defecto)
    pool1 = tf.layers.max_pooling2d(inputs=relu2, pool_size=[2, 2], strides=2)

    # Realizo dropout para evitar overfitting.
    dropout1 = tf.layers.dropout(
        inputs=pool1, rate=0.25, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Convolucion #3
    # 24 Filtros de 3x3, stride 1
    conv3 = tf.layers.conv2d(
        inputs=dropout1,
        filters=24,
        kernel_size=[3, 3],
        strides=(1, 1),
        padding="same")

    # Relu #3
    relu3 = tf.nn.relu(conv3)

    # Convolucion #4
    # 24 Filtros de 3x3, stride 1
    conv4 = tf.layers.conv2d(
        inputs=relu3,
        filters=24,
        kernel_size=[3, 3],
        strides=(1, 1),
        padding="same")

    # Relu #4
    relu4 = tf.nn.relu(conv4)

    # Pool #2, Filtro de 2x2 y Stride 2(defecto)
    pool2 = tf.layers.max_pooling2d(inputs=relu4, pool_size=[2, 2], strides=2)

    # Realizo #2 dropout para evitar overfitting.
    dropout2 = tf.layers.dropout(
        inputs=pool2, rate=0.25, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Realizo flatten de la salida
    #flatten = tf.reshape(dropout2, [-1, 14 * 14 * 24])
    flatten = tf.contrib.layers.flatten(dropout2)

    # Realizo capa full connect con un output de 21 clases
    logits = tf.layers.dense(
        inputs=flatten, units=21
    )

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        # Realizo "softmax" para la capa FullConnect
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=21)

    # Perdida con "Entropia Cruzada Softmax"
    loss = tf.losses.softmax_cross_entropy(
        onehot_labels=onehot_labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        # Utilizo Optimizado "Adam" con un aprendizaje de "0.001"
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
    # Cargo el Dataset -> Train + Test
    ftrain = h5py.File(DATASET_PATH + 'train_animals.h5', 'r')
    ftest = h5py.File(DATASET_PATH + 'test_animals.h5', 'r')

    # Divido el dataset en datos de entreno y evalucion con respectivas etiquetas
    train_data, train_labels = ftest['test_set_x'], ftest['test_set_y']
    eval_data, eval_labels = ftrain['train_set_x'],  ftrain['train_set_y']

    train_data = np.asarray(train_data, dtype=np.float32)
    # Normalizo los datos
    train_data = train_data/255.
    # Escalo el tamano de imagen para que quede mas pequeña para entrenamiento
    # Se escala solo 1/4 del tamaño
    train_data = resizeImg(train_data)
    # Convirto a escala de grises (no es necesario el color para el entreno)
    train_data = skimage.color.rgb2gray(train_data)
    # Separo las etiquetas del dataset
    train_labels = np.asarray(train_labels, dtype=np.int32).reshape(420)

    eval_data = np.asarray(eval_data, dtype=np.float32)
    eval_data = eval_data/255.
    eval_data = resizeImg(eval_data)
    eval_data = skimage.color.rgb2gray(eval_data)
    eval_labels = np.asarray(eval_labels, dtype=np.int32).reshape(1680)

    # Para pruebas, solo imprimo una imagen del dataset...
    # plt.imshow(train_data[21])
    # plt.show()

    # Creo el estimador
    # Guardo los checkpoint en la ruta tmp de la carpeta
    mnist_classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn,
        model_dir=CHECKPOINT_PATH)

    # Entreno el modelo
    # Le envio batches de imagenes x 16
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=train_labels,
        batch_size=16,
        num_epochs=50,
        shuffle=True)

    # Evaluacion del Modelo
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data},
        y=eval_labels,
        num_epochs=1,
        shuffle=False)

    #Inicio entrenamiento y envaluacion...
    #Epocas
    for _ in range(50):

        #Entrenamiento...
        mnist_classifier.train(
            input_fn=train_input_fn,
            steps=1000)

        #Evaluacion
        eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
        print(eval_results)


if __name__ == "__main__":
    tf.app.run()
