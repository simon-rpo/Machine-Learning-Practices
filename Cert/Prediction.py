import cv2
import h5py
import matplotlib.pyplot as plt
import numpy as np
import skimage.color
import tensorflow as tf
from PIL import Image
from tensorflow.examples.tutorials.mnist import input_data

from animal_main import cnn_model_fn

DATASET_PATH = 'C:\\Users\\PC\\Downloads\\test_Conv\\Cert\\animal_data\\'
CHECKPOINT_PATH = 'C:\\Users\\PC\\Downloads\\test_Conv\\Cert\\animal_data\\tmp'

def resizeImg(data):
    images = []
    for i in data:
        images.append(cv2.resize(i, dsize=(56, 56)))
    return np.asarray(images, dtype=np.float32)

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


checkpointing_config = tf.estimator.RunConfig(
    save_checkpoints_steps=500,
    keep_checkpoint_max=200
)

mnist_classifier = tf.estimator.Estimator(
    model_fn=cnn_model_fn,
    model_dir=CHECKPOINT_PATH,
    config=checkpointing_config
)

predict_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": eval_data},
    num_epochs=1,
    shuffle=False
)


out = mnist_classifier.predict(input_fn=predict_input_fn)
predictions = [gen["classes"] for gen in out]

accuracy = predictions - eval_labels
pred = 1 - np.count_nonzero(accuracy) / len(eval_labels)
print("Accuracy: ", pred)

#Cargo una imagen del dataset para probar
orig_image = train_data[21]
# Test
plt.imshow(orig_image.reshape(56, 56), cmap='gist_gray')
plt.show()
picture = np.asarray(orig_image, dtype=np.float32).reshape(-1, 56, 56, 1)

# Realizo la prediccion
predict_input_fn2 = tf.estimator.inputs.numpy_input_fn(
    x={"x": picture},
    num_epochs=1,
    shuffle=False)


predict_results = mnist_classifier.predict(input_fn=predict_input_fn2)

#Devuelvo un array de 21 salidas con respectiva probabilidad de animal
for x in predict_results:
    print(x)

