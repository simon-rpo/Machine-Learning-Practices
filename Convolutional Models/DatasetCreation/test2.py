import cv2
import h5py
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import skimage.color
from PIL import Image

from test1 import cnn_model_fn

DATA_DIR = 'C:\\Users\\PC\\Downloads\\test_Conv\\Convolutional Models\\DatasetCreation\\data_shoes\\new_set\\'

ftrain = h5py.File(DATA_DIR + 'train_dataset.h5', 'r')
ftest = h5py.File(DATA_DIR + 'test_dataset.h5', 'r')

train_data, train_labels = ftest['test_set_x'], ftest['test_set_y']
eval_data, eval_labels = ftrain['train_set_x'],  ftrain['train_set_y']

train_data = np.asarray(train_data, dtype=np.float32)
train_data = train_data/255.
train_data = skimage.color.rgb2gray(train_data)
train_labels = np.asarray(
    train_labels, dtype=np.int32).reshape(train_labels.shape[0])

eval_data = np.asarray(eval_data, dtype=np.float32)
eval_data = eval_data/255.
eval_data = skimage.color.rgb2gray(eval_data)
eval_labels = np.asarray(
    eval_labels, dtype=np.int32).reshape(eval_labels.shape[0])


checkpointing_config = tf.estimator.RunConfig(
    save_checkpoints_steps=500,
    keep_checkpoint_max=200
)

mnist_classifier = tf.estimator.Estimator(
    model_fn=cnn_model_fn,
    model_dir="C:\\Users\\PC\\Downloads\\test_Conv\\Convolutional Models\\DatasetCreation\\tmp",
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


# # ### load self made picture

# orig_image = cv2.imread(
#     'C:\\Users\\PC\\Downloads\\test_Conv\\Convolutional Models\\DatasetCreation\\t_images\\t_sneaker4.jpg', 0)
# # Test
# orig_image = orig_image/np.float(255)
# orig_image = skimage.color.rgb2gray(orig_image)
# plt.imshow(orig_image, cmap='gist_gray')
# plt.show()
# orig_image = cv2.resize(orig_image, (64, 64))
# plt.imshow(orig_image, cmap='gist_gray')
# plt.show()
# picture = np.asarray(orig_image, dtype=np.float32)


orig_image = eval_data[86]
# Test
plt.imshow(orig_image.reshape(64, 64), cmap='gist_gray')
plt.show()
picture = np.asarray(orig_image, dtype=np.float32).reshape(-1, 64, 64, 1)


predict_input_fn2 = tf.estimator.inputs.numpy_input_fn(
    x={"x": picture},
    num_epochs=1,
    shuffle=False)


predict_results = mnist_classifier.predict(input_fn=predict_input_fn2)
# prediction = [gen["probabilities"] for gen in predict_results]
# print(prediction)
for x in predict_results:
    # cls.append(x['classes'])
    # probs.append(x['probabilities'])
    print(x)