import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from PIL import Image

from Convolutional_Model_Rev2 import cnn_model_fn

DATA_DIR = 'C:\\Users\\PC\\Downloads\\test_Conv\\data\\fashion'
mnist = input_data.read_data_sets(DATA_DIR, one_hot=False, validation_size=0)

checkpointing_config = tf.estimator.RunConfig(
    save_checkpoints_steps=500,
    keep_checkpoint_max=200
)

mnist_classifier = tf.estimator.Estimator(
    model_fn=cnn_model_fn,
    model_dir="C:\\Users\\PC\\Downloads\\test_Conv\\tmp",
    config=checkpointing_config
)

predict_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": mnist.test.images},
    num_epochs=1,
    shuffle=False
)


out = mnist_classifier.predict(input_fn=predict_input_fn)
predictions = [gen["classes"] for gen in out]

accuracy = predictions - mnist.test.labels
pred = 1 - np.count_nonzero(accuracy) / len(mnist.test.labels)
print("Accuracy: ", pred)


# # ### load self made picture

# #orig_image = cv2.imread('sneaker.jpg', 0)
orig_image = mnist.train.images[86]
# Test
plt.imshow(orig_image.reshape(28, 28), cmap='gist_gray')
plt.show()
picture = np.asarray(orig_image, dtype=np.float32).reshape(-1, 28, 28, 1)

# img_output = cv2.resize(orig_image, (28, 28))
# # Test
# # plt.imshow(img_output)
# # plt.show()
# picture = np.array(img_output)/np.float32(255)
# #picture = img_output

# img = Image.open("1.png").convert('L')
# picture = np.asarray(img.getdata(), dtype=np.float32).reshape(-1, 28, 28, 1)


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


# 7 - sneaker
