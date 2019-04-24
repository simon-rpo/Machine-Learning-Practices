import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.layers import Dense, Dropout, GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing import image

MODEL_FILE = 'C:\\Users\\PC\\Downloads\\test_Conv\\Convolutional Models\\DatasetCreation\\filename.model'
IMAGES_PATH = 'C:\\Users\\PC\\Downloads\\test_Conv\\Convolutional Models\\DatasetCreation\\t_images\\'
WIDTH = 224
HEIGHT = 224

CLASSES = 3

def predict(model, img):
    """Run model prediction on image
    Args:
        model: keras model
        img: PIL format image
    Returns:
        list of predicted labels and their probabilities 
    """
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = model.predict(x)
    return preds[0]


def plot_preds(img, preds):
    """Displays image and the top-n predicted probabilities in a bar graph
    Args:
        preds: list of predicted labels and their probabilities
    """
    labels = ("high_heels", "sandals", "sneaker")
    gs = gridspec.GridSpec(2, 1, height_ratios=[4, 1])
    plt.figure(figsize=(8, 8))
    plt.subplot(gs[0])
    plt.imshow(np.asarray(img))
    plt.subplot(gs[1])
    plt.barh([0, 1, 2], preds, alpha=0.5)
    plt.yticks([0, 1, 2], labels)
    plt.xlabel('Probability')
    plt.xlim(0, 1)
    plt.tight_layout()


model = Model(MODEL_FILE)

img = image.load_img(IMAGES_PATH + 't_sneaker.jpg',
                     target_size=(HEIGHT, WIDTH))
preds = predict(model, img)

plot_preds(np.asarray(img), preds)
plt.show()
print(preds)
