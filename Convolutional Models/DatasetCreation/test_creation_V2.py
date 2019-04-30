from glob import glob
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from keras.applications.inception_v3 import preprocess_input

WIDTH = 299
HEIGHT = 299
BATCH_SIZE = 4000
PATH = 'C:\\Users\\PC\\Downloads\\test_Conv\\Convolutional Models\\DatasetCreation\\data_shoes\\new_set\\train'
OUTPUT_TRAIN = 'C:\\Users\\PC\\Downloads\\test_Conv\\Convolutional Models\\DatasetCreation\\data_shoes\\new_set\\output\\train\\sandals'

# sneakers = glob(PATH + 'train\\\\sneakers\\\\*.jpg')
# heels = glob(PATH + 'train\\\\high_heels\\\\*.jpg')
# sandals = glob(PATH + 'train\\\\sandals\\\\*.jpg')

# sneakers_train, sneakers_test = train_test_split(sneakers, test_size=0.30)
# heels_train, heels_test = train_test_split(heels, test_size=0.30)
# sandals_train, sandals_test = train_test_split(sandals, test_size=0.30)


# data prep
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

train_generator = train_datagen.flow_from_directory(
    PATH,
    target_size=(HEIGHT, WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    save_to_dir=OUTPUT_TRAIN,
    save_format='jpg')

next(train_generator)
next(train_generator)
next(train_generator)
