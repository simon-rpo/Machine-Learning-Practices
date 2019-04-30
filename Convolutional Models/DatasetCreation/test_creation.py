# -*- coding: utf-8 -*-
"""

@author: Simon Restrepo
    Creates a dataset of shoes gathering the data across the top 5 sites
    for buying shoes in USA.
"""
import glob
import os

import cv2
import h5py
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.utils import shuffle

OUTPUT_PATH = 'C:\\Users\\PC\\Downloads\\test_Conv\\Convolutional Models\\DatasetCreation\\data_shoes\\new_set\\'
parent_dir = 'C:\\Users\\PC\\Downloads\\test_Conv\\Convolutional Models\\DatasetCreation\\data_shoes\\new_set\\output\\train'

file_ext = '*'
classes_id = {
    'sandals': 0,
    'sneakers': 1,
    'high_heels': 2,
    'boots': 3
}


def label2num(x):
    if x in classes_id:
        return classes_id[x]
    else:
        return(21)


def extract_features(parent_dir, file_ext, b):
    START_BATCH = b
    imgs = []
    labels = []
    for sub_dir in os.listdir(parent_dir):
        print('Creating "' + sub_dir + '"...')
        i = 0
        for fn in glob.glob(os.path.join(parent_dir, sub_dir, file_ext))[START_BATCH:]:
            # print(fn)
            try:
                img = Image.open(fn)
                # if (img.size[0] >= 224):
                etiqueta = label2num(sub_dir)
                #img = img.resize((299, 299), Image.ANTIALIAS)
                # plt.imshow(img)
                # plt.show()
                im = np.array(img)
                if im.shape[2] == 3:
                    imgs.append(im)
                    labels.append(etiqueta)
                else:
                    im = cv2.cvtColor(im, cv2.COLOR_BGRA2BGR)
                    imgs.append(im)
                    labels.append(etiqueta)
            except Exception as e:
                print("type error: " + str(e) + " image: " + fn)

            if i == BATCH_START:
                break
            i += 1

    features = np.asarray(imgs).reshape(len(imgs), 299, 299, 3)
    return features, np.array(labels, dtype=np.int)


i = 0
BATCH_START = 800
batch = 0
t = int(len(glob.glob(os.path.join(parent_dir, 'sneakers', file_ext))) / BATCH_START)

for i in range(1, t):
    features, labels = extract_features(parent_dir, file_ext, batch)
    # Si se requiere se hace alguno de los siguientes procesos.
    # Mesclar aleatoriamente la base de datos
    x1, y1 = shuffle(features, labels)

    # O Separarla en entrenamieto y prueba (o si es necesario en validación)
    samples = y1.size
    y1 = y1.reshape((samples, 1))

    offset = int(x1.shape[0] * 0.80)
    X_train, Y_train = x1[:offset], y1[:offset]
    X_test, Y_test = x1[offset:], y1[offset:]
    Y_test = np.array(Y_test)
    Y_train = np.array(Y_train)
    # Se puede adicionar el proceso que requieran para su base de datos

    # Este sería el proceso para guardar en un formato h5 los datos
    with h5py.File(OUTPUT_PATH + 'train_dataset_'+str(i)+'_.h5', 'w') as h5data:
        h5data.create_dataset('train_set_x', data=X_train)
        h5data.create_dataset('train_set_y', data=Y_train)
    with h5py.File(OUTPUT_PATH + 'test_dataset_'+str(i)+'_.h5', 'w') as h5data:
        h5data.create_dataset('test_set_x', data=X_test)
        h5data.create_dataset('test_set_y', data=Y_test)
    print('Dataset created Successfully #'+str(i))
    batch += BATCH_START

##############################################################
##############################################################
