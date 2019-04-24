import cv2
from PIL import Image
import matplotlib.pyplot as plt
import h5py

# img = Image.open(
#     'C:\\Users\\PC\\Downloads\\test_Conv\\Convolutional Models\\DatasetCreation\\data_sneakers\\Nike\\Nike Air Bakin_ Posite _ Men_s.png')
# img = cv2.imdecode(img,-1)
# plt.imshow(img)
# plt.show()

# xxx = cv2.cvtColor(cv2.UMat(img), cv2.COLOR_RGBA2BGR)

# plt.imshow(xxx)
# plt.show()


# ii = cv2.imread("C:\\Users\\PC\\Downloads\\test_Conv\\Convolutional Models\\DatasetCreation\\data_sneakers\\Nike\\Nike Air Bakin_ Posite _ Men_s.png")
# gray_image = cv2.cvtColor(ii, cv2.COLOR_RGBA2BGR)
# print(gray_image.shape)
# plt.imshow(gray_image)
# plt.show()


OUTPUT_PATH = 'C:\\Users\\PC\\Downloads\\test_Conv\\Convolutional Models\\DatasetCreation\\data_shoes\\'

ftrain = h5py.File(OUTPUT_PATH + 'train_dataset.h5', 'r')
ftest = h5py.File(OUTPUT_PATH + 'test_dataset.h5', 'r')

# Divido el dataset en datos de entreno y evalucion con respectivas etiquetas
train_data, train_labels = ftest['test_set_x'], ftest['test_set_y']
eval_data, eval_labels = ftrain['train_set_x'],  ftrain['train_set_y']

plt.imshow(train_data[4])
plt.show()

plt.imshow(eval_data[21])
plt.show()
