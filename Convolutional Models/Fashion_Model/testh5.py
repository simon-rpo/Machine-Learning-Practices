import tensorflow as tf
import tensornets as nets
import tensorflow_hub as hub
#import h5py

inputs = tf.placeholder(tf.float32, [None, 299, 299, 3])
# model = nets.VGG16(inputs)
# load_model_weights = model.pretrained()

module = hub.Module(
        "https://tfhub.dev/google/imagenet/inception_v3/classification/1")
outputs = module(inputs)
sess.run(outputs['default'])
print(outputs['default'])

# assert isinstance(model, tf.Tensor)

# ftrain = h5py.File(
#     'C:\\Users\\PC\\Downloads\\test_Conv\\animal_data\\train_animals.h5', 'r')
# ftest = h5py.File(
#     'C:\\Users\\PC\\Downloads\\test_Conv\\animal_data\\test_animals.h5', 'r')

# train_data, train_labels = ftest['test_set_x'], ftest['test_set_y']
# eval_data, eval_labels = ftrain['train_set_x'],  ftrain['train_set_y']
