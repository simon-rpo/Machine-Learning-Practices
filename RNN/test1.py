# import heapq
import heapq
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
# import seaborn as sns
from keras.layers import LSTM, Activation, Dense, Dropout, TimeDistributed
from keras.layers.core import Activation, Dense, Dropout, RepeatVector
from keras.models import Sequential, load_model
from keras.optimizers import RMSprop

# from pylab import rcParams


np.random.seed(42)
tf.set_random_seed(42)

# %matplotlib inline

# sns.set(style='whitegrid', palette='muted', font_scale=1.5)

# rcParams['figure.figsize'] = 12, 5


path = 'C:\\Users\\PC\\Downloads\\test_Conv\\RNN\\dataset\\tmp\\nietzsche.txt'
text = open(path).read().lower()
print('corpus length:', len(text))

chars = sorted(list(set(text)))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

print('unique chars: ' + str(len(chars)))


SEQUENCE_LENGTH = 40
step = 3
sentences = []
next_chars = []
for i in range(0, len(text) - SEQUENCE_LENGTH, step):
    sentences.append(text[i: i + SEQUENCE_LENGTH])
    next_chars.append(text[i + SEQUENCE_LENGTH])
    #print('num training examples: '+str(len(sentences)))


X = np.zeros((len(sentences), SEQUENCE_LENGTH, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        X[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1

# sentences[100]
# next_chars[100]

# X[0][0])
# y[0]

# X.shape
# y.shape


model = Sequential()
model.add(LSTM(128, input_shape=(SEQUENCE_LENGTH, len(chars))))
model.add(Dense(len(chars)))
model.add(Activation('softmax'))


model.summary()

# optimizer = RMSprop(lr=0.01)
# model.compile(loss='categorical_crossentropy',
#               optimizer=optimizer, metrics=['accuracy'])

# history = model.fit(X, y, validation_split=0.05,
#                     batch_size=128, epochs=32, shuffle=True).history

# model.save('C:\\Users\\PC\\Downloads\\test_Conv\\RNN\\dataset\\tmp\\keras_model.h5')
# pickle.dump(history, open("history.p", "wb"))


model = load_model(
    'C:\\Users\\PC\\Downloads\\test_Conv\\RNN\\dataset\\tmp\\keras_model.h5')
history = pickle.load(open("history.p", "rb"))


plt.plot(history['acc'])
plt.plot(history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')


plt.plot(history['loss'])
plt.plot(history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')


def prepare_input(text):
    x = np.zeros((1, SEQUENCE_LENGTH, len(chars)))

    for t, char in enumerate(text):
        x[0, t, char_indices[char]] = 1.

    return x


def sample(preds, top_n=3):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds)
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)

    return heapq.nlargest(top_n, range(len(preds)), preds.take)


def predict_completion(text):
    original_text = text
    generated = text
    completion = ''
    while True:
        x = prepare_input(text)
        preds = model.predict(x, verbose=0)[0]
        next_index = sample(preds, top_n=1)[0]
        next_char = indices_char[next_index]

        text = text[1:] + next_char
        completion += next_char

        if len(original_text + completion) + 2 > len(original_text) and next_char == ' ':
            return completion


def predict_completions(text, n=3):
    x = prepare_input(text)
    preds = model.predict(x, verbose=0)[0]
    next_indices = sample(preds, n)
    return [indices_char[idx] + predict_completion(text[1:] + indices_char[idx]) for idx in next_indices]


quotes = [
    "It is not a lack of love, but a lack of friendship that makes unhappy marriages.",
    "That which does not kill us makes us stronger.",
    "I'm not upset that you lied to me, I'm upset that from now on I can't believe you.",
    "And those who were seen dancing were thought to be insane by those who could not hear the music.",
    "It is hard enough to remember my opinions, without also remembering my reasons for them!"
]


for q in quotes:
    seq = q[:40].lower()
    print(seq)
    print(predict_completions(seq, 5))
    print()


def predict_sentence_completions(text, n=3):
    predictions = []
    # TODO add predictions to the list
    raise NotImplementedError
    return predictions


for q in quotes:
    seq = q[:40].lower()
    print(seq)
    print(predict_sentence_completions(seq, 5))
    print()
