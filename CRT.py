__author__ = 'Rybkin & Kravchenko'

from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM
from keras.layers.core import RepeatVector
import NTM2
import numpy as np


n = 10
m = 2
sample_size = 20000


X_train = np.random.ranf((sample_size,n,1))
Y_train = X_train
for i in range(m - 1):
    Y_train = np.concatenate((Y_train, X_train), 1)
print(X_train.shape)
print(Y_train.shape)
model = Sequential()
model.add(NTM2.NeuralTuringMachine(64, return_sequences=True, input_shape = (n,1)))
model.add(NTM2.NeuralTuringMachine(64, return_sequences=False))
model.add(Dropout(0.5))
model.add(RepeatVector(m*n))
model.add(NTM2.NeuralTuringMachine(1, return_sequences=True))
model.add(Activation('sigmoid'))

model.compile(loss='mse', optimizer='adam')

model.fit(X_train, Y_train, batch_size=1000, nb_epoch=1)
