from random import randint, random, sample

import numpy as np
from sklearn.utils import shuffle

epsilon = 10 ** (-3)


def divide_data_set(data, target):
    test_data = list()
    learn_data = list()

    i = 0
    while len(data) > 0:
        if i < 100:
            x = randint(0, len(data) - 1)
            learn_data.append(data[x].tolist())
            learn_data[i].append(target[x])
            data = np.delete(data, x, 0)
            target = np.delete(target, x, 0)

        else:
            x = randint(0, len(data) - 1)
            test_data.append(data[x].tolist())
            test_data[len(test_data) - 1].append(target[x])
            data = np.delete(data, x, 0)
            target = np.delete(target, x, 0)

        i += 1

    return np.array(learn_data), np.array(test_data)


class Network:
    def __init__(self, dropout=False):
        self.biases = list()
        self.deltas = list()
        self.dropout = dropout
        self.layers = list()
        self.old_weights = list()
        self.outputs = list()

    @staticmethod
    def sigmoid(x):
        return 1 / float(1 + np.exp(-x))

    @staticmethod
    def sigmoid_vector(x):
        vector = np.array([1 / (1 + np.exp(-x[i])) for i in range(len(x))])
        return vector

    @staticmethod
    def sign(x):
        return int(x > 0)

    def sign_vector(self, x):
        return np.array([self.sign(x[i]) for i in range(len(x))])

    def layer(self, input_count, output_count):
        self.layers.append(np.multiply(np.random.rand(input_count, output_count), EPS))
        self.old_weights.append(np.zeros((input_count, output_count)))
        self.biases.append(random() / 2.0)

    def back_propagation(self, x, y, learning_rate, momentum):
        delta = [np.array([np.array([0.0 for _ in range(self.layers[i].shape[1])])
                           for _ in range(self.layers[i].shape[0])])
                 for i in range(len(self.layers))]

        for p in range(len(x)):
            self.deltas = [np.array([0.0]) for _ in range(len(self.layers))]

            for i in range(len(self.layers) - 1, -1, -1):
                self.deltas[i] = np.array([0.0 for _ in range(self.layers[i].shape[1])])

                for j in range(self.layers[i].shape[1]):
                    if i == (len(self.layers) - 1):
                        self.deltas[i][j] = np.multiply(-self.outputs[i][p][j],
                                                        np.multiply(1 - self.outputs[i][p][j],
                                                                    y[p][j] - self.outputs[i][p][j]))
                    else:
                        self.deltas[i][j] = np.multiply(self.outputs[i][p][j],
                                                        np.multiply(1 - self.outputs[i][p][j],
                                                                    self.deltas[i + 1].dot(self.layers[i + 1][j].T)))

            for i in range(len(self.layers) - 1, -1, -1):
                if i == 0:
                    for j in range(self.layers[i].shape[1]):
                        for k in range(self.layers[i].shape[0]):
                            alpha = np.subtract(0,
                                                np.multiply(np.multiply(learning_rate,
                                                                        self.deltas[i][j]),
                                                            x[p][k]))

                            if momentum != 0 & i != (len(self.layers) - 1):
                                delta[i][:, j][k] = np.add(delta[i][:, j][k],
                                                           np.add(alpha,
                                                                  np.multiply(momentum, self.old_weights[i][:, j][k])))

                            else:
                                delta[i][:, j][k] = np.add(delta[i][:, j][k], alpha)

                else:
                    for j in range(self.layers[i].shape[1]):
                        for k in range(self.layers[i].shape[0]):
                            alpha = np.subtract(0,
                                                np.multiply(np.multiply(learning_rate, self.deltas[i][j]),
                                                            self.outputs[i - 1][p][k]))

                            if momentum != 0 & i != (len(self.layers) - 1):
                                delta[i][:, j][k] = np.add(delta[i][:, j][k],
                                                           np.add(alpha,
                                                                  np.multiply(momentum, self.old_weights[i][:, j][k])))

                            else:
                                delta[i][:, j][k] = np.add(delta[i][:, j][k], alpha)

        self.old_weights = delta

        for p in range(len(self.layers)):
            self.layers[p] = np.add(self.layers[p], np.divide(delta[p], float(len(x))))

    def __fit(self, x, y, learning_rate, momentum, dropout):
        self.outputs.append(x.dot(self.layers[0]))

        if dropout:
            a = [i for i in range(len(self.outputs[0][0]))]
            b = sample(a, len(a) / 2)

            for number in b:
                for i in range(len(self.outputs[0])):
                    self.outputs[0][i][number] = 0.0

        for i in range(len(self.outputs[0])):
            self.outputs[0][i] = self.sigmoid_vector(self.outputs[0][i])

        for i in range(1, len(self.layers)):
            self.outputs.append(self.outputs[i - 1].dot(self.layers[i]))

            if dropout and i != (len(self.layers) - 1):
                a = [j for j in range(len(self.outputs[i][0]))]
                b = sample(a, len(a) / 2)

                for number in b:
                    for j in range(len(self.outputs[i])):
                        self.outputs[i][j][number] = 0.0

            for j in range(len(self.outputs[i])):
                self.outputs[i][j] = self.sigmoid_vector(self.outputs[i][j])

        self.back_propagation(x, y, learning_rate, momentum)

        if self.dropout:
            for i in range(len(self.layers) - 1):
                self.layers[i] = np.divide(self.layers[i], float(2))

        self.outputs = list()

    def fit(self, x, y, batch_size=100, nb_epoch=1000, learning_rate=0.1, momentum=0.0, dropout=False):
        for _ in range(nb_epoch):
            working = True

            this_x, this_y = shuffle(x, y, random_state=0)

            while working:
                if batch_size < len(this_x):
                    batch_x = this_x[:batch_size]
                    batch_y = this_y[:batch_size]
                    this_x = this_x[batch_size:len(x)]
                    this_y = this_y[batch_size:len(y)]

                else:
                    batch_x = this_x
                    batch_y = this_y
                    working = False

                self.__fit(batch_x, batch_y, learning_rate, momentum, dropout)

    def score(self, x, y):
        mistake = 0

        for p in range(len(x)):
            self.outputs.append(np.add(x[p].dot(self.layers[0]), self.biases[0]))
            self.outputs[0] = self.sigmoid_vector(self.outputs[0])

            for i in range(1, len(self.layers)):
                self.outputs.append(np.add(self.outputs[i - 1].dot(self.layers[i]), self.biases[i]))
                self.outputs[i] = self.sigmoid_vector(self.outputs[i])

            if y[p][np.argmax(self.outputs[len(self.layers) - 1])]:
                mistake += 1

            self.outputs = list()

        return mistake / float(len(x))
