""" AlexNet.
References:
    - Alex Krizhevsky, Ilya Sutskever & Geoffrey E. Hinton. ImageNet
    Classification with Deep Convolutional Neural Networks. NIPS, 2012.
Links:
    - [AlexNet Paper](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)
"""
#Alexnet
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from tflearn.layers.normalization import local_response_normalization
import numpy as np
from sklearn.model_selection import train_test_split
import time

def alexnet(width, height, lr):
    network = input_data(shape=[None, width, height, 1], name='input')
    network = conv_2d(network, 96, 11, strides=4, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)
    network = conv_2d(network, 256, 5, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)
    network = conv_2d(network, 384, 3, activation='relu')
    network = conv_2d(network, 384, 3, activation='relu')
    network = conv_2d(network, 256, 3, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)
    network = fully_connected(network, 4096, activation='tanh')
    network = dropout(network, 0.5)
    network = fully_connected(network, 4096, activation='tanh')
    network = dropout(network, 0.5)
    network = fully_connected(network, 4, activation='softmax')
    network = regression(network, optimizer='momentum',
                         loss='categorical_crossentropy',
                         learning_rate=lr, name='targets')

    model = tflearn.DNN(network, checkpoint_path = 'drive/colab/model.tfl.ckpt', max_checkpoints = 1)

    return model


WIDTH = 96
HEIGHT = 18
LR = 1e-3
EPOCHS = 10

#Trainer
def train_model():
    training_data = np.load('drive/colab/data.npy')

    x_data = [x[0] for x in training_data]
    y_data = [x[1] for x in training_data]

    train_X, test_X, train_Y, test_Y = train_test_split(x_data, y_data, test_size=0.20, random_state=675)

    train_X = np.array(train_X).reshape(-1, WIDTH, HEIGHT, 1)
    test_X = np.array(test_X).reshape(-1, WIDTH, HEIGHT, 1)

    model = alexnet(WIDTH, HEIGHT, LR)

    model.fit(train_X, train_Y, EPOCHS, validation_set=({'input': test_X}, {'targets': test_Y}),
              show_metric=True, snapshot_epoch=True, run_id = 'test_run')
    
    model.save('drive/colab/model.tfl')


if __name__ == "__main__":
	train_model()