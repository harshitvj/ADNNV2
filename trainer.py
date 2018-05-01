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
from alexnet import alexnet


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