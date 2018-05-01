import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from tflearn.layers.normalization import local_response_normalization
import numpy as np
from sklearn.model_selection import train_test_split
import time
from presskeys import act, ReleaseKey, W, A, D
from getkeys import key_check
from grabscreen import process_image
from alexnet import alexnet

WIDTH = 96
HEIGHT = 18
LR = 1e-3

# Predictor
def predict():
    model = alexnet(WIDTH, HEIGHT, LR)
    model.load('model.tfl')
    for i in range(5, 0, -1):
        print(i)
        time.sleep(1)

    paused = False
    while True:

        if not paused:
            screen = process_image()
            prediction = model.predict([screen.reshape(96, 18, 1)])[0]
            # print(prediction)

            output = np.argmax(prediction)
            act(output)

            if output == 0:
                print('Forward')
            elif output == 1:
                print('FLeft')
            elif output == 2:
                print('FRight')
            elif output == 3:
                print('Release')
                
        keys = key_check()
        # p pauses game and can get annoying.
        if 'T' in keys:
            if paused:
                paused = False
                time.sleep(1)
            else:
                paused = True
                ReleaseKey(A)
                ReleaseKey(W)
                ReleaseKey(D)
                time.sleep(1)

predict()
