import time
from getkeys import key_check, keys_to_output
from grabscreen import process_image
import numpy as np
from random import shuffle
from presskeys import ReleaseKey, W, A, D


filename = 'training_data.npy'

forward = []
# left = []
# right = []
forward_left = []
forward_right = []
release = []
length = 5000


def create_dataset():
    for i in range(5, 0, -1):
        print('In', i)
        time.sleep(1)

    paused = False
    while True:

        if not paused == True:

            data_vector = process_image().flatten()
            output = keys_to_output(key_check())

            #           [A, W, D, AW, DW]
            if output == [1, 0, 0, 0] and len(forward) < length:
                forward.append([data_vector, output])
            elif output == [0, 1, 0, 0] and len(forward_left) < length:
                forward_left.append([data_vector, output])
            elif output == [0, 0, 1, 0] and len(forward_right) < length:
                forward_right.append([data_vector, output])
            # elif output == [0, 0, 0, 1, 0, 0] and len(left) < length:
            #     left.append([data_vector, output])
            # elif output == [0, 0, 0, 0, 1, 0] and len(right) < length:
            #     right.append([data_vector, output])
            elif output == [0, 0, 0, 1] and len(release) < length:
                release.append([data_vector, output])
            else:
                pass

            print('Forward:', len(forward), 'FLeft:', len(forward_left), 'FRight:', len(forward_right), 'Release', len(release)) 
            if len(forward) == len(forward_left) == len(forward_right) == len(release) == length:

                print('Saving..')

                training_data = forward + forward_left + forward_right + release
                shuffle(training_data)
                np.save(filename, training_data)
                break

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

if __name__ == "__main__":
    create_dataset()