# Citation: Box Of Hats (https://github.com/Box-Of-Hats )

import win32api as wapi

keyList = ['W', 'A', 'D', 'T']


def key_check():
    keys = []
    for key in keyList:
        if wapi.GetAsyncKeyState(ord(key)):
            keys.append(key)
    return keys


def keys_to_output(keys):
    output = [0, 0, 0, 0]

    if 'A' in keys:
        if 'W' in keys:
            output[1] = 1
        # else:
        # 	output[3] = 1

    elif 'D' in keys:
        if 'W' in keys:
            output[2] = 1
        # else:
        # 	output[4] = 1

    elif 'W' in keys:
        output[0] = 1

    else:
        output[3] = 1

    return output
