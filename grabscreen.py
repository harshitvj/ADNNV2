# Done by Frannecklp, modified a little.

import cv2
import numpy as np
import win32gui
import win32ui
import win32con
import win32api


def grab_screen(region=None):
    hwin = win32gui.GetDesktopWindow()

    if region:
        left, top, x2, y2 = region
        width = x2 - left + 1
        height = y2 - top + 1
    else:
        width = win32api.GetSystemMetrics(win32con.SM_CXVIRTUALSCREEN)
        height = win32api.GetSystemMetrics(win32con.SM_CYVIRTUALSCREEN)
        left = win32api.GetSystemMetrics(win32con.SM_XVIRTUALSCREEN)
        top = win32api.GetSystemMetrics(win32con.SM_YVIRTUALSCREEN)

    hwindc = win32gui.GetWindowDC(hwin)
    srcdc = win32ui.CreateDCFromHandle(hwindc)
    memdc = srcdc.CreateCompatibleDC()
    bmp = win32ui.CreateBitmap()
    bmp.CreateCompatibleBitmap(srcdc, width, height)
    memdc.SelectObject(bmp)
    memdc.BitBlt((0, 0), (width, height), srcdc, (left, top), win32con.SRCCOPY)

    signed_ints_array = bmp.GetBitmapBits(True)
    img = np.fromstring(signed_ints_array, dtype='uint8')
    img.shape = (height, width, 4)

    srcdc.DeleteDC()
    memdc.DeleteDC()
    win32gui.ReleaseDC(hwin, hwindc)
    win32gui.DeleteObject(bmp.GetHandle())

    return cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)


# def roi(img, vertices=None):
#     """ This method returns cropped image according to passed vertices """
#
#     if vertices is None:
#         vertices = [np.array([[0, 0], [0, 255], [270, 255], [400, 75], [540, 75], [655, 255], [960, 255], [960, 0]])]
#
#     masked = np.zeros_like(img)
#     cv2.fillPoly(masked, vertices, 255)
#     masked = cv2.bitwise_and(img, masked)
#     return masked

def process_image():
    # grabbed_image = np.array(ImageGrab.grab(bbox=(0, 250, 960, 480)))
    grabbed_image = grab_screen(region=(10, 400, 970, 580))
    gray_image = cv2.cvtColor(grabbed_image, cv2.COLOR_RGB2GRAY)
    # cropped_image = np.array(roi(gray_image))
    return np.array(cv2.resize(gray_image, (96, 18)))


if __name__ == "__main__":
    while True:
        process_image()
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break