import cv2
import math
import numpy as np

def read_image(file, cam, boxsize):
    # from file
    oriImg = cv2.imread(file)
    scale = int(boxsize) / int((oriImg.shape[0] * 1))
    imageToTest = cv2.resize(oriImg, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4)

    output_img = np.ones((boxsize, boxsize, 3)) * 128

    if imageToTest.shape[1] < boxsize:
        offset = imageToTest.shape[1] % 2
        output_img[:, int(boxsize / 2 - math.ceil(imageToTest.shape[1] / 2)):int(
            boxsize / 2 + math.ceil(imageToTest.shape[1] / 2) - offset), :] = imageToTest
    else:
        output_img = imageToTest[:,
                     int(imageToTest.shape[1] / 2 - boxsize / 2):int(imageToTest.shape[1] / 2 + boxsize / 2), :]
    return output_img
def read_cam(cam,boxsize):
    ret,oriImg=cam.read()
    print(type(oriImg))
    print(type(oriImg.shape[0]))
    scale = 328 / (oriImg.shape[0] * 1.0)
    test_cam=cv2.resize(oriImg, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4)
    return test_cam