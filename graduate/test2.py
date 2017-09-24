import cv2
import numpy as np



def visualize():
    cam=cv2.VideoCapture(0)
    while True:
        ret,frame=cam.read()
        test_cam=cv2.resize(frame,(328,328))
        cv2.imshow('color',test_cam)
        cv2.moveWindow('color',2000,200)
        gray=cv2.cvtColor(test_cam,cv2.COLOR_BGR2GRAY)
        cv2.imshow('gray',gray)


        if cv2.waitKey(1) == ord('q'):
            break

    cv2.destroyAllWindows()



if __name__ == '__main__':
    visualize()
