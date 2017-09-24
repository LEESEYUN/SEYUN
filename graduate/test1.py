import tensorflow as tf
import cv2
import utils.sy_utils as utils
import numpy as np

FLAGS= tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('MODE',default_value='webcam',docstring='webcam,image')
tf.app.flags.DEFINE_integer('input_size',default_value=328,docstring='input_size')
tf.app.flags.DEFINE_integer('cam',default_value=0, docstring='webcam number')





def main(argv):
    tf_device='/gpu:0'
    sess=tf.Session()
    sess.run(tf.global_variables_initializer())
    with tf.device(tf_device):
        cam = cv2.VideoCapture(FLAGS.cam)
        while True:
            #test_img=None
            if FLAGS.MODE == 'webcam':
                test_img=utils.read_cam(cam,FLAGS.input_size)
            #elif FLAGS.MODE =='image':
            #    test_img=utils.read_image()
            gray=cv2.cvtColor(test_img,cv2.COLOR_BGR2GRAY)
            cv2.imshow('gray',gray)
            cv2.moveWindow('gray',2000,200)
            cv2.imshow('color',test_img)
            cv2.moveWindow('color',1000,750)
            if cv2.waitKey(1) == ord('q'): break

    cv2.destroyAllWindows()



if __name__ == '__main__':
    tf.app.run()