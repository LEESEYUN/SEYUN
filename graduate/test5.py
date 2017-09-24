import tensorflow as tf


x=tf.placeholder(tf.float32)
y=tf.placeholder(tf.float32)
z=tf.multiply(x,y)
with tf.Session() as sess:
    for i in range(10):
        print("Z: %d" %sess.run(z,feed_dict={x:[i], y:[i+1]}))
