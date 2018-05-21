import tensorflow as tf
import numpy as np


matrix1 = tf.constant([[3., 3.]])
matrix2 = tf.constant([[2.], [2.]])

product = tf.matmul(matrix1, matrix2)

with tf.Session() as sess:
    with tf.device("/gpu:0"):
        result = sess.run(product)
        print(result)


a = tf.placeholder(tf.int32)
b = tf.placeholder(tf.int32)

add = tf.add(a, b)
mul = tf.multiply(a, b)

with tf.Session() as sess:
    print(sess.run(add, feed_dict={a: 2, b: 11}))
    print(sess.run(mul, feed_dict={a: 9, b: 10}))


hello = tf.constant('Hello, World!')

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    print(sess.run(hello))