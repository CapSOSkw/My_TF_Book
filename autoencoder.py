import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

mnist = tf.contrib.learn.datasets.load_dataset('mnist')

lr = 0.01
num_steps = 30000
batch_size = 256

display_step = 1000
examples_to_show = 10

num_hidden1 = 256
num_hidden2 = 128
num_input = 784


X = tf.placeholder('float', [None, num_input])

weights = {
    'encoder_h1': tf.Variable(tf.random_normal([num_input, num_hidden1])),
    'encoder_h2': tf.Variable(tf.random_normal([num_hidden1, num_hidden2])),
    'decoder_h1': tf.Variable(tf.random_normal([num_hidden2, num_hidden1])),
    'decoder_h2': tf.Variable(tf.random_normal([num_hidden1, num_input]))
}

biases = {
    'encoder_b1': tf.Variable(tf.random_normal([num_hidden1])),
    'encoder_b2': tf.Variable(tf.random_normal([num_hidden2])),
    'decoder_b1': tf.Variable(tf.random_normal([num_hidden1])),
    'deocder_b2': tf.Variable(tf.random_normal([num_input]))
}


def encoder(x):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']), biases['encoder_b1']))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']), biases['encoder_b2']))
    return layer_2


def decoder(x):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']), biases['decoder_b1']))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']), biases['deocder_b2']))

    return layer_2

encoder_op = encoder(X)
decoder_op = decoder(encoder_op)

y_pred = decoder_op
y_true = X

loss = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
optimizer = tf.train.RMSPropOptimizer(lr).minimize(loss)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for i in range(1, num_steps + 1):
        batch_x, _ = mnist.train.next_batch(batch_size)

        _, l = sess.run([optimizer, loss], feed_dict={X: batch_x})

        if i % display_step == 0 or i == 1:
            print('Step %i: Minibatch Loss: %f' % (i, l))







