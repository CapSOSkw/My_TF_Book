import tensorflow as tf

sess = tf.Session()

a = tf.constant([10., 20., 40.], name='a')
b = tf.Variable(tf.random_uniform([3]), name='b')

output = tf.add_n([a, b], name='output')

sess.run(tf.global_variables_initializer())

writer = tf.summary.FileWriter('./logs', sess.graph)

writer.close()


'''
Then enter command:
tensorboard --logdir /path/to/logs
Open url: localhost:port, default localhost:6006
'''