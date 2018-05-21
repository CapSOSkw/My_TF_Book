import tensorflow as tf

mnist = tf.contrib.learn.datasets.load_dataset('mnist')

x = tf.placeholder(tf.float32, [None, 784])
#  None 表示其值大小不定，在这里作为第一个维度值，用以指代batch的大小，意即 x 的数量不定
y_ = tf.placeholder(tf.int64, [None])

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

y = tf.matmul(x, W) + b

cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=y_, logits=y)
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

for _ in range(2000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})


correct_prediction = tf.equal(tf.argmax(y, 1), y_)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(
  accuracy, feed_dict={
      x: mnist.test.images,
      y_: mnist.test.labels
  }))

