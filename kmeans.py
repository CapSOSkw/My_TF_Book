import numpy as np
import tensorflow as tf
from tensorflow.contrib.factorization import KMeans

tf.logging.set_verbosity(tf.logging.INFO)


mnist = tf.contrib.learn.datasets.load_dataset('mnist')
full_data_x = mnist.train.images

num_steps = 100
batch_size = 1024
k = 25
num_class = 10
num_feature = 784

X = tf.placeholder(tf.float32, shape=[None, num_feature])
Y = tf.placeholder(tf.float32, shape=[None, num_class])

kmeans = KMeans(inputs=X, num_clusters=k, distance_metric='cosine',
                use_mini_batch=True)

training_graph = kmeans.training_graph()

(all_scores, cluster_idx, scores, cluster_center_initialized, init_op, train_op) = training_graph

cluster_idx = cluster_idx[0]
avg_distance = tf.reduce_mean(scores)

init_vars = tf.global_variables_initializer()

sess = tf.Session()

sess.run(init_vars, feed_dict={X: full_data_x})
sess.run(init_op, feed_dict={X: full_data_x})

for i in range(1, num_steps+1):
    _, d, idx = sess.run([train_op, avg_distance, cluster_idx], feed_dict={X: full_data_x})

    if i % 10 == 0 or i ==1:
        print("Step %i, Avg Distance: %f" % (i, d))


counts = np.zeros(shape=(k, num_class))
for i in range(len(idx)):
    counts[idx[i]] += mnist.train.labels[i]
# Assign the most frequent label to the centroid
labels_map = [np.argmax(c) for c in counts]
labels_map = tf.convert_to_tensor(labels_map)

# Evaluation ops
# Lookup: centroid_id -> label
cluster_label = tf.nn.embedding_lookup(labels_map, cluster_idx)
# Compute accuracy
correct_prediction = tf.equal(cluster_label, tf.cast(tf.argmax(Y, 1), tf.int32))
accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

