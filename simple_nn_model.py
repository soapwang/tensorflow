import tensorflow as tf
from numpy.random import RandomState

batch_size = 8

w1 = tf.Variable
w1 = tf.Variable(tf.random_normal([2,3],stddev=1))
w2 = tf.Variable(tf.random_normal([3,1],stddev=1))

x = tf.placeholder(tf.float32, shape=(None,2))
y_ = tf.placeholder(tf.float32,shape=(None,1))

a = tf.nn.relu(tf.matmul(x, w1))
y = tf.nn.relu(tf.matmul(a, w2))

cross_entropy = -tf.reduce_mean(y_ *tf.log(tf.clip_by_value(y, 1e-10, 1.0)))
train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)
rdm = RandomState(1)
dataset_size = 128

X = rdm.rand(dataset_size, 2)
Y = [[int(x1+x2<1)] for (x1, x2) in X]
#test samples
X_ = rdm.rand(dataset_size * 2, 2)
Y_ = [[int(x1+x2<1)] for (x1, x2) in X_]

epochs = 10000

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(w1))
    print(sess.run(w2))

    for i in range(epochs):
        start = (i*batch_size) % dataset_size
        end = min(start+batch_size, dataset_size)
        sess.run(train_step, feed_dict={x:X[start:end], y_:Y[start:end]})
        if i % 1000 ==0:
            total_ce = sess.run(cross_entropy, feed_dict={x: X, y_: Y})
            print("After %d training steps, cross entropy of all is %g" % (i, total_ce))
    #evaluating the model
    #y_pred = sess.run(y, feed_dict={x:X_})
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(sess.run(accuracy, feed_dict={x: X_, y_: Y_}))
