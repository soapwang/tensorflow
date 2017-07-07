Python 3.5.3 (v3.5.3:1880cb95a742, Jan 16 2017, 16:02:32) [MSC v.1900 64 bit (AMD64)] on win32
Type "copyright", "credits" or "license()" for more information.
>>> import tensorflow as tf
>>> w1 = tf.Variable(tf.random_normal([2,3],stddev=1,seed=1))
>>> w2 = tf.Variable(tf.random_normal([3,1],stddev=1,seed=1))

>>> a = tf.matmul(x, w1)
>>> x = tf.constant([[0.7, 0.9]])

>>> a = tf.matmul(x, w1)
>>> y = tf.matmul(a, w2)

>>> sess = tf.Session()
>>> sess.run(tf.global_variables_initializer())
>>> print(sess.run(y))
[[ 3.95757794]]
>>> sess.close()
