Python 3.5.3 (v3.5.3:1880cb95a742, Jan 16 2017, 16:02:32) [MSC v.1900 64 bit (AMD64)] on win32
Type "copyright", "credits" or "license()" for more information.
>>> import tensorflow as tf
>>> W = tf.Variable([.3], dtype=tf.float32)
>>> b = tf.Variable([-.3], dtype=tf.float32)
>>> x = tf.placeholder(tf.float32)
>>> linear_model = W*x+b
>>> init = tf.global_variables_initializer()
>>> sess = tf.Session()
>>> sess.run(init)
>>> print(sess.run(linear_model, {x: [1,2,3,4]}))
[ 0.          0.30000001  0.60000002  0.90000004]
>>> y = tf.placeholder(tf.float32)
>>> squared_deltas = tf.square(linear_model - y)
>>> loss = tf.reduce_sum(squared_deltas)
>>> print(sess.run(loss, {x:[1,2,3,4], y:[0,-1,-2,-3]}))
23.66
>>> fixW = tf.assign(W, [-1.])
>>> fixb = tf.assign(b, [1.])
>>> sess.run([fixW, fixb])
[array([-1.], dtype=float32), array([ 1.], dtype=float32)]
>>> print(sess.run(loss, {x:[1,2,3,4], y:[0,-1,-2,-3]}))
0.0
>>> optimizer = tf.train.GradientDescentOptimizer(0.01)
>>> train = optimizer.minimize(loss)
>>> sess.run(init)
>>> for i in range(1000):
	sess.run(train, {x:[1,2,3,4], y:[0,-1,-2,-3]})

	
>>> print(sess.run([W, b]))
[array([-0.9999969], dtype=float32), array([ 0.99999082], dtype=float32)]
