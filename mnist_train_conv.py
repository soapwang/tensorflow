import os

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

import mnist_inference
import mnist_inference_conv

BATCH_SIZE = 100
LEARNING_RATE = 0.8
LEARNING_RATE_DECAY = 0.99
REGULARIZATION_RATE = 0.0001
TRAINING_STEPS = 5001
MOVING_AVERAGE_DECAY = 0.99

MODEL_SAVE_PATH = "./model/"
MODEL_NAME = "mnist_test.ckpt"

def train(mnist):
	x = tf.placeholder(
		tf.float32, [
		BATCH_SIZE, 
		mnist_inference_conv.IMAGE_SIZE,
		mnist_inference_conv.IMAGE_SIZE,
		mnist_inference_conv.NUM_CHANNELS], 
		name='x-input')
	y_ = tf.placeholder(tf.float32, [None, mnist_inference_conv.OUTPUT_NODE], name='y-input')
	#to avoid overfitting
	regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
	
	y = mnist_inference_conv.inference(x, True, regularizer)
	#identifies how many steps the model has been trained
	global_step = tf.Variable(0, trainable = False)
	
	variable_averages = tf.train.ExponentialMovingAverage(
		MOVING_AVERAGE_DECAY, global_step)
		
	variables_averages_op = variable_averages.apply(
		tf.trainable_variables())
	
	cross_entropy = tf.reduce_mean(
		tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
	
	#loss = cross_entropy + tf.add_n(tf.get_collection('losses'))
	loss = cross_entropy
	learing_rate = tf.train.exponential_decay(
		LEARNING_RATE, global_step, mnist.train.num_examples / BATCH_SIZE, LEARNING_RATE_DECAY)
		
	train_step = tf.train.GradientDescentOptimizer(learing_rate).minimize(loss, global_step=global_step)
	with tf.control_dependencies([train_step, variables_averages_op]):
		train_op = tf.no_op(name = 'train')
	
	saver = tf.train.Saver()
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		
		for i in range(TRAINING_STEPS):
			batch_xs, batch_ys = mnist.train.next_batch(BATCH_SIZE)
			reshaped_xs = np.reshape(batch_xs, (BATCH_SIZE, mnist_inference_conv.IMAGE_SIZE, mnist_inference_conv.IMAGE_SIZE, mnist_inference_conv.NUM_CHANNELS))
			_, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: reshaped_xs, y_: batch_ys})
			
			if i % 100 ==0:
				print("After %d training steps, loss on training batch is %g." % (i, loss_value))
				saver.save(sess, MODEL_SAVE_PATH+MODEL_NAME, global_step = global_step)
def main(argv=None):
	mnist = input_data.read_data_sets("/tmp/data", one_hot=True)
	train(mnist)
	
if __name__ == '__main__':
	tf.app.run()
