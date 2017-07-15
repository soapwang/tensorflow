from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

INPUT_NODE = 784
OUTPUT_NODE = 10

IMAGE_SIZE = 28
NUM_CHANNELS = 1
NUM_LABELS = 10

CONV1_DEEP = 32
CONV1_SIZE = 5

CONV2_DEEP = 64
CONV2_SIZE = 5

FC_SIZE = 1024

def get_weight_variable(shape, regularizer=None):
	#for now, tf.get_variable() = tf.Variable()
	weights = tf.get_variable(
		"weights", shape,
		initializer = tf.truncated_normal_initializer(stddev=0.1)
	)
	
	if regularizer !=None:
		tf.add_to_collection('losses', regularizer(weights))
	
	return weights

def conv2d(x, W):
	"""conv2d returns a 2d convolution layer with full stride."""
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')	
	
def max_pool_2x2(x):
	"""max_pool_2x2 downsamples a feature map by 2X."""
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

def inference(input_tensor, train, regularizer):
	#use same variable names in different scope
	with tf.variable_scope('layer1-conv1'):
		conv1_weights = get_weight_variable(
			#for mnist, it's a [5,5,1,32] tensor
			[CONV1_SIZE, CONV1_SIZE, NUM_CHANNELS, CONV1_DEEP], regularizer=None)
		
		conv1_biases = tf.get_variable(
			"biases", [CONV1_DEEP],
			initializer = tf.constant_initializer(0.0))
		
		conv1 = tf.nn.relu(conv2d(input_tensor, conv1_weights) + conv1_biases)
		
	with tf.variable_scope('layer2-pool1'):
		pool1 = max_pool_2x2(conv1)	
		
		
	with tf.variable_scope('layer3-conv2'):
		conv2_weights = get_weight_variable(
			#a [5,5,32,64] tensor
			[CONV2_SIZE, CONV2_SIZE, CONV1_DEEP, CONV2_DEEP], regularizer=None)
		
		conv2_biases = tf.get_variable(
			"biases", [CONV2_DEEP],
			initializer = tf.constant_initializer(0.0))
		
		conv2 = tf.nn.relu(conv2d(pool1, conv2_weights) + conv2_biases)
		
	with tf.variable_scope('layer4-pool2'):
		pool2 = max_pool_2x2(conv2)	
		
	#pool_shape = pool2.get_shape().as_list()
	#nodes = pool_shape[1]*pool_shape[2]*pool_shape[3]
	
	with tf.variable_scope('layer5-fc1'):
		#fc1_weights = tf.get_weight_variable
		fc1_weights = get_weight_variable([7 * 7 * 64, 1024])
		fc1_biases = tf.get_variable(
			"biases", [1024],
			initializer = tf.constant_initializer(0.1))

		pool2_flat = tf.reshape(pool2, [-1, 7*7*64])
		fc1 = tf.nn.relu(tf.matmul(pool2_flat, fc1_weights) + fc1_biases)
		if train:
			fc1 = tf.nn.dropout(fc1, 0.5)
		
	with tf.variable_scope('layer6-fc2'):

		fc2_weights = get_weight_variable([1024, 10])
		fc2_biases = tf.get_variable(
			"biases", [10],
			initializer = tf.constant_initializer(0.1))


		y_conv = tf.matmul(fc1, fc2_weights) + fc2_biases
	return y_conv


