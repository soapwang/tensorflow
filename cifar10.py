import tensorflow as tf
import pickle

CIFAR_FILE = '/tmp/cifar-10-batches-py/data_batch_1'
BATCH_SIZE = 50
DATASET_SIZE = 10000

def unpickle(file):
    with open(file, 'rb') as f:
        dict = pickle.load(f, encoding='bytes')
    return dict

def get_labels(lables):
	one_hot_labels = []
	for i in lables:
		one_hot = [0]
		one_hot = one_hot * 10
		one_hot[i] = 1
		one_hot_labels.append(one_hot)
	return one_hot_labels
	
def conv2d(x, W):
	"""conv2d returns a 2d convolution layer with full stride."""
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
	"""max_pool_2x2 downsamples a feature map by 2X."""
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
				strides=[1, 2, 2, 1], padding='SAME')
	
def weight_variable(shape):
	"""weight_variable generates a weight variable of a given shape."""
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)


def bias_variable(shape):
	"""bias_variable generates a bias variable of a given shape."""
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)
	
def cnn(x):
	x_image = tf.reshape(x, [-1, 32, 32, 3])
	W_conv1 = weight_variable([5, 5, 3, 16])
	b_conv1 = bias_variable([16])
	h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
	h_pool1 = max_pool_2x2(h_conv1)
	
	W_conv2 = weight_variable([5, 5, 16, 64])
	b_conv2 = bias_variable([64])
	h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

	h_pool2 = max_pool_2x2(h_conv2)

	h_pool2_flat = tf.reshape(h_pool2, [-1, 8*8*64])

	W_fc1 = weight_variable([8*8*64, 512])
	b_fc1 = bias_variable([512])
	

	h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

	keep_prob = tf.placeholder(tf.float32)
	h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

	W_fc2 = weight_variable([512, 10])
	b_fc2 = bias_variable([10])

	y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
	return y_conv, keep_prob
	
def main():
	#a dict with 4 keys: b'labels', b'batch_label', b'data', b'filenames'
	dataset = unpickle(CIFAR_FILE)
	
	#a 10000*3072 list
	train_set = dataset[b'data']
	
	#get "one-hot" format of the labels
	labels = get_labels(dataset[b'labels'])
	
	x = tf.placeholder(tf.float32, [None, 3072])
	#placeholder of true labels
	y_ = tf.placeholder(tf.float32, [None, 10])
	#the output of the model
	y_conv, keep_prob = cnn(x)
	
	cross_entropy = tf.reduce_mean(
		tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
	train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)
	correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	
	steps = 5000
	
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		for i in range(steps):
			start = (i*BATCH_SIZE) % DATASET_SIZE
			end = min(start+BATCH_SIZE, DATASET_SIZE)
			sess.run(train_step, feed_dict={
				x:train_set[start:end], y_:labels[start:end], keep_prob: 0.5})
			#print accuracy on the train_set
			if i % 100 == 0: 
				train_accuracy = accuracy.eval(feed_dict={
					x:train_set[start:end], y_:labels[start:end], keep_prob: 1.0})
				print("after %d steps, train accuracy: %g" % (i, train_accuracy))	
	

if __name__ == '__main__':
	main()