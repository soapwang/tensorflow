import tensorflow as tf
import pickle
import random
import numpy as np

CIFAR_FILES = ['/tmp/cifar-10-batches-py/data_batch_1', '/tmp/cifar-10-batches-py/data_batch_2', 
                '/tmp/cifar-10-batches-py/data_batch_3', '/tmp/cifar-10-batches-py/data_batch_4',
                '/tmp/cifar-10-batches-py/data_batch_5']
                
TEST_FILES = '/tmp/cifar-10-batches-py/test_batch'
 
				
BATCH_SIZE = 50
DATASET_SIZE = 50000

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
    labels_array = np.asarray(one_hot_labels, dtype=np.float32)
    return labels_array
    
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
    
    W_conv1 = weight_variable([5, 5, 3, 32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    
    h_pool1 = max_pool_2x2(h_conv1)
    
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

    h_pool2 = max_pool_2x2(h_conv2)
    h_pool2_flat = tf.reshape(h_pool2, [-1, 8*8*64])

    W_fc1 = weight_variable([8*8*64, 256])
    b_fc1 = bias_variable([256])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    W_fc2 = weight_variable([256, 10])
    b_fc2 = bias_variable([10])

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    return y_conv, keep_prob
    
def main():
    
    #train_set = []
    dataset = unpickle(CIFAR_FILES[0])
    train_set = dataset[b'data']
    labels = get_labels(dataset[b'labels'])
    print(type(train_set))
    print(type(labels))
    
    for i in range(1,5):
        #a dict with 4 keys: b'labels', b'batch_label', b'data', b'filenames'
        dataset = unpickle(CIFAR_FILES[i])
        train_set = np.vstack((train_set, dataset[b'data']))
        labels = np.vstack((labels, get_labels(dataset[b'labels'])))
        
    
    train_set_array = np.asarray(train_set, dtype=np.float32)
    train_set_array = train_set_array / 255.0
    
    test_set = unpickle(TEST_FILES)
    test_data = test_set[b'data']
    test_data_array = np.asarray(test_data, dtype=np.float32)
    test_data_array = test_data_array / 255.0
    test_labels = get_labels(test_set[b'labels'])
    #print(train_set_array)
    
    #get "one-hot" format of the labels
    #labels = get_labels(dataset[b'labels'])
    '''
    labels_raw = dataset[b'labels']
    labels = []
    for i in labels_raw:
        label = tf.one_hot(
            i, depth = 10, on_value = 1.0, off_value = 0.0)
        labels.append(label)
    
    for j in range(5):
        print(labels[j])
    '''
    x = tf.placeholder(tf.float32, [None, 3072])
    #placeholder of true labels
    y_ = tf.placeholder(tf.float32, [None, 10])
    #the output of the model
    y_conv, keep_prob = cnn(x)
    
    #safe_y_conv = tf.clip_by_value(y_conv, 1e-10, 1e10)
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
    train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    #correct =  tf.nn.in_top_k(y_conv, y_, 1)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    steps = 100001
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(steps):
            start = (i*BATCH_SIZE) % DATASET_SIZE
            end = min(start+BATCH_SIZE, DATASET_SIZE)
            sess.run(train_step, feed_dict={
                x:train_set_array[start:end], y_:labels[start:end], keep_prob: 0.5})
            #print accuracy on the train_set
            if i % 100 == 0: 
                t_start =  random.randint(0,99) * BATCH_SIZE
                t_end = t_start + BATCH_SIZE
                train_accuracy = accuracy.eval(feed_dict={
                    x:train_set_array[t_start:t_end], y_:labels[t_start:t_end], keep_prob: 1.0})
                #print('y_conv=', sess.run(y_conv, feed_dict={x:train_set_array[t_start:t_end],  keep_prob: 1.0}))
                #print('argmax=', sess.run(tf.argmax(y_conv,1), feed_dict={x:train_set[t_start:t_end],  keep_prob: 1.0}))    
                #print('labels[]=', labels[t_start:t_end])
                print("after %d steps, train accuracy: %g" % (i, train_accuracy))    
                print('cross entropy=', sess.run(cross_entropy, feed_dict={x:train_set_array[t_start:t_end], y_:labels[t_start:t_end], keep_prob: 1.0}))
                
                #run test data every 5000 steps
                if i>0 and i % 5000 == 0:
                    test_accuracy = accuracy.eval(feed_dict={
                        x:test_data_array, y_:test_labels, keep_prob: 1.0})
                    print("-----------------------------------------------")
                    print("|                                             |")
                    print("|                                             |")
                    print("|------------ test accuracy: %g ------------|" % (test_accuracy))  
                    print("|                                             |")
                    print("|                                             |")
                    print("-----------------------------------------------")
    

if __name__ == '__main__':
    main()