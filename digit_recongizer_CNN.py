# Load lib
import numpy as np
import pandas as pd
import tensorflow as tf

# Load train and test data
train_data = pd.read_csv('/home/lizejian/GitHub/Digit-Recognizer/data/train.csv')

# Separate images and labels
train_images = train_data.iloc[:, 1:].values.astype(np.float)
train_labels = train_data.iloc[:, 0].values

# Normalize
train_images = np.multiply(train_images, 1.0/255.0)

# split data into training & validation
validation_proportion = 0.7
validation_size = int(validation_proportion*train_data.shape[0])
train_images = train_images[validation_size:]
train_labels = train_labels[validation_size:]
validation_images = train_images[:validation_size]
validation_labels = train_labels[:validation_size]

# one_hot
def one_hot(labels_dense, num_classes):
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot

train_labels = one_hot(train_labels, 10)
validation_labels = one_hot(validation_labels,10)

def initialize_weight(shape):
	initial = tf.truncated_normal(shape, stddev = 0.1)
	return tf.Variable(initial)

def initialize_bias(shape):
	initial = tf.constant(0.1, shape = shape)
	return tf.Variable(initial)

def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides = [1, 1, 1, 1], padding = 'SAME')

def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')

#create CNN Model
x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float', [None, 10])
x_image = tf.reshape(x, [-1,28,28,1])#28x28x1

# layer1:conv2d
W1_conv = initialize_weight([5, 5, 1, 32])
b1_conv = initialize_bias([32])
h1_conv = tf.nn.relu(conv2d(x_image, W1_conv) + b1_conv)#28x28x1>>28x28x32
h1_pool = max_pool_2x2(h1_conv)#28x28x32>>14x14x32

# layer2:conv2d
W2_conv = initialize_weight([5, 5, 32, 64])
b2_conv = initialize_bias([64])
h2_conv = tf.nn.relu(conv2d(h1_pool, W2_conv) + b2_conv)#14x14x32>>14x14x64
h2_pool = max_pool_2x2(h2_conv)#14x14x64>>7x7x64
h2_pool_flat = tf.reshape(h2_pool, [-1, 7*7*64])#7x7x64>>7*7*64

# layer3:full connection
W3_fc = initialize_weight([7*7*64, 1024])
b3_fc = initialize_bias([1024])
h3_fc = tf.nn.relu(tf.matmul(h2_pool_flat, W3_fc) + b3_fc)#7*7*64>>1024

# drop out
keep_prob = tf.placeholder('float')
h3_fc_drop = tf.nn.dropout(h3_fc, keep_prob)

# layer4: full connection
W4_fc = initialize_weight([1024, 10])
b4_fc = initialize_bias([10])
y_conv = tf.nn.softmax(tf.matmul(h3_fc_drop, W4_fc) + b4_fc)

cross_entropy = -tf.reduce_sum(y*tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
predict = tf.argmax(y_conv,1)

epochs_completed = 0
index_in_epoch = 0
num_examples = train_images.shape[0]

# serve data by batches
def next_batch(batch_size):
    global train_images
    global train_labels
    global index_in_epoch
    global epochs_completed
    
    start = index_in_epoch
    index_in_epoch += batch_size
    
    # when all trainig data have been already used, it is reorder randomly    
    if index_in_epoch > num_examples:
        # finished epoch
        epochs_completed += 1
        # shuffle the data
        perm = np.arange(num_examples)
        np.random.shuffle(perm)
        train_images = train_images[perm]
        train_labels = train_labels[perm]
        # start next epoch
        start = 0
        index_in_epoch = batch_size
        assert batch_size <= num_examples
    end = index_in_epoch
    return train_images[start:end], train_labels[start:end]

# train model
train_steps = 2000
batch_size = 50

sess = tf.Session()
sess.run(tf.global_variables_initializer())
for i in range(train_steps):
    batch = next_batch(batch_size)
    if i % 100 == 0:
        train_accuracy = sess.run(accuracy, feed_dict = {
                x: batch[0], 
                y: batch[1], 
                keep_prob: 1.0})
        validation_accuracy = sess.run(accuracy, feed_dict = {
                x: validation_images[0:batch_size],
                y: validation_labels[0:batch_size],
                keep_prob: 1.0})
        print('step %d, train accuracy %g' % (i, train_accuracy))
        print('step %d, validation accuracy %g' % (i, validation_accuracy)) 
    sess.run(train_step, feed_dict = {x: batch[0], y: batch[1], keep_prob: 0.5})
	

# test model
test_data = pd.read_csv('/home/lizejian/GitHub/Digit-Recognizer/data/test.csv')
test_images = test_data.astype(np.float)
test_images = np.multiply(test_images, 1.0/255.0)

predicted_lables = np.zeros(test_images.shape[0])
for i in range(0,test_images.shape[0]//batch_size):
    predicted_lables[i*batch_size : (i+1)*batch_size] = sess.run(predict, feed_dict = {
            x: test_images[i*batch_size : (i+1)*batch_size], 
            keep_prob: 1.0})

np.savetxt('submission_softmax.csv', 
           np.c_[range(1, len(test_images)+1), predicted_lables], 
           delimiter = ',', 
           header = 'ImageId,Label', 
           comments = '', 
           fmt = '%d')
sess.close()
