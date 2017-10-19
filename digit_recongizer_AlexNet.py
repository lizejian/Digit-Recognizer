# Load lib
import numpy as np
import pandas as pd
import tensorflow as tf

# Load train and test data
train_data = pd.read_csv('/home/lizejian/GitHub/Digit-Recognizer/data/train.csv')
test_data = pd.read_csv('/home/lizejian/GitHub/Digit-Recognizer/data/test.csv')

# Separate images and labels
train_images = train_data.iloc[:, 1:].values.astype(np.float)
test_images = test_data.astype(np.float)
train_labels = train_data.iloc[:, 0].values

# Normalize
train_images = np.multiply(train_images, 1.0/255.0)
test_images = np.multiply(test_images, 1.0/255.0)

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

def initialize_weight(shape, stddev, name):
    initial = tf.truncated_normal(shape, dtype = tf.float32, stddev = stddev)
    return tf.Variable(initial, name = name)

def initialize_bias(shape):
    initial = tf.random_normal(shape)
    return tf.Variable(initial)

def conv2d(x, w, b):
    return tf.nn.relu((tf.nn.conv2d(x, w, strides = [1, 1, 1, 1], padding = 'SAME') + b))

def max_pool(x, f):
    return tf.nn.max_pool(x, ksize = [1, f, f, 1], strides = [1, 1, 1, 1], padding = 'SAME')

# Create AlexNet Model
x = tf.placeholder(tf.float32, [None, 784])
x_image = tf.reshape(x, shape = [-1, 28, 28, 1])
y = tf.placeholder(tf.float32, [None, 10])

# layer1: conv2d
w1_conv = initialize_weight([3, 3, 1, 64], 0.1, 'w1')
b1_conv = initialize_bias([64])
h1_conv = conv2d(x_image, w1_conv, b1_conv)#28x28x1>>28x28x64
h1_pool = max_pool(h1_conv, 2)#28x28x64>>14x14x64

# layer2: conv2d
w2_conv = initialize_weight([3, 3, 64, 64], 0.1, 'w2')
b2_conv = initialize_bias([64])
h2_conv = conv2d(h1_pool, w2_conv, b2_conv)#14x14x64>>14x14x64
h2_pool = max_pool(h2_conv, 2)#14x14x64>>7x7x64

# layer3: conv2d
w3_conv = initialize_weight([3, 3, 64, 128], 0.1, 'w3')
b3_conv = initialize_bias([128])
h3_conv = conv2d(h2_pool, w3_conv, b3_conv)#7x7x64>>7x7x128

# layer4: conv2d
w4_conv = initialize_weight([3, 3, 128, 128], 0.1, 'w4')
b4_conv = initialize_bias([128])
h4_conv = conv2d(h3_conv, w4_conv, b4_conv)#7x7x128>>7x7x128

# layer5: conv2d
w5_conv = initialize_weight([3, 3, 128, 256], 0.1, 'w5')
b5_conv = initialize_bias([256])
h5_conv = conv2d(h4_conv, w5_conv, b5_conv)#7x7x128>>7x7x256
h5_pool = max_pool(h5_conv, 2)#
shape = h5_pool.get_shape() 
h5_pool_flat = tf.reshape(h5_pool, [-1, shape[1].value*shape[2].value*shape[3].value])

# layer6: full connection
w6_fc = initialize_weight([256*28*28, 1024], 0.01, 'w6')
b6_fc = initialize_bias([1024])
h6_fc = tf.nn.relu(tf.matmul(h5_pool_flat, w6_fc) + b6_fc)
keep_prob = tf.placeholder('float')
h6_drop = tf.nn.dropout(h6_fc, keep_prob = keep_prob)

# layer7: full connection
w7_fc = initialize_weight([1024, 1024], 0.01, 'w7')
b7_fc = initialize_bias([1024])
h7_fc = tf.nn.relu(tf.matmul(h6_drop, w7_fc) + b7_fc)
h7_drop = tf.nn.dropout(h7_fc, keep_prob = keep_prob)

# layer8: softmax
w8_sf = initialize_weight([1024, 10], 0.01, 'w8')
b8_sf = initialize_bias([10])
y_conv = tf.nn.softmax(tf.matmul(h7_drop, w8_sf) + b8_sf)

cross_entropy = -tf.reduce_sum(y * tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
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
train_steps = 5000
batch_size = 50

sess = tf.Session()
sess.run(tf.global_variables_initializer())
for i in range(train_steps):
    batch = next_batch(batch_size)
    if i % 200 == 0:
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
    if (i >= 3000) & (i % 200 == 0):
        predicted_lables = np.zeros(test_images.shape[0])
        for i in range(0,test_images.shape[0]//batch_size):
            predicted_lables[i*batch_size : (i+1)*batch_size] = sess.run(predict, feed_dict = {
                x: test_images[i*batch_size : (i+1)*batch_size], 
                keep_prob: 1.0})
        np.savetxt('submission_softmax'+str(i)+'.csv', 
                   np.c_[range(1, len(test_images)+1), predicted_lables], 
                   delimiter = ',', 
                   header = 'ImageId,Label', 
                   comments = '', 
                   fmt = '%d')
sess.close()
