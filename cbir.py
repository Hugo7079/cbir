import tensorflow as tf
from keras.layers import Conv2D
from keras.layers import Add
from keras.layers import Input
import numpy as np
import scipy.spatial.distance
from scipy.spatial.distance import cdist
from os import listdir
from os.path import join
import scipy
import cv2
from math import *
import heapq
import time
from six.moves import input

# https://github.com/kentsommer/VGG16-Image-Retrieval/releases/download/v1.0/Dataset_Directory.zip
# https://github.com/kentsommer/VGG16-Image-Retrieval/releases/download/v1.0/vgg16_weights.npz

DOWN_FACTOR = 3
RDB_LAYERS = 3
BATCH_SIZE = 8
EPOCHS = 65
LR = 1e-3

class vgg16:
    def __init__(self, imgs, weights=None, sess=None):
        self.imgs = imgs
        self.convlayers()
        get_subpixel_net
        self.fc_layers()
        self.probs = tf.nn.softmax(self.fc3l)
        if weights is not None and sess is not None:
            self.load_weights(weights, sess)

    def convlayers(self):
        self.parameters = []

        # zero-mean input
        with tf.name_scope('preprocess') as scope:
            # mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
            # images = self.imgs - mean
            images = self.imgs

        # conv1_1
        with tf.name_scope('conv1_1') as scope:
            kernel = tf.Variable(tf.random.truncated_normal([3, 3, 3, 64], dtype=tf.float32, stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32), trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv1_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv1_2
        with tf.name_scope('conv1_2') as scope:
            kernel = tf.Variable(tf.random.truncated_normal([3, 3, 64, 64], dtype=tf.float32, stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv1_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32), trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv1_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool1
        self.pool1 = tf.nn.max_pool2d(self.conv1_2,
                                    ksize=[1, 2, 2, 1],
                                    strides=[1, 2, 2, 1],
                                    padding='SAME',
                                    name='pool1')

        # conv2_1
        with tf.name_scope('conv2_1') as scope:
            kernel = tf.Variable(tf.random.truncated_normal([3, 3, 64, 128], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.pool1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv2_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv2_2
        with tf.name_scope('conv2_2') as scope:
            kernel = tf.Variable(tf.random.truncated_normal([3, 3, 128, 128], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv2_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv2_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool2
        self.pool2 = tf.nn.max_pool2d(self.conv2_2,
                                    ksize=[1, 2, 2, 1],
                                    strides=[1, 2, 2, 1],
                                    padding='SAME',
                                    name='pool2')

        # conv3_1
        with tf.name_scope('conv3_1') as scope:
            kernel = tf.Variable(tf.random.truncated_normal([3, 3, 128, 256], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.pool2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv3_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv3_2
        with tf.name_scope('conv3_2') as scope:
            kernel = tf.Variable(tf.random.truncated_normal([3, 3, 256, 256], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv3_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv3_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv3_3
        with tf.name_scope('conv3_3') as scope:
            kernel = tf.Variable(tf.random.truncated_normal([3, 3, 256, 256], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv3_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv3_3 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool3
        self.pool3 = tf.nn.max_pool2d(self.conv3_3,
                                    ksize=[1, 2, 2, 1],
                                    strides=[1, 2, 2, 1],
                                    padding='SAME',
                                    name='pool3')

        # conv4_1
        with tf.name_scope('conv4_1') as scope:
            kernel = tf.Variable(tf.random.truncated_normal([3, 3, 256, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.pool3, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv4_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv4_2
        with tf.name_scope('conv4_2') as scope:
            kernel = tf.Variable(tf.random.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv4_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv4_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv4_3
        with tf.name_scope('conv4_3') as scope:
            kernel = tf.Variable(tf.random.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv4_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv4_3 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool4
        self.pool4 = tf.nn.max_pool2d(self.conv4_3,
                                    ksize=[1, 2, 2, 1],
                                    strides=[1, 2, 2, 1],
                                    padding='SAME',
                                    name='pool4')

        # conv5_1
        with tf.name_scope('conv5_1') as scope:
            kernel = tf.Variable(tf.random.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.pool4, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv5_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv5_2
        with tf.name_scope('conv5_2') as scope:
            kernel = tf.Variable(tf.random.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv5_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv5_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv5_3
        with tf.name_scope('conv5_3') as scope:
            kernel = tf.Variable(tf.random.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv5_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv5_3 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool5
        self.pool5 = tf.nn.max_pool2d(self.conv5_3,
                                    ksize=[1, 2, 2, 1],
                                    strides=[1, 2, 2, 1],
                                    padding='SAME',
                                    name='pool4')

    def fc_layers(self):
        # fc1
        with tf.name_scope('fc1') as scope:
            shape = int(np.prod(self.pool5.get_shape()[1:]))
            fc1w = tf.Variable(tf.random.truncated_normal([shape, 4096], dtype=tf.float32, stddev=1e-1), name='weights')
            fc1b = tf.Variable(tf.constant(1.0, shape=[4096], dtype=tf.float32), trainable=True, name='biases')
            pool5_flat = tf.reshape(self.pool5, [-1, shape])
            fc1l = tf.nn.bias_add(tf.matmul(pool5_flat, fc1w), fc1b)
            self.fc1 = tf.nn.relu(fc1l)
            self.parameters += [fc1w, fc1b]

        # fc2
        with tf.name_scope('fc2') as scope:
            fc2w = tf.Variable(tf.random.truncated_normal([4096, 4096],
                                                   dtype=tf.float32,
                                                   stddev=1e-1), name='weights')
            fc2b = tf.Variable(tf.constant(1.0, shape=[4096], dtype=tf.float32),
                               trainable=True, name='biases')
            fc2l = tf.nn.bias_add(tf.matmul(self.fc1, fc2w), fc2b)
            self.fc2 = tf.nn.relu(fc2l)
            self.parameters += [fc2w, fc2b]

        # fc3
        with tf.name_scope('fc3') as scope:
            fc3w = tf.Variable(tf.random.truncated_normal([4096, 1000],
                                                   dtype=tf.float32,
                                                   stddev=1e-1), name='weights')
            fc3b = tf.Variable(tf.constant(1.0, shape=[1000], dtype=tf.float32),
                               trainable=True, name='biases')
            self.fc3l = tf.nn.bias_add(tf.matmul(self.fc2, fc3w), fc3b)
            self.parameters += [fc3w, fc3b]

    def load_weights(self, weight_file, sess):
        weights = np.load(weight_file)
        keys = sorted(weights.keys())
        print('Load weights...')
        for i, k in enumerate(keys):
            sess.run(self.parameters[i].assign(weights[k]))
        print('Load complete.')

    def square_rooted(x):
        return round(sqrt(sum([a * a for a in x])), 3)

    def cosine_similarity(x, y):
        numerator = sum(a * b for a, b in zip(x, y))
        denominator = square_rooted(x) * square_rooted(y)
        return round(numerator / float(denominator), 3)

def rdb_block(inputs, numlayers):
    channels = inputs.get_shape()[-1]
    storedoutputs = [inputs]
    for _ in range(numlayers):
        localConcat = tf.concat(storedoutputs, axis=-1)
        out = Conv2D(filters=channels, kernel_size=3, padding="same",
                     activation="relu",
                     kernel_initializer="Orthogonal")(localConcat)
        storedoutputs.append(out)

        # concatenate all the outputs, pass it through a pointwise
        # convolutional layer, and add the outputs to initial inputs
        finalConcat = tf.concat(storedoutputs, axis=-1)
        finalOut = Conv2D(filters=inputs.get_shape()[-1], kernel_size=1,
                          padding="same", activation="relu",
                          kernel_initializer="Orthogonal")(finalConcat)
        finalOut = Add()([finalOut, inputs])
        # return the final output
        return finalOut

def get_subpixel_net(downsampleFactor=DOWN_FACTOR, channels=1, rdbLayers=RDB_LAYERS):
    # initialize an input layer
    inputs = Input((256, 256, 3))
    # pass the inputs through a CONV => CONV block
    x = Conv2D(64, 5, padding="same", activation="relu",
        kernel_initializer="Orthogonal")(inputs)
    x = Conv2D(64, 3, padding="same", activation="relu",
        kernel_initializer="Orthogonal")(x)
    # pass the outputs through an RDB => CONV => RDB block
    x = rdb_block(x, numlayers=rdbLayers)
    x = Conv2D(32, 3, padding="same", activation="relu",
        kernel_initializer="Orthogonal")(x)
    x = rdb_block(x, numlayers=rdbLayers)
    x = Conv2D(channels * (downsampleFactor ** 2), 3, padding="same",
               activation="relu", kernel_initializer="Orthogonal")(x)
    outputs = tf.nn.depth_to_space(x, downsampleFactor)
    # construct the final model and return it
    # model = Model(inputs, outputs)
    return outputs

if __name__ == '__main__':
    # Get number of images to match (default 4)
    num = 500
    dist_type = input("Enter distance algorithm (euc, cos, chev, color): \n") or "euc"
    print("distance type selected: " + dist_type)
    # Setup Session
    sess = tf.compat.v1.Session()
    tf.compat.v1.disable_v2_behavior()
    imgs = tf.compat.v1.placeholder(tf.float32, [None, 224, 224, 3])
    vgg = vgg16(imgs, 'D:\\Pycharm_Project\\Final_Project\\cbir\\vgg16_cbir\\vgg16_weights.npz', sess)

    # Set dataset directory path
    data_dir_query = 'D:\\Pycharm_Project\\Final_Project\\cbir\\human_query2'
    data_dir_test = 'D:\\Pycharm_Project\\Final_Project\\cbir\\humans100k_jpg\\photos'
    datalist_query = [join(data_dir_query, f) for f in listdir(data_dir_query)]
    datalist = [join(data_dir_test, f) for f in listdir(data_dir_test)]

    ####################
    ###Perform Search###
    ####################

    # Timer and precision count total + open file for saving data
    t0 = time.time()
    shouldCompute = True
    feat_dict = {}
    total_precision_cn = 0
    fp = open("Last_Run.txt", 'w')
    fp.truncate()

    # Retrieve feature vector for query image
    for i in datalist_query:
        # Setup Dict and precision tracking
        img_dict = {}
        cor_list = []
        img_query = cv2.imread(i)
        img_query = cv2.resize(img_query, (224, 224))

        # Extract image descriptor in layer fc2/Relu. If you want, change fc2 to fc1
        layer_query = sess.graph.get_tensor_by_name('fc2/Relu:0')
        # layer_query = sess.graph.get_tensor_by_name('fc1/Relu:0')
        # Run the session for feature extract at 'fc2/Relu' layer
        _feature_query = sess.run(layer_query, feed_dict={vgg.imgs: [img_query]})
        # Convert tensor variable into numpy array
        # It is 4096 dimension vector
        feature_query = np.array(_feature_query)

        img_query_color = np.array(img_query)
        img_query_color = img_query_color.reshape(1, -1)

        # Retrieve feature vector for test image
        count = 0
        # print(datalist_test)
        # print(datalist[2])

        index = []

        for j in datalist:
            if count < num:
                if count % 500 == 0:
                    print(count)
                if shouldCompute:
                    img_test = cv2.imread(j)
                    img_test = cv2.resize(img_test, (224, 224))

                    # Extract image descriptor in layer fc2/Relu. If you want, change fc2 to fc1
                    layer_test = sess.graph.get_tensor_by_name('fc2/Relu:0')
                    # layer_test = sess.graph.get_tensor_by_name('fc1/Relu:0')
                    # Run the session for feature extract at 'fc2/Relu' layer
                    _feature_test = sess.run(layer_test, feed_dict={vgg.imgs: [img_test]})
                    # Convert tensor variable into numpy array
                    # It is 4096 dimension vector
                    feature_test = np.array(_feature_test)

                    feat_dict[j] = feature_test

                    img_test_color = np.array(img_test)
                    img_test_color = img_test_color.reshape(1, -1)

                else:
                    feature_test = feat_dict[j]

                # Calculate Euclidean distance between two feature vectors
                if dist_type == "euc":
                    curr_dist = scipy.spatial.distance.euclidean(feature_query, feature_test)
                # Calculate Cosine distance between two feature vectors
                if dist_type == "cos":
                    curr_dist = scipy.spatial.distance.cosine(feature_query, feature_test)
                # Calculate Chevyshev distance between two feature vectors
                if dist_type == "chev":
                    curr_dist = scipy.spatial.distance.chebyshev(feature_query, feature_test)
                # color
                if dist_type == "color":
                    curr_dist_r = cdist(img_query_color, img_test_color)
                    curr_dist_g = cdist(img_query_color, img_test_color)
                    curr_dist_b = cdist(img_query_color, img_test_color)
                    curr_dist = float(curr_dist_r) + float(curr_dist_g) + float(curr_dist_g)

                # Add to dictionary
                img_dict[curr_dist] = str(j)
                count = count + 1

        # Get Results for Query
        for k in range(0, 5):
            keys_sorted = heapq.nsmallest(num, img_dict)
            print(img_dict[keys_sorted[0]])
            img = cv2.imread(img_dict[keys_sorted[0]])
            img = np.array(img).astype('uint8')
            cv2.imshow('image', img)
            keyValue = cv2.waitKey(0)
            if keyValue & 0xFF == ord('q'):
                img_dict.pop(keys_sorted[0])
                continue
        cv2.destroyAllWindows()
        shouldCompute = False
