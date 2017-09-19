# coding=utf-8:
from __future__ import absolute_import, division, print_function
import tensorflow as tf
from _base_model.model import *


def weight_variable(shape, name, stddev=0.01):
    initializer = tf.truncated_normal_initializer(stddev=stddev)
    var = tf.get_variable(name=name, shape=shape, initializer=initializer)
    return var


def bias_variable(shape, name, init_value=0.0):
    initializer = tf.constant_initializer(value=init_value)
    return tf.get_variable(name=name, shape=shape, initializer=initializer)


def convolution(x, W_shape, s_h, s_w, padding='SAME', name=None):
    """

    :param x: input
    :param W: filter_shape:[filter_high, filter_wight, filter_depth, num_channel]
    :param s_h: strides_high
    :param s_w: strides_wight
    :param padding: Padding
    :param name: name
    :return: output
    """

    def conv(i, f):
        """
        定义一个卷积操作，方便之后的命名
        :param i: input
        :param f: filter_shape
        :return: output of convolution layer
        """
        return tf.nn.conv2d(input=i, filter=f, strides=[1, s_h, s_w, 1], padding=padding)

    with tf.name_scope(name=name):
        with tf.variable_scope(name) as scope:
            with tf.name_scope(name='weights'):
                weights = weight_variable(shape=W_shape, name='W')
                tf.summary.histogram(name=name + '/weights', values=weights)
            output = conv(i=x, f=weights)
            with tf.name_scope(name='biases'):
                bias = bias_variable(shape=[W_shape[-1]], name='b')  # bias_shape:num_channel
                tf.summary.histogram(name=name + '/biases', values=bias)
            output = tf.nn.bias_add(value=output, bias=bias)
            tf.summary.histogram(name=name + '/output', values=output)
    return output


def max_pool(x, k_h, k_w, s_h, s_w, padding='SAME', name=None):
    """

    :param x: input
    :param k_h: kernel_high
    :param k_w: kernel_wight
    :param s_h: strides_high
    :param s_w: strides_wight
    :param padding: Padding
    :param name: name
    :return: output
    """
    with tf.name_scope(name=name):
        output = tf.nn.max_pool(value=x, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding,
                                name=name)
        tf.summary.histogram(name=name + 'output', values=output)
    return output


def fully_connected(x, output_size, fixed_weight=None, init_b=0.1, stddev=0.01, name=None):
    """

    :param x: input
    :param output_size: output_size
    :param init_b: initial biases
    :param stddev: stddev
    :return: output
    """
    x_shape = x.get_shape().as_list()  # Get the shape of a Tensor.
    input_dim = x_shape[-1]

    with tf.name_scope(name=name):
        with tf.variable_scope(name) as scope:
            # scope.reuse_variables()
            with tf.name_scope(name='weights'):
                if fixed_weight is None:
                    W = weight_variable(shape=[input_dim, output_size], stddev=stddev, name='W')
                    print(W.name)
                else:
                    W = fixed_weight
                tf.summary.histogram(name=name + '/weights', values=W)
            with tf.name_scope(name='biases'):
                b = bias_variable(shape=[output_size], init_value=init_b, name='b')
                tf.summary.histogram(name=name + '/biases', values=b)
            output = tf.nn.xw_plus_b(x=x, weights=W, biases=b)
            tf.summary.histogram(name=name + '/output', values=output)
            return output


# def dropout(x, keep_prob, is_train):
#     if is_train:
#         result = tf.nn.dropout(x=x, keep_prob=keep_prob)
#         return result





class cnn(BaseModel):
    def build(self):
        self.build_cnn()

    def build_cnn(self):  # image_shape = [batch_size, image_size, image_size, image_channel]
        img_size = self.img_size
        num_channel = self.num_channel
        num_class = self.num_class  # 标签的总数量
        batch_size = self.batch_size
        is_train = self.is_train
        with tf.name_scope('inputs'):
            images = tf.placeholder(dtype=tf.float32, shape=[batch_size, img_size, img_size, num_channel])
            labels_data = tf.placeholder(dtype=tf.float32, shape=[batch_size, num_class])  # 用于存储已经预测的标签

        conv1_features = convolution(x=images, W_shape=[11, 11, 3, 64], s_h=4, s_w=4, name='conv1')
        conv1_features = tf.nn.relu(features=conv1_features)
        lrn1 = tf.nn.lrn(input=conv1_features, depth_radius=4, bias=1.0, alpha=0.001/9, beta=0.75, name='lrn1')
        pool1_features = max_pool(x=lrn1, k_h=3, k_w=3, s_h=2, s_w=2, padding='VALID', name='pool1')

        conv2_features = convolution(x=pool1_features, W_shape=[5, 5, 64, 192], s_h=1, s_w=1, name='conv2')
        conv2_features = tf.nn.relu(features=conv2_features)
        lrn2 = tf.nn.lrn(input=conv2_features, depth_radius=4, bias=1.0, alpha=0.001/9, beta=0.75, name='lrn2')
        pool2_features = max_pool(x=lrn2, k_h=3, k_w=3, s_h=2, s_w=2, padding='VALID', name='pool2')

        conv3_features = convolution(x=pool2_features, W_shape=[3, 3, 192, 384], s_h=1, s_w=1, name='conv3')
        conv3_features = tf.nn.relu(features=conv3_features)

        conv4_features = convolution(x=conv3_features, W_shape=[3, 3, 384, 256], s_h=1, s_w=1, name='conv4')
        conv4_features = tf.nn.relu(conv4_features)

        conv5_features = convolution(x=conv4_features, W_shape=[3, 3, 256, 256], s_h=1, s_w=1, name='conv5')
        conv5_features = tf.nn.relu(features=conv5_features)
        pool5_features = max_pool(x=conv5_features, k_h=3, k_w=3, s_h=2, s_w=2, padding='VALID', name='pool5')

        pool5_features_flatten = tf.reshape(tensor=pool5_features, shape=[self.batch_size, -1])

        fc6_features = fully_connected(x=pool5_features_flatten, output_size=1024, name='fc6')
        fc6_features = tf.nn.relu(fc6_features)
        if is_train:
            fc6_features = tf.nn.dropout(x=fc6_features, keep_prob=0.6)

        fc7_features = fully_connected(x=fc6_features, output_size=num_class, name='fc7')
        fc7_features = tf.nn.sigmoid(fc7_features)

        with tf.name_scope('cost'):
            cost = (-1/batch_size) * tf.reduce_sum(self.logistic_loss(y_hat=fc7_features, y=labels_data))
            tf.summary.scalar(name='cost', tensor=cost)
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(cost)
        _, topk_idx = tf.nn.top_k(fc7_features, k=self.top_k)

        self.cost = cost
        self.optimizer = optimizer
        self.topk_idx = topk_idx
        self.images = images
        self.labels_data = labels_data
        self.init = tf.global_variables_initializer()


    def get_feed_dict(self, batch, is_train):
        if is_train:
            images_data, labels_data = batch

            return {self.images:images_data, self.labels_data:labels_data}

        else:
            images_data, labels_data = batch
            return {self.images:images_data, self.labels_data:labels_data}

    def logistic_loss(self, y_hat, y):
        result = tf.multiply(y, tf.log(y_hat)) + tf.multiply((1 - y), tf.log(1 - y_hat))
        self.result = result
        return result


