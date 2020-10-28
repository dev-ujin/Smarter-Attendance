#!/usr/bin/python3
# -*- coding: utf-8 -*-



import tensorflow as tf
from distutils.version import LooseVersion
class LayerFactory(object):
    AVAILABLE_PADDINGS = ('SAME', 'VALID')
    def __init__(self, network):
        self.__network = network
    @staticmethod
    def __validate_padding(padding):
        if padding not in LayerFactory.AVAILABLE_PADDINGS:
            raise Exception("Padding {} not valid".format(padding))

    @staticmethod
    def __validate_grouping(channels_input: int, channels_output: int, group: int):
        if channels_input % group != 0:
            raise Exception("The number of channels in the input does not match the group")

        if channels_output % group != 0:
            raise Exception("The number of channels in the output does not match the group")

    @staticmethod
    def vectorize_input(input_layer):
        input_shape = input_layer.get_shape()

        if input_shape.ndims == 4: # 공간입력을 벡터로
            dim = 1
            for x in input_shape[1:].as_list():
                dim *= int(x)
            vectorized_input = tf.reshape(input_layer, [-1, dim])
        else:
            vectorized_input, dim = (input_layer, input_shape[-1].value)

        return vectorized_input, dim

    def __make_var(self, name: str, shape: list):
        return tf.get_variable(name, shape, trainable=self.__network.is_trainable())

    def new_feed(self, name: str, layer_shape: tuple):
        feed_data = tf.placeholder(tf.float32, layer_shape, 'input')
        self.__network.add_layer(name, layer_output=feed_data)

    def new_conv(self, name: str, kernel_size: tuple, channels_output: int,
                 stride_size: tuple, padding: str='SAME',
                 group: int=1, biased: bool=True, relu: bool=True, input_layer_name: str=None):

        self.__validate_padding(padding)
        input_layer = self.__network.get_layer(input_layer_name)
        channels_input = int(input_layer.get_shape()[-1])

        self.__validate_grouping(channels_input, channels_output, group)
        convolve = lambda input_val, kernel: tf.nn.conv2d(input_val, kernel, [1, stride_size[1], stride_size[0], 1],
                                                          padding=padding)
        with tf.variable_scope(name) as scope:
            kernel = self.__make_var('weights', shape=[kernel_size[1], kernel_size[0], channels_input // group, channels_output])

            output = convolve(input_layer, kernel)
            if biased:
                biases = self.__make_var('biases', [channels_output])
                output = tf.nn.bias_add(output, biases)
            if relu:
                output = tf.nn.relu(output, name=scope.name)


        self.__network.add_layer(name, layer_output=output)

    def new_prelu(self, name: str, input_layer_name: str=None):
        input_layer = self.__network.get_layer(input_layer_name)

        with tf.variable_scope(name):
            channels_input = int(input_layer.get_shape()[-1])
            alpha = self.__make_var('alpha', shape=[channels_input])
            output = tf.nn.relu(input_layer) + tf.multiply(alpha, -tf.nn.relu(-input_layer))

        self.__network.add_layer(name, layer_output=output)

    def new_max_pool(self, name:str, kernel_size: tuple, stride_size: tuple, padding='SAME',
                     input_layer_name: str=None):
        self.__validate_padding(padding)
        input_layer = self.__network.get_layer(input_layer_name)
        output = tf.nn.max_pool(input_layer,
                                ksize=[1, kernel_size[1], kernel_size[0], 1],
                                strides=[1, stride_size[1], stride_size[0], 1],
                                padding=padding,
                                name=name)
        self.__network.add_layer(name, layer_output=output)

    def new_fully_connected(self, name: str, output_count: int, relu=True, input_layer_name: str=None):
        with tf.variable_scope(name):
            input_layer = self.__network.get_layer(input_layer_name)
            vectorized_input, dimension = self.vectorize_input(input_layer)
            weights = self.__make_var('weights', shape=[dimension, output_count])
            biases = self.__make_var('biases', shape=[output_count])
            operation = tf.nn.relu_layer if relu else tf.nn.xw_plus_b
            fc = operation(vectorized_input, weights, biases, name=name)
        self.__network.add_layer(name, layer_output=fc)

    def new_softmax(self, name, axis, input_layer_name: str=None):
        input_layer = self.__network.get_layer(input_layer_name)
        if LooseVersion(tf.__version__) < LooseVersion("1.5.0"):
            max_axis = tf.reduce_max(input_layer, axis, keep_dims=True)
            target_exp = tf.exp(input_layer - max_axis)
            normalize = tf.reduce_sum(target_exp, axis, keep_dims=True)
        else:
            max_axis = tf.reduce_max(input_layer, axis, keepdims=True)
            target_exp = tf.exp(input_layer - max_axis)
            normalize = tf.reduce_sum(target_exp, axis, keepdims=True)

        softmax = tf.div(target_exp, normalize, name)
        self.__network.add_layer(name, layer_output=softmax)

