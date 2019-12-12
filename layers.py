import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer_conv2d


def conv_layer(x, filtershape, stride, name, activation='prelu', reuse=False):
    with tf.variable_scope(name) as layer:
        if reuse:
            layer.reuse_variables()
        w = tf.get_variable(name='w',
                            shape=filtershape,
                            dtype=tf.float32,
                            initializer=tf.random_normal_initializer(mean=0, stddev=0.001))
        b = tf.get_variable(name='b',
                            shape=[filtershape[-1]],
                            dtype=tf.float32,
                            initializer=tf.zeros_initializer())
        conv = tf.nn.conv2d(input=x,
                            filter=w,
                            strides=[1, stride, stride, 1],
                            padding='SAME')
        if activation == 'prelu':
            output = prelu(conv + b)
        elif activation == 'linear':
            output = conv + b
        else:
            output = tf.nn.leaky_relu(conv + b)
        return output


def atrous_conv_layer(x, filtershape, rate, stride, name, activation='linear', reuse=False):
    with tf.variable_scope(name) as layer:
        if reuse:
            layer.reuse_variables()
        w = tf.get_variable(name='w',
                            shape=filtershape,
                            dtype=tf.float32,
                            initializer=tf.random_normal_initializer(mean=0, stddev=0.001))
        b = tf.get_variable(name='b',
                            shape=[filtershape[-1]],
                            dtype=tf.float32,
                            initializer=tf.zeros_initializer())
        conv = tf.nn.atrous_conv2d(value=x, filters=w, rate=rate, padding='SAME', name='atr_conv')
        if activation == 'prelu':
            output = prelu(conv + b)
        elif activation == 'linear':
            output = conv + b
        else:
            output = tf.nn.leaky_relu(conv + b)
        return output


def deconv_layer(x, filtershape, output_shape, stride, name, act='linear'):
    with tf.variable_scope(name):
        filters = tf.get_variable(
            name='weight',
            shape=filtershape,
            dtype=tf.float32,
            initializer=tf.random_normal_initializer(mean=0, stddev=0.001),
            trainable=True)

        deconv = tf.nn.conv2d_transpose(x, filters, output_shape, [1, stride, stride, 1], padding='SAME')

        if act == 'linear':
            return deconv
        elif act == 'relu':
            return tf.nn.relu(deconv)
        elif act == 'prelu':
            return prelu(deconv)
        else:
            return tf.nn.leaky_relu(deconv)


def max_pool_layer(x, filtershape, stride, name):
    return tf.nn.max_pool(x, filtershape, [1, stride, stride, 1], padding='SAME', name=name)


def prelu(x):
    with tf.variable_scope('prelu'):
        alphas = tf.get_variable('alpha', x.get_shape()[-1],
                                 initializer=tf.constant_initializer(0.0),
                                 dtype=tf.float32)
        pos = tf.nn.relu(x)
        neg = alphas * (x - tf.abs(x)) * 0.5
        return pos + neg


def RFB(input, in_dim, output_dim, stride, name='RFB'):
    with tf.name_scope(name):
        l1 = conv_layer(input, [1, 1, in_dim, output_dim], stride, name + '_l1', activation='linear')
        l2 = conv_layer(input, [1, 1, in_dim, output_dim], stride, name + '_l2', activation='linear')
        l3 = conv_layer(input, [1, 1, in_dim, output_dim], stride, name + '_l3', activation='linear')
        l1 = conv_layer(l1, [5, 5, output_dim, output_dim], stride, name + '_l11', activation='linear')
        l2 = conv_layer(l2, [3, 3, output_dim, output_dim], stride, name + '_l22', activation='linear')
        l1 = atrous_conv_layer(l1, [3, 3, output_dim, output_dim], 5, 1, name + 'atr_l1', activation='linear')
        l2 = atrous_conv_layer(l2, [3, 3, output_dim, output_dim], 3, 1, name + 'atr_l2', activation='linear')
        l3 = atrous_conv_layer(l3, [3, 3, output_dim, output_dim], 1, 1, name + 'atr_l3', activation='linear')
        concat = tf.concat([l1, l2, l3], -1)
        output = conv_layer(concat, [1, 1, 3 * output_dim, output_dim], 1, name + '_conv_final', activation='linear')
        if output_dim == in_dim:
            shortcut = input
        else:
            shortcut = conv_layer(input, [1, 1, in_dim, output_dim], 1, '_shortcut', activation='linear')

        return tf.nn.relu(output + shortcut)
