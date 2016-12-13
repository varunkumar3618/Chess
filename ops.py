import tensorflow as tf

def affine_layer(input_t, input_dim, output_dim, w_init, b_init, activation, name):
    w = tf.get_variable("w_" + name, shape=[input_dim, output_dim], initializer=w_init, dtype=tf.float32)
    b = tf.get_variable("b_" + name, shape=[output_dim], initializer=b_init, dtype=tf.float32)
    return activation(tf.matmul(input_t, w) + b, name=name)
