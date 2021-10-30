import tensorflow as tf


def weighted_loss(label, logits):
    logits = tf.sigmoid(logits)
    positive_wire_mask = tf.cast(label, tf.bool)
    negative_wire_mask = tf.logical_not(positive_wire_mask)
    wire_acc = tf.reduce_sum(1. - tf.boolean_mask(logits, positive_wire_mask))
    no_wire_acc = tf.reduce_sum(tf.boolean_mask(logits, negative_wire_mask))
    t = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=logits))
    loss = 5 * wire_acc + no_wire_acc + t * 100
    return loss


def weighted_loss1(label, logits):
    logit = tf.sigmoid(logits)
    t1 = tf.reduce_mean(tf.losses.mean_squared_error(label[..., 1], logit[..., 1]))
    t2 = tf.reduce_mean(tf.losses.mean_squared_error(label[..., 0], logit[..., 0]))
    loss = t1 * 100 + t2
    return loss
