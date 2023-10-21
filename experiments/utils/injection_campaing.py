import tensorflow as tf


def gen_batch(x,y,batch_size):
    batch       = [x] * batch_size
    y_batch     = [y] * batch_size
    batch       = tf.stack(batch)
    y_batch     = tf.stack(y_batch)

    return batch,y_batch