import numpy as np
import tensorflow as tf

class statsClass:
    """ statistics class for the neural network"""
    def __init__(self, dim_target):
        self.stats = {'acc_loss'        : tf.Variable(0, trainable = False, dtype = tf.float64),
                      'acc_steps'       : tf.Variable(0, trainable = False, dtype = tf.float64),
                      'acc_matchCount'  : tf.Variable(0, trainable = False, dtype = tf.float64)
                      }

        self.confusion_mat = tf.zeros(
            (dim_target , dim_target), dtype=tf.float64)
    def update_cost_steps(self, ave_loss, steps):
        """
        update total cost and total steps
        """
        stats = self.stats
        ave_loss = tf.cast(ave_loss, tf.float64)
        steps = tf.cast(steps, tf.float64)
        stats['acc_loss'].assign(stats['acc_loss'] + ave_loss * steps)
        stats['acc_steps'].assign(stats['acc_steps'] + steps)

    def update_accuracy(self, trgt0, est0, mask, dim):
        """
        Update total correct number of estimation
        """
        tmp = tf.reduce_sum(
                tf.cast(tf.math.equal(est0, trgt0), tf.float64)
                 * tf.cast(mask[:,:,0], tf.float64))
        self.stats['acc_matchCount'].assign_add(tf.cast(tmp, tf.float64))

        mat = [None] * dim * dim
        mask = tf.cast(mask, tf.int64)[:,:,0]
        for i in range(dim):
            est1 = est0 * mask
            trgt1 = trgt0 * mask
            mask_i = tf.cast(tf.math.equal(trgt1, i), dtype = tf.int64) * mask
            for j in range(dim):
                tmp = tf.cast(tf.math.equal(est1, j), dtype = tf.int64) * mask_i
                mat[i * dim + j] = tf.reduce_sum(tmp)
        mat = tf.convert_to_tensor(mat)
        mat = tf.reshape(mat, (dim, dim))

        self.confusion_mat = self.confusion_mat + tf.cast(mat, tf.float64)

    def show_loss(
            self,
            # loss,
            # total_accuracy,
            # steps,
            dim_target=257,
            show_accuracy=False,
            show_step=True, 
            lr=None):
        """
        Print the loss function
        """
        loss  = tf.cast(self.stats['acc_loss'],  tf.float64)
        steps = tf.cast(self.stats['acc_steps'], tf.float64)
        loss0 = tf_round( loss / steps / dim_target )
        accuracy0 = tf_round( self.stats['acc_matchCount'] / steps )

        tf.print('loss: ', loss0, end = '')
        if show_accuracy:
            tf.print(', accuracy: ', accuracy0, end = '')
        if lr is not None:
            tf.print(', lr: ', lr, end = '')
        if show_step:
            tf.print( ' (', steps, ')', end = '')

    def show_confusion_matrix(self, dim_intent, logger):
        """
        Print the confusion matrix
        """
        mat = self.confusion_mat
        string = ''
        string += 'confused Matrix \n'
        for i in range(dim_intent):
            tmp = 0
            # tf.print('\tclass %d: ' % i, end = '')
            for j in range(dim_intent):
                tmp = tmp + mat[i][j]
            for j in range(dim_intent):
                mat_ij = tf_round(mat[i][j] / tmp)
                string += f'{mat_ij:2.2f} '
            string+='\n'
        logger.info(string)

    def reset_stats(self):
        """
        Reset the internal states of the neural net
        """
        for _, value in self.stats.items():
            value.assign(0)

        self.confusion_mat = self.confusion_mat * 0

def tf_round(data_in, prec = 10):
    """
    Round the tensor
    """
    base = tf.pow(tf.constant(10.0, dtype = data_in.dtype), prec)
    return tf.round(data_in * base) / base
