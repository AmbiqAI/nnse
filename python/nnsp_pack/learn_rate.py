""" 
NoamAnneling learning rate scheduler
"""
import tensorflow as tf
import math
import numpy as np
import matplotlib.pyplot as plt
    
PI=tf.constant(math.pi)
class NoamAnnelingLRSchedule(
    tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    Learning rate scheduler
    """
    def __init__(
            self,
            learning_rate=2.0,
            warmup_steps=10000,
            d_model=176,
            min_lr=1e-6,
            step_shift=0):
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.d_model = d_model
        self.min_lr = min_lr
        self.step_shift = step_shift

    def __call__(self, step):
        step= tf.cast((step+1+self.step_shift),tf.float32)
        warmup_steps = self.warmup_steps
        lr =  self.learning_rate * self.d_model**-0.5 \
            * tf.minimum(step ** (-0.5), step * warmup_steps ** (-1.5))
        return tf.maximum(lr, self.min_lr)

class CosineSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """ Cosine learning rate scheduler """
    def __init__(
            self, base_lr=0.0004,
            end_lr=0,
            warmup_steps=2000,
            total_steps=545*150,
            start_step=0):
        """ Initialize the cosine scheduler """
        super(CosineSchedule, self).__init__()

        self.base_lr = tf.constant(base_lr, dtype=tf.float32)
        self.end_lr = tf.constant(end_lr, dtype=tf.float32)
        self.warmup_steps = tf.constant(warmup_steps, dtype=tf.float32)
        self.total_steps = tf.constant(total_steps, dtype=tf.float32)
        self.start_step = tf.constant(start_step, dtype=tf.float32)

    def __call__(self, step):
        """ Call the learning rate scheduler """
        step = tf.cast(step, tf.float32)
        step = step + self.start_step
        step_s = step - self.warmup_steps
        lr_high = (0.5 + tf.cos( 2.0 * PI / 2.0 *step_s /(self.total_steps-self.warmup_steps)) + 0.5)/2

        lr_high = lr_high * (self.base_lr - self.end_lr) + self.end_lr
        lr_low=step * (self.base_lr) / (self.warmup_steps)

        lr = tf.where(step < self.warmup_steps, lr_low, lr_high)

        return lr

if __name__ == "__main__":
    total_steps = 300* 5 * 150
    lr_gen = CosineSchedule(total_steps=total_steps)
    lrs = np.arange(total_steps, dtype=np.float32)

    outs = lr_gen(lrs)

    plt.plot(outs)
    plt.show()
