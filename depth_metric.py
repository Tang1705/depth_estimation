# from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import numpy as np

def compute_errors(gt, pred):
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25   ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()
    abs_rel = np.mean(np.abs(gt - pred) / gt)
    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())
    log_10 = (np.abs(np.log10(gt)-np.log10(pred))).mean()
    return a1, a2, a3, abs_rel, rmse, log_10

def log10(x):
    numerator = tf.log(x)
    denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator

def metric_log(gt, pred):
    gt = log10(gt)
    pred = log10(pred)
    metric_log = tf.reduce_mean(tf.abs(gt - pred))
    return metric_log
    
def metric_delta(gt, pred):
    threshold_v = tf.maximum(tf.div(gt, pred + 1e-3), tf.div(pred, gt + 1e-3))
    zeros_matrix = tf.zeros_like(gt)
    delta_1 = tf.reduce_mean(tf.where(threshold_v < 1.25, threshold_v, zeros_matrix))
    delta_2 = tf.reduce_mean(tf.where(threshold_v < 1.25 ** 2, threshold_v, zeros_matrix))
    delta_3 = tf.reduce_mean(tf.where(threshold_v < 1.25 ** 3, threshold_v, zeros_matrix))
    return delta_1, delta_2, delta_3

def metric_rel(gt, pred):
    abs_rel = tf.reduce_mean(tf.abs(tf.div((gt - pred), gt + 1e-3)))
    return abs_rel

def metric_rmse(gt, pred):
    rmse = tf.sqrt(tf.reduce_mean(tf.square(gt - pred)))
    return rmse

def main():
    gt = np.array([[1.0, 2],[3.0, 4]])
    pred = np.array([[2.1, 4.1],[3.1, 4.1]])
    a1, a2, a3, abs_rel, rmse, log_10 = compute_errors(gt, pred)
    print(a1, a2, a3, abs_rel, rmse, log_10)
    
    gt = tf.constant([[1, 2], [3, 4]], tf.float32)
    pred = tf.constant([[2.1, 4.1], [3.1, 4.1]], tf.float32)

    # delta_1, delta_2, delta_3
    delta_1, delta_2, delta_3 = metric_delta(gt, pred)

    # rel_error
    abs_rel = metric_rel(gt, pred)

    # rmse
    rmse = metric_rmse(gt, pred)
    
    # log metric
    log_10 = metric_log(gt, pred)

    init_op = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init_op)
        print(sess.run([delta_1, delta_2, delta_3, rmse, abs_rel, log_10]))
    
    return 0

if __name__ == '__main__':
    main()
