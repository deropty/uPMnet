from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import time
import pickle
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import network
import association
import utils
import matplotlib.pyplot as plt
import re

from datetime import datetime
from scipy.io import savemat


# Flags governing network training
tf.app.flags.DEFINE_string('model_name', 'mobilenet_v1',
                           'The name of the architecture to train.')
tf.app.flags.DEFINE_float('weight_decay', 0.00004,
                          'The weight decay on the model weights.')
tf.app.flags.DEFINE_string('pretrained_model_checkpoint_path', '',
                           'If specified, restore this pretrained model (e.g. ImageNet pretrained).')
tf.app.flags.DEFINE_float('ema_decay', 0.9999,
                          'The decay to use for the moving average.')
tf.app.flags.DEFINE_float('grad_clip', 2.0,
                          'The gradient clipping threshold to stabilize the training.')
tf.app.flags.DEFINE_float('label_smoothing', 0.1,
                          'The amount of label smoothing.')
tf.app.flags.DEFINE_string('train_dir', '/tmp/tfmodel/',
                           'Directory where checkpoints and event logs are written to.')
tf.app.flags.DEFINE_string('feature_dir', None,
                           'Directory where the pre-extracted features are stored for anchor initialization.')
# Flags governing data preprocessing
tf.app.flags.DEFINE_integer('num_readers', 4,
                            'The number of parallel readers that read data from the dataset.')
tf.app.flags.DEFINE_integer('num_preprocessing_threads', 4,
                            'The number of threads used to create the batches.')
tf.app.flags.DEFINE_integer('batch_size', 128,
                            'The number of samples in each batch.')
tf.app.flags.DEFINE_integer('max_steps', 1000000,
                            'Number of batches to run.')
# Flags governing the employed hardware
tf.app.flags.DEFINE_integer('num_gpus', 1,
                            'How many GPUs to use.')
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            'Whether to log device placement.')
# Flags governing dataset characteristics
tf.app.flags.DEFINE_string('dataset_name', 'MARS',
                           'The name of the dataset, either "MARS", "PRID2011" or "iLIDS-VID".')
tf.app.flags.DEFINE_string('dataset_dir', None,
                           'The directory where the dataset files are stored.')
tf.app.flags.DEFINE_string('preprocessing_name', 'reidnet',
                           'The name of the preprocessing to use. ')
tf.app.flags.DEFINE_integer('image_size', 224,
                            'Train image size')
tf.app.flags.DEFINE_integer('num_classes', None,
                            'Number of classes.')
tf.app.flags.DEFINE_integer('num_samples', None,
                            'Number of classes.')
tf.app.flags.DEFINE_integer('num_cams', 2,
                            'Number of cameras.')
# Flags governing optimiser
tf.app.flags.DEFINE_string('optimizer', 'rmsprop',
                           'The name of the optimizer, either "sgd" or "rmsprop".')
tf.app.flags.DEFINE_float('initial_learning_rate', 0.045,
                          'Initial learning rate.')
tf.app.flags.DEFINE_float('num_epochs_per_decay', 2.0,
                          'Epochs after which learning rate decays.')
tf.app.flags.DEFINE_float('learning_rate_decay_factor', 0.94,
                          'Learning rate decay factor.')
# Flags governing the anchor learning
tf.app.flags.DEFINE_string('feature_name', 'AvgPool_',
                           'Name of the feature layer.')
tf.app.flags.DEFINE_integer('feature_dim', 1024,
                            'Dimension of feature vector.')
tf.app.flags.DEFINE_integer('warm_up_epochs', 2,
                            'Number of epochs to start tracklet association.')
tf.app.flags.DEFINE_float('margin', 0.5,
                          'Margin of triplet loss.')
tf.app.flags.DEFINE_float('eta', 0.5,
                          'Learning rate to update anchors.')
# display
tf.app.flags.DEFINE_integer('disp_interval', 30,
                            'number of iterations to display.')
# store
tf.app.flags.DEFINE_integer('store_interval', 5000,
                            'number of iterations to display.')
# uPMnet
tf.app.flags.DEFINE_integer('n_part', 0,
                            'Number of parts within each feature')
tf.app.flags.DEFINE_string('relation', 'local',
                            'The type of relation')
# tensorboard
tf.app.flags.DEFINE_boolean('tb', True,
                            'Use tensorboard to debug.')

FLAGS = tf.app.flags.FLAGS
FLAGS.nPart = FLAGS.n_part
print('%s: training use n_part: %d' % (datetime.now(), FLAGS.n_part))

# Constants dictating the learning rate schedule.
RMSPROP_DECAY = 0.9  # Decay term for RMSProp.
RMSPROP_MOMENTUM = 0.9  # Momentum in RMSProp.
RMSPROP_EPSILON = 1.0  # Epsilon term for RMSProp.

# *****************************************************************************************
from datasets import tracket_num
import scipy.io as sio

normalize = lambda v: v ** 2 / (np.sum(v ** 2, 1, keepdims=True))
l2_norm = lambda x: tf.nn.l2_normalize(x, 1, 1e-10)

def init_anchor(anchors_name, cam, num_trackets):
    # initialize a set of anchor under a certain camera
    # the following two ways of initialization lead to similar performance
    if FLAGS.feature_dir:
        # initialize by pre-extracted features
        filename = FLAGS.feature_dir + 'train' + str(cam + 1) + '.mat'
        print('load features ' + filename)
        mat_contents = sio.loadmat(filename)
        train_feature = normalize(mat_contents['train' + str(cam + 1)])
        return tf.get_variable(anchors_name,
                               dtype=tf.float32,
                               initializer=train_feature,
                               trainable=False)
    elif os.path.isdir(FLAGS.pretrained_model_checkpoint_path):
        ckptfile = [re.search(r'\d+.ckpt', i) for i in \
                    os.listdir(FLAGS.pretrained_model_checkpoint_path)]
        global_step = list(set([i.group()[:-5] for i in ckptfile if i is not None]))
        global_step = [int(i) for i in global_step]
        global_step.sort()
        lastanchor = FLAGS.pretrained_model_checkpoint_path + 'anchors_' \
                     + str(global_step[-1]) + '.pkl'
        with open(lastanchor, 'r') as f:
            last_anchors = pickle.load(f)
        k = 0 if anchors_name[:5] == 'intra' else 1
        anchor_value = last_anchors[k][int(anchors_name[-2])][int(anchors_name[-1])]
        return tf.get_variable(anchors_name,
                               [num_trackets, FLAGS.feature_dim],
                               dtype=tf.float32,
                               initializer=tf.constant_initializer(anchor_value),
                               trainable=False)
    else:
        # initialize as 0
        return tf.get_variable(anchors_name,
                               [num_trackets, FLAGS.feature_dim],
                               dtype=tf.float32,
                               initializer=tf.constant_initializer(0),
                               trainable=False)

def get_anchor(reuse_variables):
    num_trackets = tracket_num.get_tracket_num(FLAGS.dataset_name)
    print('number of trackets is '+str(num_trackets))

    # initialize the whole sets of anchors
    # with tf.device('/cpu:0'):
    with tf.variable_scope(tf.get_variable_scope(), reuse=reuse_variables):
        intra_anchors_ = []
        cross_anchors_ = []
        for j in range(FLAGS.nPart):
            intra_anchors = []
            cross_anchors = []
            for i in range(FLAGS.num_cams):
                anchors_name = 'intra_anchors'+ str(j) + str(i)
                intra_anchors.append(init_anchor(anchors_name, i, num_trackets[i]))

                anchors_name = 'cross_anchors' + str(j) + str(i)
                cross_anchors.append(init_anchor(anchors_name, i, num_trackets[i]))
            intra_anchors_.append(intra_anchors)
            cross_anchors_.append(cross_anchors)

    return intra_anchors_, cross_anchors_

# *****************************************************************************************

def _tower_loss(images, labels, cams, num_classes,
                reuse_variables=None, start_sign=0.0):

    # Build inference graph.
    with tf.variable_scope(tf.get_variable_scope(), reuse=reuse_variables):
        _, features = \
            network.inference_with_feature(
                images, num_classes, FLAGS.feature_name)

    # obtain the set of anchors under each camera
    with tf.device('/cpu:0'):
        intra_anchors, cross_anchors = get_anchor(reuse_variables)

    # Build anchor learning graph & compute loss
    # with tf.device('/cpu:0'):
    final_loss_ = 0
    intra_anchors_ = []
    cross_anchors_ = []
    for i in range(FLAGS.nPart):
        final_loss, anchors = \
            association.learning_graph(
                l2_norm(features['AvgPool_%d'%i]), labels, cams, start_sign, intra_anchors[i], cross_anchors[i])
        final_loss_ += final_loss
        intra_anchors_.append(anchors[0])
        cross_anchors_.append(anchors[1])

    # Assemble all of the losses for the current tower only.
    losses =[final_loss_ / FLAGS.nPart]
    regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    total_loss = tf.add_n(losses + regularization_losses, name='total_loss')

    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    loss_averages_op = loss_averages.apply(losses + [total_loss])

    with tf.control_dependencies([loss_averages_op]):
        total_loss = tf.identity(total_loss)

    return total_loss, [intra_anchors_, cross_anchors_]


def stats_graph(graph):
	flops = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.float_operation())
	params = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.trainable_variables_parameter())
	print('FLOPs: {};    Trainable params: {}'.format(flops.total_float_ops, params.total_parameters))

def train():
    with tf.Graph().as_default() as graph, tf.device('/cpu:0'):
        # split the batch across GPUs.
        assert FLAGS.batch_size % FLAGS.num_gpus == 0, (
            'Batch size must be divisible by number of GPUs')

        start_sign_placeholder = tf.placeholder(tf.bool, name='start_sign')

        images, labels, cams = utils.prepare_data('train')

        # Split the batch of images and labels for towers.
        images_splits = tf.split(images, FLAGS.num_gpus, 0)
        labels_splits = tf.split(labels, FLAGS.num_gpus, 0)
        cams_splits = tf.split(cams, FLAGS.num_gpus, 0)

        num_classes = FLAGS.num_classes + 1
        global_step = slim.create_global_step()

        # Create an optimizer that performs gradient descent.
        if FLAGS.optimizer == 'rmsprop':
            # Calculate the learning rate schedule.
            num_batches_per_epoch = (FLAGS.num_samples / FLAGS.batch_size)
            decay_steps = int(num_batches_per_epoch * FLAGS.num_epochs_per_decay)
            # Decay the learning rate exponentially based on the number of steps.
            lr = tf.train.exponential_decay(FLAGS.initial_learning_rate,
                                            global_step,
                                            decay_steps,
                                            FLAGS.learning_rate_decay_factor,
                                            staircase=True)
            opt = tf.train.RMSPropOptimizer(lr, RMSPROP_DECAY,
                                            momentum=RMSPROP_MOMENTUM,
                                            epsilon=RMSPROP_EPSILON)
        elif FLAGS.optimizer =='sgd':
            boundaries = [int(1/2 * float(FLAGS.max_steps))]
            boundaries = list(np.array(boundaries, dtype=np.int64))
            values = [0.01*FLAGS.initial_learning_rate, 0.001*FLAGS.initial_learning_rate]
            lr = tf.train.piecewise_constant(global_step, boundaries, values)
            opt = tf.train.MomentumOptimizer(learning_rate=lr,
                                             momentum=0.9,
                                             use_nesterov=True)

        tower_grads = []
        anchors_op = []
        reuse_variables = None
        for i in range(FLAGS.num_gpus):
            with tf.device('/gpu:%d' % i):
                with tf.name_scope('%s_%d' % (network.TOWER_NAME, i)) as scope:
                    with slim.arg_scope(slim.get_model_variables(scope=scope), device='/cpu:0'):
                        # Calculate the loss for one tower of the model.
                        loss, anchors = \
                            _tower_loss(images_splits[i], labels_splits[i], cams_splits[i],
                                        num_classes, reuse_variables, start_sign_placeholder)

                        anchors_op.append(anchors)
                    # compute FLOPs and Trainable params
                    # stats_graph(graph)
                    # Reuse variables for the next tower.
                    reuse_variables = True

                    batchnorm = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope)
                    batchnorm = [var for var in batchnorm if not 'Logits' in var.name]

                    trainable_var = tf.trainable_variables()
                    trainable_var = [var for var in trainable_var if not 'Logits' in var.name]

                    grads = opt.compute_gradients(loss, var_list=trainable_var)
                    tower_grads.append(grads)

        # synchronize gradients across all towers
        grads = network.average_gradients(tower_grads)
        gradient_op = opt.apply_gradients(grads, global_step=global_step)

        var_averages = tf.train.ExponentialMovingAverage(FLAGS.ema_decay, global_step)
        var_average = tf.trainable_variables()
        if FLAGS.relation == 'global':
            var_average = [var for var in var_average if not 'Logits/Conv2d' in var.name]
        else:
            var_average = [var for var in var_average if not 'Logits' in var.name]
        var_op = var_averages.apply(var_average)

        batchnorm_op = tf.group(*batchnorm)
        train_op = tf.group(gradient_op, var_op, batchnorm_op)

        saver = tf.train.Saver(tf.global_variables(),max_to_keep=None)
        init = tf.global_variables_initializer()

        sess = tf.Session(config=tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=FLAGS.log_device_placement))
        sess.run(init)

        start_step  = 0
        # continue training from existing model
        if os.path.isdir(FLAGS.pretrained_model_checkpoint_path):
            ckptfile = [re.search(r'\d+.ckpt', i) for i in \
                        os.listdir(FLAGS.pretrained_model_checkpoint_path)]
            global_step = list(set([i.group()[:-5] for i in ckptfile if i is not None]))
            global_step = [int(i) for i in global_step]
            global_step.sort()
            start_step = global_step[-1]
            lastckpt = FLAGS.pretrained_model_checkpoint_path + 'model_' \
                       + str(start_step) + '.ckpt-' + str(start_step)
            restorer = tf.train.Saver()
            restorer.restore(sess, lastckpt)
            print('%s: Pre-trained model restored from %s' %(datetime.now(), lastckpt))
        else:
            var_to_restore = [var for var in trainable_var if not 'Logits' in var.name]
            restorer = tf.train.Saver(var_to_restore)
            restorer.restore(sess, FLAGS.pretrained_model_checkpoint_path)
            print('%s: Pre-trained model restored from %s' %
                  (datetime.now(), FLAGS.pretrained_model_checkpoint_path))

        tf.train.start_queue_runners(sess=sess)
        step_1_epoch = int(float(FLAGS.num_samples)/float(FLAGS.batch_size))

        if FLAGS.tb:
            tbWriter = tf.summary.FileWriter(FLAGS.train_dir)
            merged_summary_op = tf.summary.merge_all()

        loss_temp = 0
        for step in range(start_step, FLAGS.max_steps):
            if step % step_1_epoch == 0:
                loss_temp = 0

            start_time = time.time()
            # np_image, _, anchors_value, loss_value = \
            #     sess.run([images, train_op, anchors_op, loss],
            #              feed_dict={start_sign_placeholder:
            #                         step>=step_1_epoch*FLAGS.warm_up_epochs})
            # plt.figure(figsize=(12, 8))
            # for i in range(64):
            #     height, width, _ = np_image[i].shape
            #     plt.subplot(8, 8, i+1)
            #     plt.axis('off')
            #     plt.imshow(np_image[i])
            # plt.show()
            # plt.savefig('%05d.png'%step)
            _, anchors_value, loss_value = \
                sess.run([train_op, anchors_op, loss],
                         feed_dict={start_sign_placeholder:
                                        step >= step_1_epoch * FLAGS.warm_up_epochs})
            duration = time.time() - start_time
            loss_temp += loss_value
            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

            if FLAGS.tb and step % 5 == 0:
                summary = sess.run(merged_summary_op, feed_dict= \
                    {start_sign_placeholder:step >= step_1_epoch * FLAGS.warm_up_epochs})
                tbWriter.add_summary(summary, step)

            if (step+1) % FLAGS.disp_interval == 0:
                if step > 0:
                    loss_temp /= FLAGS.disp_interval
                examples_per_sec = FLAGS.batch_size / float(duration)
                format_str = ('%s: step %d, loss = %.4f '
                              '(%.1f examples/sec; %.3f sec/batch)')
                print(format_str % (datetime.now(), step+1, loss_temp,
                                    examples_per_sec, duration))
                loss_temp = 0
            # if step % (FLAGS.disp_interval*10) == 0:
            #     os.system('nvidia-smi --query-gpu=memory.used,memory.total --format=csv')

            if step != 0 and (step+1) % FLAGS.store_interval == 0:
                # checkpoint_path = os.path.join(FLAGS.train_dir, 'model_%d.ckpt'%(step+1))
                checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
                # anchors_path = os.path.join(FLAGS.train_dir, 'anchors_%d.pkl'%(step+1))
                saver.save(sess, checkpoint_path, global_step=step+1)
                # with open(anchors_path, 'w') as f:
                #     pickle.dump(anchors_value[0], f)

def main(_):
    if not tf.gfile.Exists(FLAGS.train_dir):
        tf.gfile.MakeDirs(FLAGS.train_dir)
    # if tf.gfile.Exists(FLAGS.train_dir):
    #     tf.gfile.DeleteRecursively(FLAGS.train_dir)
    # tf.gfile.MakeDirs(FLAGS.train_dir)
    train()


if __name__ == '__main__':
    tf.app.run()
