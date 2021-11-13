import os,sys
import tensorflow as tf
import numpy as np
from skimage import io
from matplotlib import pyplot as plt
import cv2

from nets import nets_factory
from preprocessing import preprocessing_factory
from datetime import datetime
import re

flags = tf.app.flags
flags.DEFINE_string("input", "images/cat.jpg", "Path to input image ['images/cat.jpg']")
flags.DEFINE_string("output", "output.png", "Path to output image ['output.png']")
flags.DEFINE_string("layer_name", None, "Layer till which to backpropagate")
flags.DEFINE_string("model_name", "resnet_v2_50", "Name of the model")
flags.DEFINE_string("preprocessing_name", 'reidnet', "Name of the image preprocessor")
flags.DEFINE_integer("image_size", 256, "Resize images to this size before eval")
flags.DEFINE_string("steps", None, "Resize images to this size before eval")
flags.DEFINE_integer("num_classes", None, "Number of classes for the network")
flags.DEFINE_string("dataset_dir", "./imagenet", "Location of the labels.txt")
flags.DEFINE_string("checkpoint_path", "./imagenet/resnet_v2_50.ckpt", "saved weights for model")
flags.DEFINE_integer("label_offset", 1, "Used for imagenet with 1001 classes for background class")
flags.DEFINE_integer('n_part', 0,'NumbPer of parts within each feature')
flags.DEFINE_string('relation', 'local', 'The type of relation')
flags.DEFINE_float('ema_decay', 0.9999, 'The decay to use for the moving average.')


FLAGS = flags.FLAGS
FLAGS.nPart = FLAGS.n_part
print('%s: training use n_part: %d' % (datetime.now(), FLAGS.n_part))

slim = tf.contrib.slim

_layer_names = { "resnet_v2_50":       ["PrePool","predictions"],
                 "resnet_v2_101":       ["PrePool","predictions"],
                 "resnet_v2_152":       ["PrePool","predictions"],
                 }

_logits_name = 'AvgPool_'

def load_labels_from_file(dataset_dir):
  labels = {}
  labels_name = os.path.join(dataset_dir,'labels.txt')
  with open(labels_name) as label_file:
    for line in label_file:
      idx,label = line.rstrip('\n').split(':')
      labels[int(idx)] = label
  assert len(labels) > 1
  return labels


def load_image(img_path):
  print("Loading image")
  img = cv2.imread(img_path)
  if img is None:
    sys.stderr.write('Unable to load img: %s\n' % img_path)
    sys.exit(1)
  img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
  return img


def preprocess_image(image):
  preprocessing_name = FLAGS.preprocessing_name
  image_preprocessing_fn = preprocessing_factory.get_preprocessing(
    preprocessing_name, is_training=False)
  image = image_preprocessing_fn(image, FLAGS.image_size, int(FLAGS.image_size/2))
  return image

def grad_cam(img, imgs0, end_points, sess, layer_name):
  # Conv layer tensor [?,10,10,2048]
  conv_layer = end_points[layer_name]
  # [1000]-D tensor with target class index set to 1 and rest as 0
  # one_hot = tf.sparse_to_dense(predicted_class, [nb_classes], 1.0)
  # signal = tf.multiply(end_points[_logits_name], one_hot)
  # loss = tf.reduce_mean(signal)

  features = {i: tf.squeeze(tf.squeeze(j, squeeze_dims=1), squeeze_dims=1)
              for i, j in end_points.items() if i[:8] == _logits_name}
  loss = sum([tf.reduce_mean(j) for i,j in features.items()])

  grads = tf.gradients(loss, conv_layer)[0]
  # Normalizing the gradients
  norm_grads = tf.divide(grads, tf.sqrt(tf.reduce_mean(tf.square(grads))) + tf.constant(1e-5))

  output, grads_val = sess.run([conv_layer, norm_grads], feed_dict={imgs0: img})
  output = output[0]           # [10,10,2048]
  grads_val = grads_val[0]	 # [10,10,2048]

  weights = np.mean(grads_val, axis = (0, 1)) 			# [2048]
  cam = np.ones(output.shape[0 : 2], dtype = np.float32)	# [10,10]

  # Taking a weighted average
  for i, w in enumerate(weights):
    cam += w * output[:, :, i]

  # Passing through ReLU
  cam = np.maximum(cam, 0)
  cam = cam / np.max(cam)
  cam3 = cv2.resize(cam, (int(FLAGS.image_size/2), FLAGS.image_size))

  return cam3


def main(_):
  checkpoint_path=FLAGS.checkpoint_path
  img = load_image(FLAGS.input)

  num_classes = FLAGS.num_classes


  model_name = FLAGS.model_name + '_uPMnet'

  network_fn = nets_factory.get_network_fn(
    model_name,
    num_classes=num_classes,
    is_training=False)

  print("\nLoading Model")
  imgs0 = tf.placeholder(tf.uint8, [None,None, 3])
  imgs = preprocess_image(imgs0)
  imgs = tf.expand_dims(imgs,0)

  with slim.arg_scope(nets_factory.arg_scopes_map[model_name]()):
    logits, end_points = network_fn(imgs)

  var_averages = tf.train.ExponentialMovingAverage(FLAGS.ema_decay)
  var_to_restore = var_averages.variables_to_restore()
  # only restores the feature extractors
  if FLAGS.relation == 'global':
    var_to_restore = {k: v for k, v in var_to_restore.iteritems() if not 'Logits/Conv2d' in k}
  else:
    var_to_restore = {k: v for k, v in var_to_restore.iteritems() if not 'Logits' in k}
  saver = tf.train.Saver(var_to_restore)

  ckptfile = [re.search(r'\d+.ckpt', i) for i in os.listdir(FLAGS.checkpoint_path)]
  assert len(ckptfile) > 0, "No checkpoint file found"
  global_step = set([i.group()[:-5] for i in ckptfile if i is not None])

  # init_fn = slim.assign_from_checkpoint_fn(checkpoint_path, slim.get_variables_to_restore())

  print("\nFeedforwarding")

  with tf.Session() as sess:
    # init_fn(sess)
    # Restores from checkpoint with absolute path.
    ckpt = os.path.join(os.getcwd(), FLAGS.checkpoint_path, 'model_' + FLAGS.steps + '.ckpt-' + FLAGS.steps)
    print('Restores from checkpoint path: ' + ckpt)
    saver.restore(sess, ckpt)
    print('Successfully loaded model from %s at step=%s.' % (ckpt, FLAGS.steps))

    # Target layer for visualization
    layer_name = FLAGS.layer_name

    cam3 = grad_cam(img, imgs0, end_points, sess, layer_name)

    img = cv2.resize(img, (int(FLAGS.image_size/2), FLAGS.image_size))
    img = img.astype(float)
    img /= img.max()


    cam3 = cv2.applyColorMap(np.uint8(255*cam3), cv2.COLORMAP_JET)
    cam3 = cv2.cvtColor(cam3, cv2.COLOR_BGR2RGB)

    # Superimposing the visualization with the image.
    alpha = 0.0025
    new_img = img+alpha*cam3
    new_img /= new_img.max()

    # Display and save
    io.imshow(new_img)
    plt.axis('off')
    FLAGS.output = os.path.splitext(os.path.split(FLAGS.input)[1])[0] + \
                   '_' + str(FLAGS.n_part) + \
                   '_' + str(FLAGS.full_part) + \
                   '_' + FLAGS.checkpoint_path.split('/')[-2] + \
                   '_' + str(FLAGS.relation) + \
                   '_' + str(FLAGS.steps) + \
                   '_layer' + str(layer_name[7:9])+ '.png'
    plt.savefig(FLAGS.output,bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
  tf.app.run()

