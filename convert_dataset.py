"""Converts data to TFRecords of TF-Example protos.
python convert_dataset.py --dataset_name=cub_200 --num_shards=10
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys

import tensorflow as tf

sys.path.insert(0, './slim/')
from datasets import dataset_utils

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
    'dataset_name',
    None,
    'The name of the dataset to convert, one of "ILSVRC2012", "inat2017", '
    '"aircraft", "cub_200", "flower_102", "food_101", "nabirds", '
    '"stanford_cars", "stanford_dogs"')

tf.app.flags.DEFINE_integer(
    'num_shards', 10, 'The number of shards per dataset split.')


class ImageReader(object):
  """Helper class that provides TensorFlow image coding utilities."""

  def __init__(self):
    # Initializes function that decodes RGB JPEG data.
    self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
    self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

  def read_image_dims(self, sess, image_data):
    image = self.decode_jpeg(sess, image_data)
    return image.shape[0], image.shape[1]

  def decode_jpeg(self, sess, image_data):
    image = sess.run(self._decode_jpeg,
                     feed_dict={self._decode_jpeg_data: image_data})
    assert len(image.shape) == 3
    assert image.shape[2] == 3
    return image


def _get_filenames_and_labels(dataset_dir):
  train_filenames = []
  val_filenames = []
  train_labels = []
  val_labels = []
  for line in open(os.path.join(dataset_dir, 'train.txt'), 'r'):
    line_list = line.strip().split(': ')
    train_filenames.append(os.path.join(dataset_dir, line_list[0]))
    train_labels.append(int(line_list[1]))
  for line in open(os.path.join(dataset_dir, 'val.txt'), 'r'):
    line_list = line.strip().split(': ')
    val_filenames.append(os.path.join(dataset_dir, line_list[0]))
    val_labels.append(int(line_list[1]))
  return train_filenames, val_filenames, train_labels, val_labels


def _get_dataset_filename(dataset_dir, split_name, shard_id):
  output_filename = '%s_%05d-of-%05d.tfrecord' % (
      split_name, shard_id, FLAGS.num_shards)
  return os.path.join(dataset_dir, output_filename)


def _convert_dataset(split_name, filenames, labels, dataset_dir):
  """Converts the given filenames to a TFRecord dataset.

  Args:
    split_name: The name of the dataset, either 'train' or 'validation'.
    filenames: A list of absolute paths to png or jpg images.
    labels: A list of class ids (integers start with 0).
    dataset_dir: The directory where the converted datasets are stored.
  """
  assert split_name in ['train', 'validation']

  num_per_shard = int(math.ceil(len(filenames) / float(FLAGS.num_shards)))

  with tf.Graph().as_default():
    image_reader = ImageReader()

    with tf.Session('') as sess:

      for shard_id in range(FLAGS.num_shards):
        output_filename = _get_dataset_filename(
            dataset_dir, split_name, shard_id)

        with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
          start_ndx = shard_id * num_per_shard
          end_ndx = min((shard_id+1) * num_per_shard, len(filenames))
          for i in range(start_ndx, end_ndx):
            sys.stdout.write('\r>> Converting %s image %d/%d shard %d' % (
                split_name, i+1, len(filenames), shard_id))
            sys.stdout.flush()

            # Read the filename and label:
            image_data = tf.gfile.FastGFile(filenames[i], 'rb').read()
            height, width = image_reader.read_image_dims(sess, image_data)
            class_id = labels[i]

            example = dataset_utils.image_to_tfexample(
                image_data, b'jpg', height, width, class_id)
            tfrecord_writer.write(example.SerializeToString())

  sys.stdout.write('\n')
  sys.stdout.flush()


def _dataset_exists(dataset_dir):
  for split_name in ['train', 'validation']:
    for shard_id in range(FLAGS.num_shards):
      output_filename = _get_dataset_filename(
          dataset_dir, split_name, shard_id)
      if not tf.gfile.Exists(output_filename):
        return False
  return True


def run(dataset_dir):
  """Runs the conversion operation.

  Args:
    dataset_dir: The dataset directory where the dataset is stored.
  """
  if not tf.gfile.Exists(dataset_dir):
    tf.gfile.MakeDirs(dataset_dir)

  if _dataset_exists(dataset_dir):
    print('Dataset files already exist. Exiting without re-creating them.')
    return

  train_filenames, val_filenames, train_labels, val_labels = \
      _get_filenames_and_labels(dataset_dir)

  train_idx = list(zip(train_filenames, train_labels))
  random.shuffle(train_idx)
  train_filenames, train_labels = zip(*train_idx)

  _convert_dataset('train', train_filenames, train_labels, dataset_dir)
  _convert_dataset('validation', val_filenames, val_labels, dataset_dir)

  print('\nFinished converting the dataset!')


def main(_):
  if not FLAGS.dataset_name:
    raise ValueError('You must supply the dataset name with --dataset_name')

  run(os.path.join('./data', FLAGS.dataset_name))


if __name__ == '__main__':
  tf.app.run()
