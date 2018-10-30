"""Implementation of Squeeze-and-Excitation block.

Original paper: https://arxiv.org/abs/1709.01507.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

slim = tf.contrib.slim


def se_block(net, r=16):
  """Squeeze-and-Excitation block.

  Args:
    net: network activations from the previous layer.
    r: reduction ratio.

  Returns:
    network activations after scaling by SE block.
  """
  dims = net.get_shape().as_list()

  # input shape: [B, H, W, C]
  # output shape: [B, 1, 1, C]
  se = slim.avg_pool2d(net, dims[1:3], padding='VALID')

  # output shape: [B, 1, 1, C/r]
  se = slim.conv2d(se,
                   int(dims[3] / r),
                   [1, 1],
                   activation_fn=tf.nn.relu,
                   normalizer_fn=None)

  # output shape: [B, 1, 1, C]
  se = slim.conv2d(se,
                   dims[3],
                   [1, 1],
                   activation_fn=tf.nn.sigmoid,
                   normalizer_fn=None)

  # output shape: [B, H, W, C]
  return tf.multiply(net, se)
