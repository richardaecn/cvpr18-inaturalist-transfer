# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""A factory-pattern class which returns classification image/label pairs."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from datasets import fgvc

datasets_map = {
  'ILSVRC2012': {'num_samples': {'train': 1281167, 'validation': 50000},
                 'num_classes': 1000},
  'inat2017': {'num_samples': {'train': 665473, 'validation': 9697},
               'num_classes': 5089},
  'aircraft': {'num_samples': {'train': 6667, 'validation': 3333},
               'num_classes': 100},
  'cub_200': {'num_samples': {'train': 5994, 'validation': 5794},
              'num_classes': 200},
  'flower_102': {'num_samples': {'train': 2040, 'validation': 6149},
                 'num_classes': 102},
  'food_101': {'num_samples': {'train': 75750, 'validation': 25250},
               'num_classes': 101},
  'nabirds': {'num_samples': {'train': 23929, 'validation': 24633},
              'num_classes': 555},
  'stanford_cars': {'num_samples': {'train': 8144, 'validation': 8041},
                    'num_classes': 196},
  'stanford_dogs': {'num_samples': {'train': 12000, 'validation': 8580},
                    'num_classes': 120}
}


def get_dataset(name, split_name, root_dir, file_pattern=None, reader=None):
  """Given a dataset name and a split_name returns a Dataset.

  Args:
    name: String, the name of the dataset.
    split_name: A train/validation split name.
    root_dir: The root directory of all datasets.
    file_pattern: The file pattern to use for matching the dataset source files.
    reader: The subclass of tf.ReaderBase. If left as `None`, then the default
      reader defined by each dataset is used.

  Returns:
    A `Dataset` class.

  Raises:
    ValueError: If the dataset `name` is unknown.
  """
  if name not in datasets_map:
    raise ValueError('Name of dataset unknown %s' % name)
  return fgvc.get_split(
      split_name,
      os.path.join(root_dir, name),
      datasets_map[name]['num_samples'],
      datasets_map[name]['num_classes'],
      file_pattern,
      reader)
