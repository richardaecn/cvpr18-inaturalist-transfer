from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import sys
import time
import tensorflow as tf

slim = tf.contrib.slim
sys.path.insert(0, './slim/')
from nets import inception, resnet_v2
from preprocessing import inception_preprocessing

data_dir = './data'
# dataset needs to be one of ['ILSVRC2012', 'inat2017', 'cub_200', 'flower_102',
# 'stanford_cars', 'stanford_dogs', 'aircraft', 'nabirds', 'food_101']
dataset = 'cub_200'
# base_network needs to be one of ['InceptionV3', 'InceptionV3SE',
# 'InceptionV4', 'InceptionResnetV2', 'InceptionResnetV2SE', 'ResNet50',
# 'ResNet101', 'ResNet152']
base_network = 'InceptionV3'
checkpoints_path = './checkpoints/inception/inception_v3_iNat_299.ckpt'
# base_network = 'ResNet101'
# checkpoints_path = './checkpoints/resnet/resnet_101_ILSVRC_iNat_299.ckpt'

image_size = 299
moving_average_decay = 0.9999
fea_dim = 2048

# Read train and val list.
train_list = []
val_list = []
for line in open(os.path.join(data_dir, dataset, 'train.txt'), 'r'):
    train_list.append(
        (os.path.join(data_dir, dataset, line.strip().split(': ')[0]),
        int(line.strip().split(': ')[1])))
for line in open(os.path.join(data_dir, dataset, 'val.txt'), 'r'):
    val_list.append(
        (os.path.join(data_dir, dataset, line.strip().split(': ')[0]),
        int(line.strip().split(': ')[1])))
print('Length of train: %d' %len(train_list))
print('Length of val: %d' %len(val_list))

# Base network architecture
if base_network == 'InceptionV3':
    endpoint = 'Mixed_7c'
    arg_scope = inception.inception_v3_arg_scope()
elif base_network == 'InceptionV3SE':
    endpoint = 'Mixed_7c'
    arg_scope = inception.inception_v3_se_arg_scope()
elif base_network == 'InceptionV4':
    endpoint = 'Mixed_7d'
    arg_scope = inception.inception_v4_arg_scope()
elif base_network == 'InceptionResnetV2':
    endpoint = 'Conv2d_7b_1x1'
    arg_scope = inception.inception_resnet_v2_arg_scope()
elif base_network == 'InceptionResnetV2SE':
    endpoint = 'Conv2d_7b_1x1'
    arg_scope = inception.inception_resnet_v2_se_arg_scope()
elif base_network[:6] == 'ResNet':
    layers = base_network.split('ResNet')[1]
    base_network = 'ResNet'

# Feature extraction.
fea_train = np.zeros((len(train_list), fea_dim), dtype=np.float32)
label_train = np.zeros((len(train_list), ), dtype=np.int32)
fea_val = np.zeros((len(val_list), fea_dim), dtype=np.float32)
label_val = np.zeros((len(val_list), ), dtype=np.int32)

with tf.Graph().as_default():
    tf_global_step = tf.train.get_or_create_global_step()
    image_path = tf.placeholder(tf.string)
    image = tf.image.decode_jpeg(tf.read_file(image_path), channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = inception_preprocessing.preprocess_image(image,
                                                     image_size,
                                                     image_size,
                                                     is_training=False)
    images  = tf.expand_dims(image, 0)

    if base_network == 'ResNet':
        with slim.arg_scope(resnet_v2.resnet_arg_scope(use_batch_norm=True)):
            if layers == '50':
                net, _ = resnet_v2.resnet_v2_50(images, is_training=False)
            elif layers == '101':
                net, _ = resnet_v2.resnet_v2_101(images, is_training=False)
            elif layers == '152':
                net, _ = resnet_v2.resnet_v2_152(images, is_training=False)
    else:
        with slim.arg_scope(arg_scope):
            slim_args = [slim.batch_norm, slim.dropout]
            with slim.arg_scope(slim_args, is_training=False):
                with tf.variable_scope(base_network, reuse=None) as scope:
                    if base_network == 'InceptionV3':
                        net, _ = inception.inception_v3_base(
                            images, final_endpoint=endpoint, scope=scope)
                    elif base_network == 'InceptionV3SE':
                        net, _ = inception.inception_v3_se_base(
                            images, final_endpoint=endpoint, scope=scope)
                    elif base_network == 'InceptionV4':
                        net, _ = inception.inception_v4_base(
                            images, final_endpoint=endpoint, scope=scope)
                    elif base_network == 'InceptionResnetV2':
                        net, _ = inception.inception_resnet_v2_base(
                            images, final_endpoint=endpoint, scope=scope)
                    elif base_network == 'InceptionResnetV2SE':
                        net, _ = inception.inception_resnet_v2_se_base(
                            images, final_endpoint=endpoint, scope=scope)
    net = tf.reduce_mean(net, [0,1,2])

    variable_averages = tf.train.ExponentialMovingAverage(
        moving_average_decay, tf_global_step)
    variables_to_restore = variable_averages.variables_to_restore()
    init_fn = slim.assign_from_checkpoint_fn(
        checkpoints_path, variables_to_restore)

    config_sess = tf.ConfigProto(allow_soft_placement=True)
    config_sess.gpu_options.allow_growth = True
    with tf.Session(config=config_sess) as sess:
        init_fn(sess)
        start = time.time()
        print('Feature extraction on training set...')
        for i in range(len(fea_train)):
            if i%1000 == 0:
                print('%d/%d %.2fs'%(i, len(fea_train), time.time() - start))
            fea = sess.run(net, feed_dict={image_path:train_list[i][0]})
            fea_train[i, :] = fea
            label_train[i] = train_list[i][1]
        print('Feature extraction on validation set...')
        for i in range(len(fea_val)):
            if i%1000 == 0:
                print('%d/%d %.2fs'%(i, len(fea_val), time.time() - start))
            fea = sess.run(net, feed_dict={image_path:val_list[i][0]})
            fea_val[i, :] = fea
            label_val[i] = val_list[i][1]

model_name = checkpoints_path.split('/')[-1].split('.ckpt')[0]
if not os.path.exists(os.path.join('./feature', model_name)):
    os.makedirs(os.path.join('./feature', model_name))

save_dir = os.path.join('./feature', model_name, dataset)
np.save(os.path.join(save_dir + '_feature_train.npy'), fea_train)
np.save(os.path.join(save_dir + '_label_train.npy'), label_train)
np.save(os.path.join(save_dir + '_feature_val.npy'), fea_val)
np.save(os.path.join(save_dir + '_label_val.npy'), label_val)
