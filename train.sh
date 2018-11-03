#!/bin/sh
# CUDA_VISIBLE_DEVICES=0 ./train.sh

DATASET_NAME=cub_200
DATASET_DIR=./data/
MODEL_DIR=./checkpoints/${DATASET_NAME}
CHECKPOINT_PATH=./checkpoints/inception/inception_v3_iNat_299.ckpt
CHECKPOINT_EXCLUDE_SCOPES=Logits
# fine-tune last layer
TRAINABLE_SCOPES=Logits

MODEL_NAME=inception_v3

OPTIMIZER=momentum

LR=0.01
END_LR=0.0001
LR_DECAY=0.1
EPOCHS_PER_DECAY=10.0

TRAIN_BATCH_SIZE=64
TRAIN_IMAGE_SIZE=299

LOG_STEPS=10
SAVE_SUMMARIES_SECS=10
SAVE_MODEL_SECS=60

# fine-tune last layer for 30 epochs.
# cub_200 has 5994 training images, with 64 train batch size:
# 1 epoch = 94 steps, 30 epochs = 2810 steps.
MAX_STEPS=2810

python train.py \
    --train_dir=${MODEL_DIR} \
    --dataset_dir=${DATASET_DIR} \
    --dataset_name=${DATASET_NAME} \
    --model_name=${MODEL_NAME} \
    --checkpoint_path=${CHECKPOINT_PATH} \
    --checkpoint_exclude_scopes=${CHECKPOINT_EXCLUDE_SCOPES} \
    --trainable_scopes=${TRAINABLE_SCOPES} \
    --optimizer=${OPTIMIZER} \
    --learning_rate=${LR} \
    --end_learning_rate=${END_LR} \
    --learning_rate_decay_factor=${LR_DECAY} \
    --num_epochs_per_decay=${EPOCHS_PER_DECAY} \
    --batch_size=${TRAIN_BATCH_SIZE} \
    --train_image_size=${TRAIN_IMAGE_SIZE} \
    --max_number_of_steps=${MAX_STEPS} \
    --log_every_n_steps=${LOG_STEPS} \
    --save_summaries_secs=${SAVE_SUMMARIES_SECS} \
    --save_interval_secs=${SAVE_MODEL_SECS}
