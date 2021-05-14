#!/bin/bash

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

tbsm_py="python tbsm_pytorch.py "

TIMESTAMP=(date +%s)
DLRM_PATH=../dlrm # location of https://github.com/facebookresearch/dlrm/
SEQ_LEN=10
RAW_TRAIN_FILE=/home/ubuntu/datasets/taobao/taobao_train.txt
RAW_TEST_FILE=/home/ubuntu/datasets/taobao/taobao_test.txt
PROCESSED_TRAIN_FILE=./output/taobao_train_t$SEQ_LEN.npz # location to save processed data to re-use betewen runs
PROCESSED_VAL_FILE=./output/taobao_val_t$SEQ_LEN.npz
PROCESSED_TEST_FILE=./output/taobao_test_t$SEQ_LEN.npz
OUTPUT_MODEL=./output/model_$SEQ_LEN_$TIMESTAMP


$tbsm_py  --use-gpu  --mode="train-test"  --dlrm-path=$DLRM_PATH --datatype="taobao" \
--model-type="tsl" --tsl-inner="def"  --tsl-num-heads=1 \
--save-model=OUTPUT_MODEL --num-train-pts=690000 --num-val-pts=300000 --points-per-user=10 --mini-batch-size=256 \
--nepochs=1 --numpy-rand-seed=123 --arch-embedding-size="987994-4162024-9439" --print-freq=100 --test-freq=2000 --num-batches=0  \
--raw-train-file=$RAW_TRAIN_FILE --raw-test-file=$RAW_TEST_FILE --pro-train-file=$PROCESSED_TRAIN_FILE \
--pro-val-file=$PROCESSED_VAL_FILE --pro-test-file=$PROCESSED_TEST_FILE --ts-length=$SEQ_LEN \
--device-num=0 --tsl-interaction-op="dot" --tsl-mechanism="mlp" --learning-rate=0.05  --arch-sparse-feature-size=16 \
--arch-mlp-bot="1-16" --arch-mlp-top="15-15" --tsl-mlp="15-15" --arch-mlp="60-1"
