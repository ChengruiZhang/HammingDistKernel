#!/usr/bin/python3
# coding=utf-8
#
# Copyright (C) 2023-2025. Huawei Technologies Co., Ltd. All rights reserved.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# ===============================================================================

import numpy as np
import tensorflow as tf
import sys

# bfloat16 = tf.bfloat16.as_numpy_dtype
# dtype_emu = {bfloat16: 0, np.float16: 1, np.float32: 2, np.int8: 3, np.int16: 4, np.int32: 5}

def gen_golden_data_simple():
    # dtype = np.float32
    dtype = np.uint16
    # dtype = np.int8


    input_shape_x = [17, 1022]
    input_shape_y = [1, 1022]

    input_x = np.random.uniform(-50, 50, input_shape_x).astype(dtype)
    input_y = np.random.uniform(-50, 50, input_shape_y).astype(dtype)
    # golden = (input_x + input_y).astype(dtype)

    # if np.size(input_x) > np.size(input_y):
    #     if input_shape_y[0] == 1:
    #         axis = 0
    #         coef = np.size(input_y)
    #     elif input_shape_y[1] == 1:
    #         axis = 1
    #         coef = np.size(input_x) / np.size(input_y)
    # else:
    #     if input_shape_x[0] == 1:
    #         axis = 0
    #         coef = np.size(input_x)
    #     elif input_shape_x[1] == 1:
    #         axis = 1
    #         coef = np.size(input_y) / np.size(input_x)

    # tiling.tofile("./input/input_tiling.bin")
    input_x.tofile("./input/input_qhash.bin")
    input_y.tofile("./input/input_y.bin")
    # golden.tofile("./output/golden.bin")

if __name__ == "__main__":

    batchSize = int(sys.argv[1])
    seqLen = int(sys.argv[2])
    HeadQ = int(sys.argv[3])
    HeadK = int(sys.argv[4])
    hidDim = int(sys.argv[5])
    topK = int(sys.argv[6])

    input_x = np.random.randint(0, 10, (batchSize * 1 * HeadQ * hidDim // 16)).astype(np.uint16)
    input_y = np.random.randint(0, 10, (batchSize * seqLen * HeadK * hidDim // 16)).astype(np.uint16)

    input_x.tofile("./input/input_qhash.bin")
    input_y.tofile("./input/input_khash.bin")
    # gen_golden_data_simple(batchSize, seqLen, HeadQ, HeadK, hidDim, topK)
