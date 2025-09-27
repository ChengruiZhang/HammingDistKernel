#!/usr/bin/python3
# coding=utf-8

# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

import os
import numpy as np

K = 128
OUTTER = 1
INNER = 2048
N = 2048


def gen_golden_data_simple():
    x1 = np.random.uniform(-60000, 60000, [OUTTER, INNER]).astype(np.float32)
    x2 = [1000 for i in range(INNER)]
    x2 = np.array(x2).astype(np.int32)

    finishlocal = np.random.randint(0, 2, OUTTER, dtype=bool)

    os.system("mkdir -p ./input")
    x1.tofile("./input/input_srcGmValue.bin")
    x2.tofile("./input/input_srcGmIndex.bin")
    finishlocal.tofile("./input/input_finishGm.bin")

    indices = np.argsort(x1, axis=-1, kind="stable")
    golden1 = (indices[:, :K]).astype(np.int32)
    golden0 = np.take_along_axis(x1, golden1, axis=-1)

    k_pad = (K + 7) // 8 * 8
    kpad_i = (K + 7) // 8 * 8
    diff1 = k_pad - K
    diff2 = kpad_i - K
    if diff1 != 0:
        golendiff1 = np.zeros((OUTTER, diff1)).astype(np.float32)
        golden0 = np.concatenate((golden0, golendiff1), axis=1).astype(np.float32)
    if diff2 != 0:
        golendiff2 = np.zeros((OUTTER, diff2)).astype(np.int32)
        golden1 = np.concatenate((golden1, golendiff2), axis=1).astype(np.int32)
    
    for i in range(OUTTER):
        if (finishlocal[i] == True):
            golden1[i, :K] = N

    os.system("mkdir -p ./output")
    golden0.tofile("./output/golden_dstGmValue.bin")
    golden1.tofile("./output/golden_dstGmIndex.bin")

if __name__ == "__main__":
    gen_golden_data_simple()
