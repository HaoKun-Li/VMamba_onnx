# --------------------------------------------------------
# Modified by $@#Anonymous#@$
# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import os
import time
import json
import random
import argparse
import datetime
import tqdm, pdb
import numpy as np
import onnx

from collections import Counter


def parse_option():
    parser = argparse.ArgumentParser('calculate op number of onnx', add_help=False)
    # easy config modification
    parser.add_argument('--onnx_path', type=str, default='model_lhk.onnx',
                        help='save as xxx.onnx')

    args, unparsed = parser.parse_known_args()

    return args



def main(args):

    # 加载ONNX模型
    onnx_model = onnx.load(args.onnx_path)

    graph = onnx_model.graph
    # 计数算子类型
    operator_counter = Counter(node.op_type for node in graph.node)
    op_sum = 0
    for op_type, count in operator_counter.items():
        op_sum+= count
        print(f"{op_type}: {count}")
    print(f"op_sum: {op_sum}")


if __name__ == '__main__':
    args = parse_option()

    main(args)
