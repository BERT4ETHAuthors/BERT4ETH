# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
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
"""Run masked LM/next sentence masked_lm pre-training for BERT."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import modeling
import optimization
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import numpy as np
import sys
import pickle as pkl
import time
from run_pretrain import *

def del_flags(FLAGS, keys_list):
    for keys in keys_list:
        FLAGS.__delattr__(keys)

    return

if __name__ == '__main__':
    del_flags(FLAGS, ["do_train", "do_eval", "init_checkpoint"])
    flags.DEFINE_bool("do_train", False, "")
    flags.DEFINE_bool("do_eval", True, "")
    flags.DEFINE_string("init_checkpoint", "ckpt_dir/model_100", "Initial checkpoint (usually from a pre-trained BERT model).")

    tf.app.run()