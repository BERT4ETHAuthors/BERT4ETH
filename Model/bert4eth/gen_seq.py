# coding=utf-8
# Copyright 2022 BERT4ETH Authors.
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

import numpy as np
import pandas as pd
import pickle as pkl
import functools
from vocab import FreqVocab
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_bool("phisher", True, "whether to include phisher detection dataset.")
flags.DEFINE_bool("deanon", True, "whether to include de-anonymization dataset.")
flags.DEFINE_bool("mev", True, "whether to include mev-bot dataset.")
flags.DEFINE_string("bizdate", None, "the date of running experiments.")

if FLAGS.bizdate is None:
    raise ValueError("bizdate is required..")

HEADER = 'hash,nonce,block_hash,block_number,transaction_index,from_address,to_address,value,gas,gas_price,input,block_timestamp,max_fee_per_gas,max_priority_fee_per_gas,transaction_type'.split(",")

def cmp_udf(x1, x2):
    time1 = int(x1[2])
    time2 = int(x2[2])
    if time1 < time2:
        return -1
    elif time1 > time2:
        return 1
    else:
        return 0

def cmp_udf_reverse(x1, x2):
    time1 = int(x1[2])
    time2 = int(x2[2])

    if time1 < time2:
        return 1
    elif time1 > time2:
        return -1
    else:
        return 0

def load_data(f_in, f_out):
    eoa2seq_out = {}
    error_trans = []
    while True:
        trans = f_out.readline()
        if trans == "":
            break
        record = trans.split(",")
        trans_hash = record[0]
        block_number = int(record[3])
        from_address = record[5]
        to_address = record[6]
        value = int(record[7]) / (pow(10, 12))
        gas = int(record[8])
        gas_price = int(record[9])
        block_timestamp = int(record[11])
        if from_address == "" or to_address == "":
            error_trans.append(trans)
            continue
        try:
            eoa2seq_out[from_address].append([to_address, block_number, block_timestamp, value, "OUT", 1])
        except:
            eoa2seq_out[from_address] = [[to_address, block_number, block_timestamp, value, "OUT", 1]]

    eoa2seq_in = {}
    while True:
        trans = f_in.readline()
        if trans == "":
            break
        record = trans.split(",")
        block_number = int(record[3])
        from_address = record[5]
        to_address = record[6]
        value = int(record[7]) / (pow(10, 12))
        gas = int(record[8])
        gas_price = int(record[9])
        block_timestamp = int(record[11])
        if from_address == "" or to_address == "":
            error_trans.append(trans)
            continue
        try:
            eoa2seq_in[to_address].append([from_address, block_number, block_timestamp, value, "IN", 1]) # not process trans
        except:
            eoa2seq_in[to_address] = [[from_address, block_number, block_timestamp, value, "IN", 1]] # in/out, cnt
    return eoa2seq_in, eoa2seq_out

def seq_duplicate(eoa2seq_in, eoa2seq_out):
    eoa2seq_agg_in = {}
    for eoa in eoa2seq_in.keys():
        if len(eoa2seq_in[eoa]) >= 10000:
            continue
        seq_sorted = sorted(eoa2seq_in[eoa], key=functools.cmp_to_key(cmp_udf))
        seq_tmp = [e.copy() for e in seq_sorted]
        for i in range(len(seq_tmp) - 1, 0, -1):
            l_acc = seq_tmp[i][0]  # latter
            f_acc = seq_tmp[i - 1][0]  # former
            l_time = int(seq_tmp[i][2])
            f_time = int(seq_tmp[i - 1][2])
            delta_time = l_time - f_time
            if f_acc != l_acc or delta_time > 86400 * 3:
                continue
            # value add
            seq_tmp[i - 1][3] += seq_tmp[i][3]
            seq_tmp[i - 1][5] += seq_tmp[i][5]
            del seq_tmp[i]
        eoa2seq_agg_in[eoa] = seq_tmp

    eoa2seq_agg_out = {}
    for eoa in eoa2seq_out.keys():
        if len(eoa2seq_out[eoa])>=10000:
            continue
        seq_sorted = sorted(eoa2seq_out[eoa], key=functools.cmp_to_key(cmp_udf))
        seq_tmp = [e.copy() for e in seq_sorted]
        for i in range(len(seq_tmp) - 1, 0, -1):
            l_acc = seq_tmp[i][0]  # latter
            f_acc = seq_tmp[i - 1][0]  # former
            l_time = int(seq_tmp[i][2])
            f_time = int(seq_tmp[i - 1][2])
            delta_time = l_time - f_time
            if f_acc != l_acc or delta_time > 86400 * 3:
                continue
            # value add
            seq_tmp[i - 1][3] += seq_tmp[i][3]
            seq_tmp[i - 1][5] += seq_tmp[i][5]
            del seq_tmp[i]
        eoa2seq_agg_out[eoa] = seq_tmp

    eoa_list = list(eoa2seq_agg_out.keys()) # eoa_list must include eoa account only (i.e., have out transaction at least)
    eoa2seq_agg = {}

    for eoa in eoa_list:
        out_seq = eoa2seq_agg_out[eoa]
        try:
            in_seq = eoa2seq_agg_in[eoa]
        except:
            in_seq = []

        seq_agg = sorted(out_seq + in_seq, key=functools.cmp_to_key(cmp_udf_reverse))
        cnt_all = 0
        for trans in seq_agg:
            cnt_all += trans[5]
            if cnt_all > 2:
                eoa2seq_agg[eoa] = seq_agg
                break

    return eoa2seq_agg

def feature_bucketization(eoa2seq_agg):

    for eoa in eoa2seq_agg.keys():
        seq = eoa2seq_agg[eoa]
        for trans in seq:
            amount = trans[3]
            cnt = trans[5]

            if amount == 0:
                amount_bucket = 1
            elif amount<= 591:
                amount_bucket = 2
            elif amount<= 6195:
                amount_bucket = 3
            elif amount <= 21255:
                amount_bucket = 4
            elif amount <= 50161:
                amount_bucket = 5
            elif amount <= 100120:
                amount_bucket = 6
            elif amount <= 208727:
                amount_bucket = 7
            elif amount <= 508961:
                amount_bucket = 8
            elif amount <= 1360574:
                amount_bucket = 9
            elif amount <= 6500000:
                amount_bucket = 10
            elif amount <= 143791433950:
                amount_bucket = 11
            else:
                amount_bucket = 12

            trans[3] = amount_bucket

            if cnt == 0:
                cnt_bucket = 0
            elif cnt == 1:
                cnt_bucket = 1
            elif cnt == 2:
                cnt_bucket = 2
            elif cnt == 3:
                cnt_bucket = 3
            elif cnt == 4:
                cnt_bucket = 4
            elif cnt == 5:
                cnt_bucket = 5
            elif cnt == 6:
                cnt_bucket = 6
            elif cnt == 7:
                cnt_bucket = 7
            elif 8 < cnt <= 10:
                cnt_bucket = 8
            elif 10 < cnt <= 20:
                cnt_bucket = 9
            else:
                cnt_bucket = 10

            trans[5] = cnt_bucket

    return eoa2seq_agg

def main():

    # f_in = open("../../Data/normal_eoa_transaction_in_slice_100K.csv", "r")
    # f_out = open("../../Data/normal_eoa_transaction_out_slice_100K.csv", "r")

    f_in = open("../../Data/normal_eoa_transaction_in_slice_1000K.csv", "r")
    f_out = open("../../Data/normal_eoa_transaction_out_slice_1000K.csv", "r")
    eoa2seq_in, eoa2seq_out = load_data(f_in, f_out)
    eoa2seq_agg = seq_duplicate(eoa2seq_in, eoa2seq_out)

    if FLAGS.phisher:
        phisher_f_in = open("../../Data/phisher_transaction_in.csv", "r")
        phisher_f_out = open("../../Data/phisher_transaction_out.csv", "r")
        phisher_eoa2seq_in, phisher_eoa2seq_out = load_data(phisher_f_in, phisher_f_out)
        phisher_eoa2seq_agg = seq_duplicate(phisher_eoa2seq_in, phisher_eoa2seq_out)
        eoa2seq_agg.update(phisher_eoa2seq_agg)

    if FLAGS.deanon:
        dean_f_in = open("../../Data/dean_trans_in_new.csv", "r")
        dean_f_out = open("../../Data/dean_trans_out_new.csv", "r")
        dean_eoa2seq_in, dean_eoa2seq_out = load_data(dean_f_in, dean_f_out)
        dean_eoa2seq_agg = seq_duplicate(dean_eoa2seq_in, dean_eoa2seq_out)
        eoa2seq_agg.update(dean_eoa2seq_agg)

    if FLAGS.mev:
        mev_f_in = open("../../Data/filtered_mev_bot_transaction_in.csv", "r")
        mev_f_out = open("../../Data/filtered_mev_bot_transaction_out.csv", "r")
        mev_eoa2seq_in, mev_eoa2seq_out = load_data(mev_f_in, mev_f_out)
        mev_eoa2seq_agg = seq_duplicate(mev_eoa2seq_in, mev_eoa2seq_out)
        eoa2seq_agg.update(mev_eoa2seq_agg)

    eoa2seq_agg = feature_bucketization(eoa2seq_agg)

    with open("./data/eoa2seq_" + FLAGS.bizdate + ".pkl", "wb") as f:
        pkl.dump(eoa2seq_agg, f)

print("pause")

if __name__ == '__main__':
    main()