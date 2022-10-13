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

import pickle as pkl
import functools
from vocab import FreqVocab
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_bool("phisher", False, "whether to include phisher detection dataset")
flags.DEFINE_bool("deanon", False, "whether to include de-anonymization dataset")
flags.DEFINE_string("bizdate", None, "the date of running experiments")

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

def load_data(f_in_list, f_out_list, erc20_trans, trans_to_log_eoa_list, hash_to_erc20_trans):
    eoa2seq_out = {}
    error_trans = []
    for f_out in f_out_list:
        cnt = 0
        while True:
            trans = f_out.readline()
            if trans == "":
                break
            record = trans.split(",")
            trans_hash = record[0]

            block_number = int(record[3])
            from_address = record[5]
            to_address = record[6]
            value = int(record[7])
            gas = int(record[8])
            gas_price = int(record[9])
            block_timestamp = int(record[11])
            if from_address == "" or to_address in ("", "0x000000000000000000000000000000000000dead", "0x0000000000000000000000000000000000000000"):
                error_trans.append(trans)
                continue

            if trans_hash in erc20_trans:
                cnt += 1
                for eoa in trans_to_log_eoa_list[trans_hash]:
                    try:
                        eoa2seq_out[from_address].append([eoa, block_number, block_timestamp, value, "OUT", 1, "ERC20"])# in/out, cnt
                    except:
                        eoa2seq_out[from_address] = [[eoa, block_number, block_timestamp, value, "OUT", 1, "ERC20"]]# in/out, cnt
                # continue
                # not delete
                try:
                    eoa2seq_out[from_address].append([to_address, block_number, block_timestamp, value, "OUT", 1, "TRANS"])# in/out, cnt
                except:
                    eoa2seq_out[from_address] = [[to_address, block_number, block_timestamp, value, "OUT", 1, "TRANS"]]# in/out, cnt

            else:
                try:
                    eoa2seq_out[from_address].append([to_address, block_number, block_timestamp, value, "OUT", 1, "TRANS"])# in/out, cnt
                except:
                    eoa2seq_out[from_address] = [[to_address, block_number, block_timestamp, value, "OUT", 1, "TRANS"]]# in/out, cnt

        print(f_out)
        print(cnt)

    eoa2seq_in = {}
    for f_in in f_in_list:
        while True:
            trans = f_in.readline()
            if trans == "":
                break
            record = trans.split(",")
            block_number = int(record[3])
            from_address = record[5]
            to_address = record[6]
            value = int(record[7])
            gas = int(record[8])
            gas_price = int(record[9])
            block_timestamp = int(record[11])
            if from_address == "" or to_address in ("", "0x000000000000000000000000000000000000dead", "0x0000000000000000000000000000000000000000"):
                error_trans.append(trans)
                continue
            try:
                eoa2seq_in[to_address].append([from_address, block_number, block_timestamp, value, "IN", 1, "TRANS"]) # not process trans
            except:
                eoa2seq_in[to_address] = [[from_address, block_number, block_timestamp, value, "IN", 1, "TRANS"]] # in/out, cnt

    # eoa_set = set(eoa2seq_out.keys())
    # for trans in erc20_trans:
    #     record = hash_to_erc20_trans[trans].split(",")
    #     block_number = int(record[3])
    #     from_address = record[5]
    #     to_address = record[6]
    #     value = int(record[7])
    #     gas = int(record[8])
    #     gas_price = int(record[9])
    #     block_timestamp = int(record[11])
    #
    #     for eoa in trans_to_log_eoa_list[trans]:
    #         if eoa not in eoa_set:
    #             continue
    #         try:
    #             eoa2seq_in[eoa].append([from_address, block_number, block_timestamp, value, "IN", 1, "ERC20"])
    #         except:
    #             eoa2seq_in[eoa] = [[from_address, block_number, block_timestamp, value, "IN", 1, "ERC20"]]

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

    eoa_list = list(eoa2seq_agg_out.keys())  # eoa_list must include eoa account only (i.e., have out transaction at least)

    eoa2seq_agg = {}
    for eoa in eoa_list:
        try:
            out_seq = eoa2seq_agg_out[eoa]
        except:
            out_seq = []
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

def main():

    with open("../../Data/exp_trans_to_erc20_eoa.pkl", "rb") as f:
        trans_to_log_eoa_list = pkl.load(f)
    erc20_trans = set()
    for trans in trans_to_log_eoa_list:
        length = len(trans_to_log_eoa_list[trans])
        if length <= 2:
            erc20_trans.add(trans)

    exp_erc20_trans = open("../../Data/exp_erc20_trans.csv", "r")
    exp_erc721_trans = open("../../Data/exp_erc721_trans.csv", "r")
    trans_files = [exp_erc20_trans]  # exp_erc721_trans
    hash_to_erc20_trans = {}
    for file in trans_files:
        while True:
            trans = file.readline()
            if trans == "":
                break
            record = trans.split(",")
            trans_hash = record[0]
            hash_to_erc20_trans[trans_hash] = trans

    f_in_list = []
    f_out_list = []

    f_in = open("../../Data/normal_eoa_transaction_in_slice_1000K.csv", "r")
    f_out = open("../../Data/normal_eoa_transaction_out_slice_1000K.csv", "r")
    f_in_list.append(f_in)
    f_out_list.append(f_out)

    if FLAGS.phisher:
        phisher_f_in = open("../../Data/phisher_transaction_in.csv", "r")
        phisher_f_out = open("../../Data/phisher_transaction_out.csv", "r")
        f_in_list.append(phisher_f_in)
        f_out_list.append(phisher_f_out)

    if FLAGS.deanon:
        dean_f_in = open("../../Data/dean_trans_in.csv", "r")
        dean_f_out = open("../../Data/dean_trans_out.csv", "r")
        f_in_list.append(dean_f_in)
        f_out_list.append(dean_f_out)

    eoa2seq_in, eoa2seq_out = load_data(f_in_list, f_out_list, erc20_trans, trans_to_log_eoa_list, hash_to_erc20_trans)
    eoa2seq_agg = seq_duplicate(eoa2seq_in, eoa2seq_out)

    with open("./data/eoa2seq_" + FLAGS.bizdate + ".pkl", "wb") as f:
        pkl.dump(eoa2seq_agg, f)

print("pause")

if __name__ == '__main__':
    main()