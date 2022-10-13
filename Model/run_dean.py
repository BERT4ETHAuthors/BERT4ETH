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

import pandas as pd
import numpy as np
import pickle as pkl
from tqdm import tqdm
from numpy import dot
from numpy.linalg import norm
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

flags = tf.flags
FLAGS = flags.FLAGS
flags.DEFINE_string("metric", "euclidean", "")
flags.DEFINE_string("ens_dataset", "../Data/dean_all_ens_pairs.csv", "")
flags.DEFINE_integer("max_cnt", 2, "")
flags.DEFINE_string("algo", "bert4eth", "algorithm for embedding generation" )

def euclidean_dist(a, b):
    return np.sqrt(np.sum(np.square(a-b)))

def cosine_dist(a, b):
    return 1-dot(a, b)/(norm(a)*norm(b)) # notice, here need 1-

def cosine_dist_multi(a, b):
    num = dot(a, b.T)
    denom = norm(a) * norm(b, axis=1)
    res = num/denom
    return -1 * res

def euclidean_dist_multi(a, b):
    return np.sqrt(np.sum(np.square(b-a), axis=1))

def get_neighbors(X, idx, metric="cosine" ,include_idx_mask=[]):
    a = X[idx, :]
    indices = list(range(X.shape[0]))
    if metric == "cosine":
        # dist = np.array([cosine_dist(a, X[i, :]) for i in indices])
        dist = cosine_dist_multi(a, X)
    elif metric == "euclidean":
        dist = euclidean_dist_multi(a, X)
    else:
        raise ValueError("Distance Metric Error")
    sorted_df = pd.DataFrame(list(zip(indices, dist)), columns=["idx", "dist"]).sort_values("dist")
    indices = list(sorted_df["idx"])
    indices.remove(idx)  # exclude self distance

    if len(include_idx_mask) > 0:
        # filter indices
        indices_tmp = []
        for i, res_idx in enumerate(indices):
            if res_idx in include_idx_mask:
                indices_tmp.append(res_idx)
        indices = indices_tmp
    return indices

def get_rank(X, query_idx, target_idx, include_idx_mask=[]):
    indices = get_neighbors(X, query_idx, FLAGS.metric, include_idx_mask)
    if len(indices) > 0 and target_idx in indices:
        trg_idx = indices.index(target_idx)
        return trg_idx+1, len(indices)
    else:
        return None, len(indices)


def generate_pairs(ens_pairs, min_cnt=2, max_cnt=2, mirror=True):
    """
    Generate testing pairs based on ENS name
    :param ens_pairs:
    :param min_cnt:
    :param max_cnt:
    :param mirror:
    :return:
    """
    pairs = ens_pairs.copy()
    ens_counts = pairs["name"].value_counts()
    address_pairs = []
    all_ens_names = []
    ename2addresses = {}
    for idx, row in pairs.iterrows():
        try:
            ename2addresses[row["name"]].append(row["address"]) # note: cannot use row.name
        except:
            ename2addresses[row["name"]] = [row["address"]]
    for cnt in range(min_cnt, max_cnt + 1):
        ens_names = list(ens_counts[ens_counts == cnt].index)
        all_ens_names += ens_names
        # convert to indices
        for ename in ens_names:
            addrs = ename2addresses[ename]
            for i in range(len(addrs)):
                for j in range(i + 1, len(addrs)):
                    addr1, addr2 = addrs[i], addrs[j]
                    address_pairs.append([addr1, addr2])
                    if mirror:
                        address_pairs.append([addr2, addr1])
    return address_pairs, all_ens_names


def load_embedding():

    exp_addr_set = set(np.load("bert4eth/data/exp_addr.npy"))

    if FLAGS.algo == "bert4eth":
        embeddings = np.load("bert4eth/data/xxx.npy")
        address_for_embedding = np.load("bert4eth/data/xxx.npy")
    else:
        raise ValueError("should choose right algo..")

    # group by embedding according to address
    address_to_embedding = {}
    for i in range(len(address_for_embedding)):
        address = address_for_embedding[i]
        embedding = embeddings[i]
        # if address not in exp_addr_set:
        #     continue
        try:
            address_to_embedding[address].append(embedding)
        except:
            address_to_embedding[address] = [embedding]

    # group to one
    address_list = []
    embedding_list = []

    for addr, embeds in address_to_embedding.items():
        address_list.append(addr)
        if len(embeds) > 1:
            embedding_list.append(np.mean(embeds, axis=0))
        else:
            embedding_list.append(embeds[0])

    # final embedding table
    X = np.array(np.squeeze(embedding_list))
    return X, address_list

def main():

    # load dataset
    ens_pairs = pd.read_csv(FLAGS.ens_dataset)
    max_ens_per_address = 1
    num_ens_for_addr = ens_pairs.groupby("address")["name"].nunique().sort_values(ascending=False).reset_index()
    excluded = list(num_ens_for_addr[num_ens_for_addr["name"] > max_ens_per_address]["address"])
    ens_pairs = ens_pairs[~ens_pairs["address"].isin(excluded)]
    address_pairs, all_ens_names = generate_pairs(ens_pairs, max_cnt=FLAGS.max_cnt)

    X, address_list = load_embedding()
    # map address to int
    cnt = 0
    address_to_idx = {}
    idx_to_address = {}
    for address in address_list:
        address_to_idx[address] = cnt
        idx_to_address[cnt] = address
        cnt += 1

    idx_pairs = []
    failed_address = []
    for pair in address_pairs:
        try:
            idx_pairs.append([address_to_idx[pair[0]], address_to_idx[pair[1]]])
        except:
            failed_address.append(pair[0])
            failed_address.append(pair[1])
            continue

    pbar = tqdm(total=len(idx_pairs))
    records = []
    for pair in idx_pairs:
        rank, num_set = get_rank(X, pair[1], pair[0])
        records.append((pair[1], pair[0], rank, num_set, "none"))
        pbar.update(1)

    result = pd.DataFrame(records, columns=["query_idx", "target_idx", "rank", "set_size", "filter"])
    result["query_addr"] = result["query_idx"].apply(lambda x: idx_to_address[x])
    result["target_addr"] = result["target_idx"].apply(lambda x: idx_to_address[x])
    result.drop(["query_idx", "target_idx"], axis=1)


    output_file = "bert4eth/data/ens/xxx.csv"
    result.to_csv(output_file, index=False)


if __name__ == '__main__':
    main()


