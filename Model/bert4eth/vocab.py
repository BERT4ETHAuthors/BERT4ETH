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

from collections import Counter

def convert_by_vocab(vocab, tokens):
    """Converts a sequence of [tokens|ids] using the vocab."""
    output = []
    for token in tokens:
        output.append(vocab[token])
    return output

class FreqVocab(object):
    """Runs end-to-end tokenziation."""

    def __init__(self):
        # layout of the  ulary
        # item_id based on freq
        # special token
        # user_id based on nothing
        self.counter = Counter()
        self.frequency = []

    def update(self, eoa2seq):
        for eoa in eoa2seq.keys():
            seq = eoa2seq[eoa]
            self.counter[eoa] = len(seq)
            self.counter.update(map(lambda x:x[0], seq))

    def generate_vocab(self):
        self.token_count = len(self.counter.keys())
        self.special_tokens = {"[pad]", "[MASK]", '[NO_USE]'}
        self.token_to_ids = {}  # index begin from 1
        #first items

        for token, count in self.counter.most_common():
            self.token_to_ids[token] = len(self.token_to_ids) + 1

        # then special tokens
        for token in self.special_tokens:
            self.token_to_ids[token] = len(self.token_to_ids) + 1
            self.counter[token] = 0

        self.id_to_tokens = {v: k for k, v in self.token_to_ids.items()}
        self.vocab_words = list(self.token_to_ids.keys())

        id_list = sorted(list(self.token_to_ids.values()))
        for id in id_list:
            token = self.id_to_tokens[id]
            self.frequency.append(self.counter[token]) # used for negative sampling

    def convert_tokens_to_ids(self, tokens):
        return convert_by_vocab(self.token_to_ids, tokens)

    def convert_ids_to_tokens(self, ids):
        return convert_by_vocab(self.id_to_tokens, ids)
