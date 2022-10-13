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

#!/usr/bin/env bash
echo "Warning: running this script would take a lot of time!"
echo "### Preprocess ###"


python gen_seq.py --phisher=True --deanon=True --mev=True --bizdate=xxx
python gen_pretrain_data.py --dupe_factor=10 --bizdate=xxx --do_eval=False
python run_pretrain.py --bizdate=xxx --epoch=5 --neg_strategy=zip --checkpointDir=xxx
python run_embed.py --init_checkpoint=xxx/xxx --bizdate=xxx --neg_strategy=zip
