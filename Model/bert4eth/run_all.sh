#!/usr/bin/env bash
echo "Warning: running this script would take a lot of time!"
echo "### Preprocess ###"


python gen_seq.py --phisher=True --deanon=True --mev=True --bizdate=xxx
python gen_pretrain_data.py --dupe_factor=10 --bizdate=xxx --do_eval=False
python run_pretrain.py --bizdate=xxx --epoch=5 --neg_strategy=zip --checkpointDir=xxx
python run_embed.py --init_checkpoint=xxx/xxx --bizdate=xxx --neg_strategy=zip
