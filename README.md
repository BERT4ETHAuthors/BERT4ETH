# BERT4ETH

This is the code and data of the paper "BERT4ETH: Pre-training of Transformer Encoder Representation for Ethereum Fraud Detection".

The majority of code has been uploaded

We will complete this repository soon..

---
(update in Oct. 23, 2022)
## Getting Start

### Requirements
* Python >= 3.6.1
* NumPy >= 1.12.1
* TensorFlow >= 1.4.0

### Preprocess dataset 

#### Step 1: Download dataset from Google Drive. 
* Transaction Dataset:
* * [Phishing Account](https://drive.google.com/file/d/11UAhLOcffzLyPhdsIqRuFsJNSqNvrNJf/view?usp=sharing)

* * [De-anonymization](https://drive.google.com/file/d/1Yveis90jCx-nIA6pUL_4SUezMsVJr8dp/view?usp=sharing)

* * [MEV Bot](https://drive.google.com/file/d/10br9Xki_E443MJzGzQHQqLGds-uuGTRU/view?usp=sharing)

* * [Normal Account](https://drive.google.com/file/d/1-htLUymg1UxDrXcI8tslU9wbn0E1vl9_/view?usp=sharing)

* [ERC-20 Log Dataset (all in one)](https://drive.google.com/file/d/1mB2Tf7tMq5ApKKOVdctaTh2UZzzrAVxq/view?usp=sharing)

#### Step 2: Unzip dataset under the directory of "BERT4ETH/Data/" 


```sh
cd BERT4ETH/Data;
unzip ...;
``` 
The total volume of unzipped dataset is quite huge (more than 15GB).

#### Step 3: Transaction Sequence Generation

(In Step 3 we apply the transaction de-duplication strategy.)

```sh
cd Model/bert4eth;
python gen_seq.py --phisher=True \ 
                  --deanon=True \
                  --mev=True \
                  --bizdate=xxx
                  
python gen_seq_erc20.py;
``` 

### Pre-training

#### Step 0: Model Configuration

The configuration file is "Model/bert4eth/bert_config_eth_64.json"
```
{
  "attention_probs_dropout_prob": 0.2,
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.2,
  "hidden_size": 64,
  "initializer_range": 0.02,
  "intermediate_size": 64,
  "max_position_embeddings": 50,
  "num_attention_heads": 2,
  "num_hidden_layers": 2,
  "type_vocab_size": 2,
  "vocab_size": 3000000
}
```

#### Step 1: Pre-train Sequence Generation 

[//]: # (&#40;Masking, I/O separation and ERC20 log&#41;)

```sh
python gen_pretrain_data.py --dupe_factor=10 \
                            --bizdate=xxx \
                            --do_eval=False
```

#### Step 2: Pre-train BERT4ETH Model

```sh
python run_pretrain.py --bizdate=xxx \
                       --epoch=5 --neg_strategy=zip \
                       --checkpointDir=xxx
```

#### Step 3: Output Representation

```sh
python run_embed.py --init_checkpoint=xxx/xxx \
                    --bizdate=xxx \
                    --neg_strategy=zip

```

### Testing:

#### Phshing Account Detection
```sh
cd BERT4ETH/Model;
python run_dean.py
``` 

#### De-anonymization

```sh
cd BERT4ETH/Model;
python run_phisher.py
``` 

#### MEV Bot Detection

```sh
cd BERT4ETH/Model;
python run_mev_bot.py
``` 
