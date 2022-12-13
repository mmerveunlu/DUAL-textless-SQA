# DUAL-textless-SQA 

This repository contains the implementation of DUAL model. 
The official implementation is in [DUAL-textless-SQA](https://github.com/DanielLin94144/DUAL-textless-SQA). 
Differences from the original implementation: 
 * `preprocess_utils` script was added to generate necessary files for training.
 * `notebooks` folder was added to store Jupyter notebooks. 
 * `requirements.txt` was added. 

# Steps
In this section I summarize the usage of the model with the collected dataset. 
The preprocessed and raw dataset can be found in the official code. 

## Preprocessing 

### Preparing the files for preprocess and training 
Creating meta_data.csv, segments.json and hash2question files. 

> python preprocess_utils.py --input original_train.json \
                             --audio audios/ \
                             --output train/ 

The script creates three files: meta_data.csv, data_segment_id.json and hash2question.json
The details of preprocessing can be found in `notebooks/Understand-DUAL-Preprocess.ipynb`.

### Extracting features 

1. The script `extract_features.py` is used to extract features for train and dev sets. 
Usage: 
 > python extract_features.py --part train \
                           --hubert hubert_large_ll60k \
                           --kmeans km_100h_c128/km_feat_layer_22 \
                           --meta train/meta_data.csv \
                           --audios train/audios/ \
                           --output train/train_code \
                           --audio_type mp3

2. The script `merge_passage.py` is used to merge the features of the context utterances into one. 
Usage:
> python merge_passage.py --segment data_segment_id.json \
                          --data train_code \
                          --output train_code

3. The script `alignment.py` is used to find answer spans, first in utterance, then context.
Usage: 
> python alignment.py --input textGrid/ \
                      --data train-v1.1.json \
                      --meta meta_data.csv
                      --output train_answer_span.csv

4. The script `code_answer.py` is used to find answer spans in the code, adds two columns: code start/end
Usage: 
> python code_answer.py --input train-answer-span.csv \
                        --code train_code/ \
                        --output train_final.csv
## Training 


## Results 