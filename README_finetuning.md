# Fine-tuning from the pre-trained model in Chinese

## Installation
Refer to [README instrallation](README.md#installation)

## Fine-tuning
#### Download pre-trained model
Download [BaiduCN1.2k Model](https://deepspeech.bj.bcebos.com/demo_models/baidu_cn1.2k_model_fluid.tar.gz)
```
wget https://deepspeech.bj.bcebos.com/demo_models/baidu_cn1.2k_model_fluid.tar.gz
mkdir models/baidu_cn1.2k
tar -zxvf baidu_cn1.2k_model_fluid.tar.gz -C models/baidu_cn1.2k
```
#### Prepare dataset
Prepare audio files with 16bit/16kHz. 
To use above files, you need to generate manifest files to summarize dataset for training, development and test as below.
```
{"audio_filepath": "/DeepSpeech/data/callhome/train/0719/0719_B_0008.wav", "duration": 2.36, "text": "他就是讲这个东西都收到了"}
{"audio_filepath": "/DeepSpeech/data/callhome/train/0719/0719_A_0018.wav", "duration": 1.79, "text": "哦大概应该很快了反正"}
```
You should set `audio_filepath` to your path that you put the audio file. `duration` means the audio duration in seconds and `text` means the transcript for the audio file.

When you generated mafifest files, you need to make `mean_std.npz` to normalize features with the development dataset (e.g. data/callhome/manifest.dev).
```
python tools/compute_mean_std.py \
--num_samples 2000 \
--specgram_type linear \
--manifest_path data/callhome/manifest.dev \
--output_path data/callhome/mean_std.npz
```

#### Training
Run `train.py` as below.
```
CUDA_VISIBLE_DEVICES=0 \
python -u train.py \
--batch_size=16 \
--num_epoch=50 \
--num_conv_layers=2 \
--num_rnn_layers=3 \
--rnn_layer_size=2048 \
--num_iter_print=100 \
--save_epoch=1 \
--num_samples=28041 \
--learning_rate=5e-4 \
--max_duration=27.0 \
--min_duration=0.0 \
--test_off=False \
--use_sortagrad=True \
--use_gru=True \
--use_gpu=True \
--is_local=True \
--share_rnn_weights=False \
--train_manifest='data/callhome/manifest.dev' \
--dev_manifest='data/callhome/manifest.train' \
--mean_std_path='data/callhome/mean_std.npz' \
--vocab_path='models/baidu_cn1.2k/vocab.txt' \
--init_from_pretrained_model='models/baidu_cn1.2k' \
--output_model_dir='models/deepspeech-finetuning' \
--augment_conf_path='conf/augmentation.config' \
--specgram_type='linear' \
--shuffle_method='batch_shuffle_clipped'
```
#### Evalation
Download LM in Chinese
```
cd models/lm
bash download_lm_ch.sh
```
Run `test.py` as below.
```
CUDA_VISIBLE_DEVICES=0 \
python -u test.py \
--batch_size=16 \
--beam_size=300 \
--num_proc_bsearch=8 \
--num_conv_layers=2 \
--num_rnn_layers=3 \
--rnn_layer_size=2048 \
--alpha=2.6 \
--beta=5.0 \
--cutoff_prob=0.99 \
--cutoff_top_n=40 \
--use_gru=True \
--use_gpu=True \
--share_rnn_weights=False \
--infer_manifest='data/callhome/manifest.test' \
--mean_std_path='data/callhome/mean_std.npz' \
--vocab_path='models/baidu_cn1.2k/vocab.txt' \
--model_path='models/deepspeech-finetuning/epoch_4' \ #Choose the model that achieved the lowest loss for development dataset  
--lang_model_path='models/lm/zh_giga.no_cna_cmn.prune01244.klm' \
--decoding_method='ctc_beam_search' \
--error_rate_type='cer' \
--specgram_type='linear'
```
