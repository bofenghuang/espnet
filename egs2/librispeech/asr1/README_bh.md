
```bash
# tensorboard
tensorboard --host 0.0.0.0 --logdir exp/lm_train_lm_transformer2_hmhm_fr_bpe128/tensorboard

# convert to s3prl
python /home/bhuang/asr/espnet/tools/s3prl/s3prl/upstream/wav2vec2/convert.py /home/bhuang/asr/espnet/egs2/librispeech/asr1/pretrained/wav2vec2-FR-7K-large/checkpoint_best.pt --output_dir pretrained/wav2vec2-FR-7K-large/converted_ckpts

# lm
./run_asr_hmhm.sh --stage 7 --stop_stage 8 --lm_stats_dir exp/lm_stats_fr_bpe128_flaubert_hmhm --lm_tag train_lm_transformer2_hmhm_fr_bpe128_flaubert_hmhm --num_splits_lm 10 --ngpu 1

# decode
./run_asr_hmhm.sh --stage 12 --use_lm false --inference_args "--ctc_weight 0.7" --use_ngram true --inference_ngram 6gram_hmhm_flaubert_prune1e-8.bin
```


```bash
# tokenize text
python3 -m espnet2.bin.tokenize_text --token_type bpe --bpemodel data/fr_token_list/bpe_unigram128/bpe.model --input dump/raw/lm_train.txt -f 2- --output dump/raw/lm_train_tokenized.txt

# get ngram
lmplz -S 80% -T /home/bhuang/tmp --discount_fallback -o 6 <dump/raw/lm_train_tokenized.txt > exp/ngram/6gram_hmhm.arpa
```