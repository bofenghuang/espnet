#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

export WANDB_MODE=disabled
# export WANDB_PROJECT=espnet-asr-hmhm
# export WANDB_NAME=wav2vec2-xls-r-1b-cv-fr-ft

# export CUDA_VISIBLE_DEVICES=5,4,3,2
export CUDA_VISIBLE_DEVICES=5
# export CUDA_VISIBLE_DEVICES=2,1

# train_set="train_960"
# valid_set="dev"
# test_sets="test_clean test_other dev_clean dev_other"

train_set="train_hmhm"
valid_set="test_hmhm"
test_sets="test_hmhm"

# asr_config=conf/train_asr_conformer.yaml
# asr_config=conf/tuning/train_asr_transformer3_w2v_large_fr_7k_finetuning_last_1layer.yaml
asr_config=conf/tuning/train_asr_conformer7_wav2vec2_hmhm.yaml
lm_config=conf/tuning/train_lm_transformer2_hmhm.yaml
# lm_config=conf/train_rnn_lm.yaml
inference_config=conf/decode_asr.yaml

# ./asr.sh \
#     --lang en \
#     --ngpu 4 \
#     --nbpe 5000 \
#     --max_wav_duration 30 \
#     --speed_perturb_factors "0.9 1.0 1.1" \
#     --asr_config "${asr_config}" \
#     --lm_config "${lm_config}" \
#     --inference_config "${inference_config}" \
#     --train_set "${train_set}" \
#     --valid_set "${valid_set}" \
#     --test_sets "${test_sets}" \
#     --lm_train_text "data/${train_set}/text data/local/other_text/text" \
#     --bpe_train_text "data/${train_set}/text" "$@"


# --feats_normalize uttmvn to reduce the time used in collect_stats step
# use another frontend: https://github.com/espnet/espnet/issues/3961
# --nbpe 2000 in swbd, --nbpe 128 \
# 

# --asr_tag train_asr_conformer7_wav2vec2_xls_r_1b_cv_fr_raw_fr_bpe128_sp \
# --asr_tag "train_asr_conformer7_wav2vec2_fr_7k_large_raw_fr_bpe128_sp_w2v_ft" \
# --asr_tag "train_asr_conformer7_wav2vec2_fr_7k_large_raw_fr_bpe128_sp_w2v_ft_ctc_weight_0.7" \

# --asr_tag "train_asr_conformer7_wav2vec2_hmhm_raw_fr_bpe2000_sp_frozen_w2v" \

./asr.sh \
    --stage 11 \
    --nj 32 \
    --inference_nj 10 \
    --lang fr \
    --ngpu 1 \
    --nbpe 2000 \
    --min_wav_duration 1 \
    --max_wav_duration 30 \
    --feats_normalize uttmvn \
    --speed_perturb_factors "0.9 1.0 1.1" \
    --asr_config "${asr_config}" \
    --lm_config "${lm_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --lm_train_text "data/${train_set}/text" \
    --bpe_train_text "data/${train_set}/text" "$@"


# --inference_asr_model "valid.acc.best.pth"
# --stage 6 --stop_stage 8