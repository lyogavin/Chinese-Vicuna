

set -x -e

run_ts=$(date +%s)
echo "RUN TS: $run_ts"

echo "START TIME: $(date)"


ROOT_DIR_BASE=/home/ubuntu/cloudfs/saved_models/cn_alpaca_lora
OUTPUT_PATH=$ROOT_DIR_BASE/output_$run_ts

mkdir -p $OUTPUT_PATH

TOT_CUDA="0"
CUDAs=(${TOT_CUDA//,/ })
CUDA_NUM=${#CUDAs[@]}
PORT="12345"

DATA_PATH="/home/ubuntu/cloudfs/ghost_data/newred_redbook_link_download/api_0305_download/merge_all_till0305_with_multichoice_scores_and_templates_val_sample_1679690410.csv.tgz" #"../dataset/instruction/guanaco_non_chat_mini_52K-utf8.json" #"./sample/merge_sample.json"

MODEL_PATH="decapoda-research/llama-7b-hf"

# using checkpoint-final will cuase issue
lora_checkpoint="/home/ubuntu/cloudfs/Chinese-Vicuna/lora-Vicuna/checkpoint-11600"
TEST_SIZE=200



CUDA_VISIBLE_DEVICES=${TOT_CUDA} torchrun --nproc_per_node=$CUDA_NUM --master_port=$PORT finetune_multichoice_tt_til0305.py \
--data_path $DATA_PATH \
--output_path $OUTPUT_PATH \
--model_path $MODEL_PATH \
--eval_steps 200 \
--save_steps 200 \
--test_size $TEST_SIZE \
--resume_from_checkpoint $lora_checkpoint \
--run_ts $run_ts \
--max_seq_len 700 \
--wandb \
--use_test

