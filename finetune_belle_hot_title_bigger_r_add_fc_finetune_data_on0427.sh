

set -x -e

run_ts=$(date +%s)
echo "RUN TS: $run_ts"

echo "START TIME: $(date)"


ROOT_DIR_BASE=/home/ubuntu/cloudfs/saved_models/belle7blora
OUTPUT_PATH=$ROOT_DIR_BASE/output_$run_ts

mkdir -p $OUTPUT_PATH
mkdir -p /home/ubuntu/cloudfs/logs


DATA_PATH="/home/ubuntu/cloudfs/ghost_data/newred_redbook_link_download/api_0305_download/all_merged_til0305_cleanedup_filtered_for_peft_training_0426_train_1679092112.csv.tgz" #"../dataset/instruction/guanaco_non_chat_mini_52K-utf8.json" #"./sample/merge_sample.json"

#MODEL_PATH="decapoda-research/llama-7b-hf"
MODEL_PATH="/home/ubuntu/cloudfs/saved_models/BelleGroup/BELLE-7B-2M/"

# use local model
#MODEL_PATH="/home/ubuntu/cloudfs/saved_models/decapoda-research/llama-13b-hf"
TOKENIZER_PATH="BelleGroup/BELLE-7B-2M"
# using checkpoint-final will cuase issue
#lora_checkpoint="/home/ubuntu/cloudfs/Chinese-Vicuna/lora-Vicuna/checkpoint-final"
#lora_checkpoint="Chinese-Vicuna/Chinese-Vicuna-lora-13b-belle-and-guanaco"
#lora_checkpoint="/home/ubuntu/cloudfs/Chinese-Vicuna/lora-Vicuna/Chinese-Vicuna-lora-13b-belle-and-guanaco"
TEST_SIZE=5
from_data_beginning=True # False

#--use_test \
#--resume_from_checkpoint $lora_checkpoint \

CUDA_VISIBLE_DEVICES="0" nohup python finetune_belle_hot_title_bigger_r_add_fc_finetune_data_0427.py \
--data_path $DATA_PATH \
--output_path $OUTPUT_PATH \
--model_path $MODEL_PATH \
--tokenizer_path $TOKENIZER_PATH \
--eval_steps 15 \
--save_steps 60 \
--test_size $TEST_SIZE \
--run_ts $run_ts \
--max_seq_len 800 \
--wandb \
--ignore_data_skip $from_data_beginning &

