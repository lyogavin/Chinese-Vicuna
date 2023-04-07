

set -x -e

run_ts=$(date +%s)
echo "RUN TS: $run_ts"

echo "START TIME: $(date)"


ROOT_DIR_BASE=/home/ubuntu/cloudfs/saved_models/cn_alpaca_lora
OUTPUT_PATH=$ROOT_DIR_BASE/output_$run_ts

mkdir -p $OUTPUT_PATH


DATA_PATH="/home/ubuntu/cloudfs/ghost_data/newred_redbook_link_download/api_0305_download/merge_all_till0305_with_multichoice_scores_and_templates_1679690410.csv.tgz" #"../dataset/instruction/guanaco_non_chat_mini_52K-utf8.json" #"./sample/merge_sample.json"

#MODEL_PATH="decapoda-research/llama-7b-hf"
MODEL_PATH="decapoda-research/llama-13b-hf"

# using checkpoint-final will cuase issue
#lora_checkpoint="/home/ubuntu/cloudfs/Chinese-Vicuna/lora-Vicuna/checkpoint-final"
#lora_checkpoint="Chinese-Vicuna/Chinese-Vicuna-lora-13b-belle-and-guanaco"
lora_checkpoint="/home/ubuntu/cloudfs/Chinese-Vicuna/lora-Vicuna/Chinese-Vicuna-lora-13b-belle-and-guanaco"
TEST_SIZE=0.05
from_data_beginning=True # False

#--use_test \

CUDA_VISIBLE_DEVICES="0" python finetune_multichoice_tt_til0305.py \
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
--use_test \
--ignore_data_skip $from_data_beginning

