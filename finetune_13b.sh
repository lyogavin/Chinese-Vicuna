TOT_CUDA="0"
CUDAs=(${TOT_CUDA//,/ })
CUDA_NUM=${#CUDAs[@]}
PORT="12345"

DATA_PATH="./sample/merge_sample.json" #"../dataset/instruction/guanaco_non_chat_mini_52K-utf8.json" #"./sample/merge_sample.json"
OUTPUT_PATH="lora-Vicuna"
MODEL_PATH="decapoda-research/llama-13b-hf"
lora_checkpoint="/home/ubuntu/cloudfs/Chinese-Vicuna/lora-Vicuna/Chinese-Vicuna-lora-13b-belle-and-guanaco"
TEST_SIZE=1

CUDA_VISIBLE_DEVICES="0" nohup python  finetune.py \
--data_path $DATA_PATH \
--output_path $OUTPUT_PATH \
--model_path $MODEL_PATH \
--eval_steps 200 \
--save_steps 200 \
--test_size $TEST_SIZE &
