

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
TEST_SIZE=50
#from_data_beginning=True # False



#--model_name_or_path "openai/whisper-large-v2" \
#--language "Marathi" \
#--language_abbr "mr" \
#--task "transcribe" \
#--dataset_name "mozilla-foundation/common_voice_11_0" \
#--push_to_hub \
#--hub_token $HUB_TOKEN \
#--config_file config.yaml



# lr follow adalora paper Appendix.E,  language gen, but much bigger model so bigger lr:
# theirs: bart-large: batch 32/64 5e-4
# adalora hyper parms also follow paper Appendix.E
# we use 8% params according to the curve in page 9
# ref appendix C for init r final r
accelerate launch  finetune_belle_hot_title_filtered_data_adalora_0427.py \
    --debug_mode \
    --data_path $DATA_PATH \
    --run_ts $run_ts \
    --model_path $MODEL_PATH \
    --tokenizer_path $TOKENIZER_PATH \
    --preprocessing_num_workers 2 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --dataloader_pin_memory \
    --dataloader_num_workers 2 \
    --learning_rate 1e-3 \
    --weight_decay 1e-4 \
    --num_train_epochs 3 \
    --test_size $TEST_SIZE \
    --max_seq_len 200 \
    --gradient_accumulation_steps 16 \
    --lr_scheduler_type "linear" \
    --num_warmup_steps 50 \
    --output_dir $OUTPUT_PATH \
    --seed 42 \
    --load_best_model \
    --with_tracking \
    --report_to "wandb" \
    --checkpointing_steps 2000 \
    --evaluation_steps 2000 \
    --logging_steps 25 \
    --use_peft \
    --use_adalora \
    --init_r 32 \
    --target_r 24 \
    --tinit 100 \
    --tfinal 800 \
    --delta_t 10 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --orth_reg_weight 0.5

