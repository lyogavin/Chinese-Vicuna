
import os
import logging, sys

# proxy for wandb
#os.environ["HTTPS_PROXY"] = "https://exmpl:abcd1234@43.156.235.42:8128"
#os.environ["HTTP_PROXY"] = "http://exmpl:abcd1234@43.156.235.42:8128"


import torch
import torch.nn as nn
import bitsandbytes as bnb
from datasets import load_dataset, Dataset
import pandas as pd
import transformers
import argparse
import warnings
from functools import partial

from generate_and_tokenize_prompt_belle import generate_and_tokenize_prompt, df_cols_to_use

assert (
    "LlamaTokenizer" in transformers._import_structure["models.llama"]
), "LLaMA is now in HuggingFace's main branch.\nPlease reinstall it: pip uninstall transformers && pip install git+https://github.com/huggingface/transformers.git"
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from peft import (
    prepare_model_for_int8_training,
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)
from peft import PeftModel

parser = argparse.ArgumentParser()
parser.add_argument("--wandb", action="store_true", default=False)
parser.add_argument("--use_test", action="store_true", default=False)
parser.add_argument("--data_path", type=str, default="merge.json")
parser.add_argument("--output_path", type=str, default="belle_7b_2m")
parser.add_argument("--model_path", type=str, default="BelleGroup/BELLE-7B-2M")
parser.add_argument("--tokenizer_path", type=str, default="BelleGroup/BELLE-7B-2M")

parser.add_argument("--eval_steps", type=int, default=200)
parser.add_argument("--save_steps", type=int, default=200)
parser.add_argument("--run_ts", type=int)
parser.add_argument("--test_size", type=float, default=0.05)
parser.add_argument("--resume_from_checkpoint", type=str, default=None)
parser.add_argument("--ignore_data_skip", type=str, default="False")
parser.add_argument("--max_seq_len", type=int, default=1000)
args = parser.parse_args()

if args.use_test:
    print(f"use test, eval/save steps set to 50")
    args.eval_steps = 2
    args.save_steps = 2



logging_file_path = f"/home/ubuntu/cloudfs/logs/training_log_{args.run_ts}.log"

handlers = [
    logging.FileHandler(logging_file_path),
    logging.StreamHandler(sys.stdout)
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=handlers
)
logger = logging.getLogger(__name__)

logger.info(f"logging in file: {logging_file_path}")
logger.info(f"running args: {args}")


if not args.wandb:
    os.environ["WANDB_MODE"] = "disable"

USE_TEST = args.use_test



# optimized for RTX 4090. for larger GPUs, increase some of these?
MICRO_BATCH_SIZE = 4  # this could actually be 5 but i like powers of 2
BATCH_SIZE = 128
MAX_STEPS = None
GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // MICRO_BATCH_SIZE
EPOCHS = 10  # we don't always need 3 tbh
LEARNING_RATE = 1e-4  # the Karpathy constant
CUTOFF_LEN = 256  # 256 accounts for about 96% of the data
LORA_R = 16
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
VAL_SET_SIZE = args.test_size #2000

if USE_TEST:
    EPOCHS = 1

if VAL_SET_SIZE > 1.:
    VAL_SET_SIZE = int(VAL_SET_SIZE)


TARGET_MODULES = ['query_key_value']
DATA_PATH = args.data_path #"/home/cciip/private/fanchenghao/dataset/instruction/merge.json"
OUTPUT_DIR = args.output_path #"lora-Vicuna"

device_map = "auto"
world_size = int(os.environ.get("WORLD_SIZE", 1))
ddp = world_size != 1
if ddp:
    device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
    GRADIENT_ACCUMULATION_STEPS = GRADIENT_ACCUMULATION_STEPS // world_size
logger.info(args.model_path)
model = AutoModelForCausalLM.from_pretrained(
    args.model_path,
    load_in_8bit=True,
    device_map=device_map,
    cache_dir="/home/ubuntu/cloudfs/huggingfacecache/",
    proxies={
        'http':'socks5h://exmpl:abcd1234@43.156.235.42:8128',
        'https':'socks5h://exmpl:abcd1234@43.156.235.42:8128'}
)
tokenizer = AutoTokenizer.from_pretrained(
    args.tokenizer_path, #, add_eos_token=True
    cache_dir="/home/ubuntu/cloudfs/huggingfacecache/",
    proxies={
        'http':'socks5h://exmpl:abcd1234@43.156.235.42:8128',
        'https':'socks5h://exmpl:abcd1234@43.156.235.42:8128'}
)

model = prepare_model_for_int8_training(model)

config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=TARGET_MODULES,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM",
)


model = get_peft_model(model, config)


LOAD_PEFT_CHECKPOINT_FROM_PRETRAIN = False
# https://github.com/tloen/alpaca-lora/issues/253
# to fix Parameter at index 127 has been marked as ready twice issue
# ^^^^^ -> not working!

if LOAD_PEFT_CHECKPOINT_FROM_PRETRAIN:
    model=PeftModel.from_pretrained(model,
                                    args.resume_from_checkpoint,
                                    torch_dtype=torch.float16,
                                    device_map=device_map,
                                   ) #"/lora-alpaca-output-dir")

    logger.info(f"loaded checkpoint: {args.resume_from_checkpoint}")
    args.resume_from_checkpoint = None


    logger.info(f"loaded checkpoint, setting to NONE: {args.resume_from_checkpoint}")

tokenizer.pad_token_id = 0  # unk. we want this to be different from the eos token
tokenizer.padding_side = "left"  # Allow batched inference

#data = load_dataset("json", data_files=DATA_PATH)

df = pd.read_csv(DATA_PATH)

#logger.info(f"model from: get_peft_model: {model}")

if USE_TEST:
    logger.info(f"USE_TEST, sampled 100")
    df = df.sample(100)

logger.info(f"loadded df from: {DATA_PATH}, len:{len(df)}")



data = Dataset.from_pandas(df[df_cols_to_use])



val_size_items = VAL_SET_SIZE


if isinstance(VAL_SET_SIZE, float):
    logger.info(f"converting float val size: {VAL_SET_SIZE}")

    val_size_items = int(len(data) * VAL_SET_SIZE)
    logger.info(f"converted to: val size: {val_size_items}")


now_max_steps = max((len(data) - val_size_items) // BATCH_SIZE * EPOCHS, EPOCHS)

MAX_STEPS = now_max_steps

logger.info(f"max steps: {MAX_STEPS}, epochs: {EPOCHS}")

if args.resume_from_checkpoint:
# Check the available weights and load them
    checkpoint_name = os.path.join(
        args.resume_from_checkpoint, "pytorch_model.bin"
)  # Full checkpoint
    if not os.path.exists(checkpoint_name):
        pytorch_bin_path = checkpoint_name
        checkpoint_name = os.path.join(
            args.resume_from_checkpoint, "adapter_model.bin"
        )  # only LoRA model - LoRA config above has to fit
        if os.path.exists(checkpoint_name):
            os.rename(checkpoint_name, pytorch_bin_path)
            warnings.warn("The file name of the lora checkpoint'adapter_model.bin' is replaced with 'pytorch_model.bin'")
        else:
            args.resume_from_checkpoint = (
                None  # So the trainer won't try loading its state
            )
    # The two files above have a different name depending on how they were saved, but are actually the same.
    if os.path.exists(checkpoint_name):
        logger.info(f"Restarting from {checkpoint_name}")
        adapters_weights = torch.load(checkpoint_name)
        #logger.info(f"before set_peft_model_state_dict: model:{model}")
        #logger.info(f"weights to load: {adapters_weights}")
        #model = set_peft_model_state_dict(model, adapters_weights)
        # latest version of peft doesn't return any more
        set_peft_model_state_dict(model, adapters_weights)
        #logger.info(f"set_peft_model_state_dict: model:{model}")
    else:
        logger.info(f"Checkpoint {checkpoint_name} not found")
    
    train_args_path = os.path.join(args.resume_from_checkpoint, "trainer_state.json")
    
    if os.path.exists(train_args_path):
        logger.info(f"loading: {train_args_path}")
        import json
        base_train_args = json.load(open(train_args_path, 'r'))
        base_max_steps = base_train_args["max_steps"]
        resume_scale = base_max_steps / now_max_steps
        if base_max_steps > now_max_steps:
            warnings.warn("epoch {} replace to the base_max_steps {}".format(EPOCHS, base_max_steps))
            EPOCHS = None
            MAX_STEPS = base_max_steps
        else:
            MAX_STEPS = now_max_steps
else:
    MAX_STEPS = now_max_steps

#logger.info(f"loaded weight from {checkpoint_name}, model: {model}")

model.print_trainable_parameters()

cols = data.column_names

if VAL_SET_SIZE > 0:
    train_val = data.train_test_split(
        test_size=VAL_SET_SIZE, shuffle=True, seed=42
    )
    train_data = train_val["train"].shuffle().map(partial(generate_and_tokenize_prompt, tokenizer=tokenizer, max_seq_length=args.max_seq_len),
                                                 remove_columns = cols)
    val_data = train_val["test"].shuffle().map(partial(generate_and_tokenize_prompt, tokenizer=tokenizer, max_seq_length=args.max_seq_len),
                                                 remove_columns = cols)
else:
    train_data = data.shuffle().map(partial(generate_and_tokenize_prompt, tokenizer=tokenizer, max_seq_length=args.max_seq_len),
                                                 remove_columns = cols)
    val_data = None

#for batch in train_data:
#    print(batch)
#    break


training_args = transformers.TrainingArguments(
        per_device_train_batch_size=MICRO_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        warmup_steps=100,
        log_level='info',
        num_train_epochs=EPOCHS,
        max_steps=MAX_STEPS,
        learning_rate=LEARNING_RATE,
        fp16=True,
        logging_steps=20,
        evaluation_strategy="steps" if VAL_SET_SIZE > 0 else "no",
        save_strategy="steps",
        eval_steps=args.eval_steps if VAL_SET_SIZE > 0 else None,
        save_steps=args.save_steps,
        output_dir=OUTPUT_DIR,
        save_total_limit=10,
        load_best_model_at_end=True if VAL_SET_SIZE > 0 else False,
        ddp_find_unused_parameters=False if ddp else None,
        report_to="wandb" if args.wandb else [],
        ignore_data_skip=args.ignore_data_skip,
    )

class MyCallback(transformers.TrainerCallback):
    "A callback that prints a message at the beginning of training"

    def on_evaluate(self, args, state, control, **kwargs):
        if "model" in kwargs:
            logger.info("on_evaluate...")
            inputs = "你好,中国的首都在哪里？"  # "你好,美国的首都在哪里？"
            logger.info(f"test input: {inputs}")
            #tokenizer = kwargs['tokenizer']
            model = kwargs['model']
            input_ids = tokenizer(inputs, return_tensors="pt")['input_ids']
            input_ids = input_ids.to('cuda')
            generation_output = model.generate(
                input_ids=input_ids,
                max_new_tokens=35,
            )
            #print(generation_output)
            logger.info(tokenizer.decode(generation_output[0]))

            inputs='''你是小红书文案创作者，需要根据文案撰写标题。
一种起标题的方式是套用"标题的悬念感比较强"模版。其原理是：只强调、夸张结果，不提解决方案，保留部分信息，引发好奇，促进点击。
比如这个标题："摆脱焦虑迷茫，读完这6本书，我突然开窍了"。
比如这个标题："哭了， 我参加校招的时候怎么没刷到这个❗"。
比如这个标题："已成功减肥70斤‼️练这个真的瘦太快啦??"。
比如这个标题："感谢小红书，让我1.5k就装修好modelY"。
比如这个标题："差点以为这小子抽到真的摩拉克斯了！"。
根据以下文案，按照这个模版，创作小红书标题：
文案：法式吊灯 吸顶灯
标题：'''
            logger.info(f"test input: {inputs}")
            #tokenizer = kwargs['tokenizer']
            model = kwargs['model']
            input_ids = tokenizer(inputs, return_tensors="pt")['input_ids']
            input_ids = input_ids.to('cuda')

            generation_config = GenerationConfig(
                temperature=1.,
                top_p=0.9,
                top_k=3,
                num_beams=4,
                bos_token_id=1,
                eos_token_id=2,
                pad_token_id=0,
                max_new_tokens=35,  # max_length=max_new_tokens+input_sequence
                min_new_tokens=10,  # min_length=min_new_tokens+input_sequence
                #num_return_sequences=2,
                #**kwargs,
            )

            generation_output = model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                max_new_tokens=35,
            )
            #print(generation_output)
            logger.info(tokenizer.decode(generation_output[0]))
        else:
            logger.info(f"model not found in kwargs, skipping")



logger.info(f"training args: {training_args}")
trainer = transformers.Trainer(
    model=model,
    train_dataset=train_data,
    eval_dataset=val_data,
    #data_collator=transformers.DataCollatorWithPadding(),
    callbacks=[MyCallback],
    args=training_args,
    data_collator=transformers.DataCollatorForSeq2Seq(tokenizer, label_pad_token_id=0, pad_to_multiple_of=8, return_tensors="pt")
)
model.config.use_cache = False

old_state_dict = model.state_dict
model.state_dict = (
    lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())
).__get__(model, type(model))

if torch.__version__ >= "2" and sys.platform != "win32":
    model = torch.compile(model)

logger.info("\n If there's a warning about missing keys above, please disregard :)")

trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

model.save_pretrained(OUTPUT_DIR)


logger.info(f"saved pretrained into: {OUTPUT_DIR}")

logger.info(f"training done!")

