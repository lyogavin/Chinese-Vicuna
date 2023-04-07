
import os
import logging, sys

import torch
import torch.nn as nn
import bitsandbytes as bnb
from datasets import load_dataset, Dataset
import pandas as pd
import transformers
import argparse
import warnings
from functools import partial

from MultichoiceTemplateALpacaLORAWAYDataset import generate_and_tokenize_prompt, df_cols_to_use

assert (
    "LlamaTokenizer" in transformers._import_structure["models.llama"]
), "LLaMA is now in HuggingFace's main branch.\nPlease reinstall it: pip uninstall transformers && pip install git+https://github.com/huggingface/transformers.git"
from transformers import LlamaForCausalLM, LlamaTokenizer
from peft import (
    prepare_model_for_int8_training,
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)
from peft import PeftModel


model = LlamaForCausalLM.from_pretrained(
    args.model_path,
    load_in_8bit=True,
    device_map=device_map,
    proxies={
        'http':'socks5h://exmpl:abcd1234@43.156.235.42:8128',
        'https':'socks5h://exmpl:abcd1234@43.156.235.42:8128'}
)
OUTPUT_DIR = "/home/ubuntu/cloudfs/saved_models/decapoda-research/llama-13b-hf"

model.save_pretrained(OUTPUT_DIR)


logger.info(f"saved pretrained into: {OUTPUT_DIR}")

