# coding=utf8
import os
#import pytorch_lightning as pl
#from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from datasets import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer
import pandas as pd
import numpy as np
import torch
from functools import partial





## example:
'你是小红书文案创作者，需要根据文案撰写标题。\n一种起标题的方式是套用"标题的悬念感比较强"模版。其原理是：只强调、夸张结果，不提解决方案，保留部分信息，引发好奇，促进点击。\n比如这个标题："摆脱焦虑迷茫，读完这6本书，我突然开窍了"。\n比如这个标题："哭了， 我参加校招的时候怎么没刷到这个❗"。\n比如这个标题："已成功减肥70斤‼️练这个真的瘦太快啦??"。\n比如这个标题："感谢小红书，让我1.5k就装修好modelY"。\n比如这个标题："差点以为这小子抽到真的摩拉克斯了！"。' \
    '\n根据以下文案，按照这个模版，创作小红书标题：\n文案：{content}\n标题：'



# follow BELLE's imeplementation as we pretrain based on BELLE, keep it consistent
# https://github.com/LianjiaTech/BELLE/blob/de98f72dedc542ead65dfb2b7d24602e07266bb0/train/finetune.py#L95
def tokenize(prompt, tokenizer=None, add_eos_token=True, cutoff_len=-1):
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=cutoff_len,
        padding=False,
        return_tensors=None,
    )
    if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < cutoff_len
            and add_eos_token
    ):
        result["input_ids"].append(tokenizer.eos_token_id)
        result["attention_mask"].append(1)

    if add_eos_token and len(result["input_ids"]) >= cutoff_len:
        result["input_ids"][cutoff_len - 1] = tokenizer.eos_token_id
        result["attention_mask"][cutoff_len - 1] = 1

    result["labels"] = result["input_ids"].copy()

    return result

g_first_time_asserted = False

def generate_and_tokenize_prompt(data_point, tokenizer=None, max_seq_length=-1):
    global g_first_time_asserted

    if not g_first_time_asserted:
        g_first_time_asserted = True
        assert tokenizer.pad_token_id == 0
        assert tokenizer.padding_side == "left"





    prompt_part = f'你是小红书博主，需要根据输入的内容，创作爆款小红书标题。\n需要起标题的内容：{data_point["tags"]}\n爆款标题：'



    # 1.5 truncate target
    truncated_title = data_point['title']
    truncated_title = truncated_title[:120] if isinstance(truncated_title, str) else ' '


    # 3. get input
    input_text = prompt_part

    #instruction = data_point['instruction']
    #input_text = data_point["input"]
    #input_text = "Human: " + instruction + input_text + "\n\nAssistant: "

    input_text = tokenizer.bos_token + input_text if tokenizer.bos_token!=None else input_text


    # 4. get target
    target_text = truncated_title + tokenizer.eos_token

    # !!!! tokenize first, then jion to keep consistent with inferrence !!!!

    # 5. put everything together...
    full_prompt = input_text+target_text
    #tokenized_full_prompt = tokenize(full_prompt, tokenizer, cutoff_len=max_seq_length)
    tokenized_input = tokenize(input_text, tokenizer, add_eos_token=False, cutoff_len=max_seq_length)
    tokenized_target = tokenize(target_text, tokenizer, add_eos_token=False, cutoff_len=max_seq_length)

    train_on_inputs = False
    if not train_on_inputs:

        # 6. set prompt part to -100
        # better user target len as there might be cases multi char maps to one token like "：我"
        #user_prompt = input_text
        tokenized_input['input_ids'] = tokenized_input['input_ids'] + tokenized_target['input_ids']

        target_len = len(tokenized_target["input_ids"]) #- 1 # bos

        tokenized_input["labels"] = [
                -100
            ] * (len(tokenized_input["input_ids"]) - target_len) + tokenized_target["input_ids"]

        tokenized_input['attention_mask'] = [1] * (len(tokenized_input["input_ids"]))
    return tokenized_input










if __name__ == '__main__':
    import argparse
    #modelfile = '/cognitive_comp/wuziwei/pretrained_model_hf/medical_v2'
    #datafile = '/home/ubuntu/cloudfs/ghost_data/merge_all_add_1208_1228//merge_all_0108_with_multichoice_scores_and_templates_val_sample_1673194850.csv.tgz'
    data_file='/home/ubuntu/cloudfs/ghost_data/title_multi_choice_training/hot_title_tags2title_val_0418_1681864800.csv.tgz'

    parser = argparse.ArgumentParser(description='hf test', allow_abbrev=False)
    group = parser.add_argument_group(title='test args')
    group.add_argument('--pretrained-model-path', type=str, default="/home/ubuntu/cloudfs/saved_models/bigscience/bloomz-3b",
                       help='Number of transformer layers.')
    group.add_argument('--max-seq-length', type=int, default=100)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
            "BelleGroup/BELLE-7B-2M", #, add_eos_token=True
            proxies={
                'http':'socks5h://exmpl:abcd1234@43.156.235.42:8128',
                'https':'socks5h://exmpl:abcd1234@43.156.235.42:8128'}
        )

    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"


    df = pd.read_csv(data_file, usecols=['title', 'tags'])
    print(f"df types: {df.dtypes}")
    print(f"df head: {df.head()}")


    data = Dataset.from_pandas(df)
    cols = data.column_names
    train_data = data.shuffle().map(partial(generate_and_tokenize_prompt, tokenizer=tokenizer, max_seq_length=105),
                                            remove_columns = [x for x in cols if x not in ['title', 'tags']])


    print(f"\n\n1. going through training dataset, asserting everything")
    for i, batch in enumerate(train_data):
        #print(f"{i}: {batch}")
        decoded = tokenizer.decode(batch['input_ids'], skip_special_tokens=True)
        decoded_labels = tokenizer.decode([x if x!=-100 else 0 for x in batch['labels']], skip_special_tokens=True)
        #print(f"decoded: {decoded}")
        #print(f"decoded label: {decoded_labels}")

        assert len(batch['input_ids']) == len(batch['labels'])
        assert len(batch['input_ids']) == len(batch['attention_mask'])
        assert len(batch['input_ids']) <= 105
        assert batch['input_ids'][0] == tokenizer.encode(tokenizer.bos_token)[0]
        assert batch['input_ids'][-1] == tokenizer.encode(tokenizer.eos_token)[0]
        assert tokenizer.encode(tokenizer.bos_token)[0] not in batch['input_ids'][1:]
        assert tokenizer.encode(tokenizer.eos_token)[0] not in batch['input_ids'][:-1]

        assert decoded_labels.endswith(batch['title']), f"{decoded_labels} has to be equal to  {batch['title']}, content: {batch['content']}" \
            f"batch:{batch}"
        assert decoded.endswith(decoded_labels)

        if -100 in batch['labels']:
            rindex = len(batch['labels']) - batch['labels'][-1::-1].index(-100) - 1
            assert -100 not in batch['labels'][rindex+1:]
            assert len([x for x in batch['labels'][:rindex] if x != -100]) ==0
            assert batch['labels'][rindex+1:] == batch['input_ids'][rindex+1:]
            assert tokenizer.decode(batch['labels'][rindex+1:], skip_special_tokens=True) == decoded_labels






    #print(testml[10])
    #print(testml.tokenizer.decode(testml[10]['input_ids']))
    #print(testml.tokenizer.decode(testml[23]['input_ids']))
    #print(testml.tokenizer.decode(testml[17]['input_ids']))
    #print(len(testml))
    #print(f"max len:{testml.max_seq_length}")



    from tqdm import tqdm

    print(f"3. go over whole train data to make sure no exception...")

    dataset = Dataset.from_pandas(df).shuffle().map(partial(generate_and_tokenize_prompt, tokenizer=tokenizer, max_seq_length=650))



    #dl = DataLoader(testml, batch_size=4, num_workers=32,pin_memory=False)
    for x in tqdm(dataset, total=len(dataset)):
        abc = x

    #print(f"test iterate through all val data...")
    #testml = GPT2QADataset(
    #    "/home/ubuntu/cloudfs/ghost_data/merge_all_add_1208_1228//merge_all_0108_with_multichoice_scores_and_templates_val_1673194850.csv.tgz",
    #    'medical_qa', args=args)

    #dl = DataLoader(testml, batch_size=4, num_workers=32,pin_memory=False)
    #for x in tqdm(testml, total=len(testml)):
    #    abc = x