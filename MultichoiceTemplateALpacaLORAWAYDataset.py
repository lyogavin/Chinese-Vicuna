# coding=utf8
import os
#import pytorch_lightning as pl
#from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from datasets import Dataset
from tqdm import tqdm
from transformers import LlamaTokenizer
import pandas as pd
import numpy as np
import torch
from functools import partial

def generate_and_tokenize_prompt(data_point, tokenizer=None, max_seq_length=100):
    def get_promote_title_type(row):
        return row['title_template_name']

    truncated_title = data_point['title']
    truncated_title = truncated_title[:120] if isinstance(truncated_title, str) else ' '

    instruction_prompt_first = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
你是自媒体创作者，需要为输入的文案内容，撰写指定类型的爆款标题。需要撰写的标题类型：{get_promote_title_type(data_point)}。

### Input:
{data_point['content']}"""

    instruction_prompt_second = f"""

### Response:{truncated_title}
"""

    second_input_ids = tokenizer.encode(instruction_prompt_second,
                                      truncation=False,
                                      max_length=max_seq_length,
                                      add_special_tokens=False)

    first_input_ids = tokenizer.encode(instruction_prompt_first,
                                      truncation=True,
                                      max_length=max_seq_length - len(second_input_ids))

    input_ids = first_input_ids + second_input_ids

    truncated_title_input_ids = tokenizer.encode(truncated_title, add_special_tokens=False)

    target_len = len(truncated_title_input_ids)

    # input_ids = torch.tensor([prefix_input_ids + postfix_input_ids])
    # attention_mask = torch.tensor([[1] * (len(postfix_input_ids) + len(prefix_input_ids))])

    # inputs_dict = self.tokenizer.encode_plus(item['prompted_content'],
    #                                         max_length=self.max_seq_length, padding='max_length',
    #                                         truncation=True, return_tensors='pt')
    # prompt_inputs_dict = self.tokenizer.encode_plus(item['prompt'],
    #                                         max_length=self.max_seq_length, padding=False,
    #                                         truncation=True, return_tensors='pt')
    #target = input_ids
    #labels = target.clone().detach()
    #labels[target == tokenizer.pad_token_id] = -100
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),  # attention_mask.squeeze(),
        "labels": [-100] * (len(input_ids) - target_len) + truncated_title_input_ids,
        # labels.squeeze(),
        # "labels_mask": torch.tensor([[0] * len(prefix_input_ids) + [1] * len(postfix_input_ids)])
        # "answer_end_pos":len(answer_input_ids) + len(postfix_input_ids) + len(prefix_input_ids) - 1
    }


if __name__ == '__main__':
    import argparse
    #modelfile = '/cognitive_comp/wuziwei/pretrained_model_hf/medical_v2'
    #datafile = '/home/ubuntu/cloudfs/ghost_data/merge_all_add_1208_1228//merge_all_0108_with_multichoice_scores_and_templates_val_sample_1673194850.csv.tgz'
    data_file='/home/ubuntu/cloudfs/ghost_data/newred_redbook_link_download/api_0305_download/' \
              'merge_all_till0305_with_multichoice_scores_and_templates_val_sample_1679690410.csv.tgz'

    parser = argparse.ArgumentParser(description='hf test', allow_abbrev=False)
    group = parser.add_argument_group(title='test args')
    group.add_argument('--pretrained-model-path', type=str, default="/home/ubuntu/cloudfs/saved_models/bigscience/bloomz-3b",
                       help='Number of transformer layers.')
    group.add_argument('--max-seq-length', type=int, default=1024)
    args = parser.parse_args()

    tokenizer = LlamaTokenizer.from_pretrained(
            "decapoda-research/llama-7b-hf"#, add_eos_token=True
        )

    df = pd.read_csv(data_file)

    data = Dataset.from_pandas(df).train_test_split(
            test_size=100, shuffle=True, seed=42
        )
    train_data = data["test"].shuffle().map(partial(generate_and_tokenize_prompt, tokenizer=tokenizer, max_seq_length=1000))

    for i, batch in enumerate(train_data):
        print(f"{i}: {batch}")
        decoded = tokenizer.decode(batch['input_ids'], skip_special_tokens=True)
        decoded_labels = tokenizer.decode([x if x!=-100 else 0 for x in batch['labels']], skip_special_tokens=True)
        print(f"decoded: {decoded}")
        print(f"decoded label: {decoded_labels}")

        assert len(batch['input_ids']) == len(batch['labels'])

    #print(testml[10])
    #print(testml.tokenizer.decode(testml[10]['input_ids']))
    #print(testml.tokenizer.decode(testml[23]['input_ids']))
    #print(testml.tokenizer.decode(testml[17]['input_ids']))
    #print(len(testml))
    #print(f"max len:{testml.max_seq_length}")

    for testing_type in ["newrank_healthcare"]:#, "zhihu_search", "baidubaijia"]:



        for i in range(200, 40, -1):
            print(f"\nfor max len: {i}")

            res = partial(generate_and_tokenize_prompt, tokenizer=tokenizer, max_seq_length=i)({"data_type":"redbook_content_title",
                                 "title_template_name":'制造悬念感',
                                 "source_category":testing_type, 'content':'将数据转换成模型训练的输入将数据转换成模型训练的输入将数据转换成模型训练的输入', 'title':'将数据转换成模型训练的输入'})

            print(f"encode len: {np.array(res['input_ids']).shape}")
            deres = tokenizer.decode(res['input_ids'])
            print(f"decoded encode: {deres}")


    for title_len in range(30, 150):
        title = ['数'] * title_len
        title = ''.join(title)
        print(f"title len:{title_len}")

        res = partial(generate_and_tokenize_prompt, tokenizer=tokenizer, max_seq_length=200)({"data_type":"redbook_content_title",
                             "source_category":"newrank_healthcare",
                                 "title_template_name":'权威/内部人士/过来人身份推荐背书',
                             'content':'将数据转换成模型训练的输入将数据转换成模型训练的输入将数据转换成模型训练的输入', 'title':title})

        print(f"encode len: {np.array(res['input_ids']).shape}")
        deres = tokenizer.decode(res['input_ids'])
        print(f"decoded encode: {deres}")

    from tqdm import tqdm

    print(f"test iterate through all train data...")

    dataset = Dataset.from_pandas(df).shuffle().map(partial(generate_and_tokenize_prompt, tokenizer=tokenizer))



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