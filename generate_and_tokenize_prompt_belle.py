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


df_cols_to_use = ['title_template_name', 'title', 'content']

tt_df = pd.read_pickle('/home/ubuntu/cloudfs/ghost_data/newrank_hotrank_download/total_merged_downloads/' \
                 'multichoices_title_template_info_public_add_langchain_oa_templates_0327_1679930635.pickle')

tags_prompt = {'权威_内部推荐背书':'权威/内部人士/过来人身份推荐背书',
               '标题中强化对比反差冲突': '强化反差冲突对比',
               '标题中使用数字': '使用数字',
               '标题的悬念感比较强': '制造悬念感'}

tt_df['title_template_name'] = tt_df.title_template_name.apply(lambda x: tags_prompt[x] if x in tags_prompt else x)


## example:
'你是小红书文案创作者，需要根据文案撰写标题。\n一种起标题的方式是套用"标题的悬念感比较强"模版。其原理是：只强调、夸张结果，不提解决方案，保留部分信息，引发好奇，促进点击。\n比如这个标题："摆脱焦虑迷茫，读完这6本书，我突然开窍了"。\n比如这个标题："哭了， 我参加校招的时候怎么没刷到这个❗"。\n比如这个标题："已成功减肥70斤‼️练这个真的瘦太快啦??"。\n比如这个标题："感谢小红书，让我1.5k就装修好modelY"。\n比如这个标题："差点以为这小子抽到真的摩拉克斯了！"。' \
    '\n根据以下文案，按照这个模版，创作小红书标题：\n文案：{content}\n标题：'



# follow BELLE's imeplementation as we pretrain based on BELLE, keep it consistent
# https://github.com/LianjiaTech/BELLE/blob/main/train/finetune.py#L119
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

    # 1. get prompt for tt
    assert (tt_df['title_template_name'] == data_point['title_template_name']).sum() ==1, f"{data_point['title_template_name']} has to match 1 rec in {tt_df['title_template_name'].unique().tolist()}"
    langchain_oa_prompt = tt_df[tt_df['title_template_name'] == data_point['title_template_name']].iloc[0]['langchain_oa_prompt']

    # 1.5 truncate target
    truncated_title = data_point['title']
    truncated_title = truncated_title[:120] if isinstance(truncated_title, str) else ' '


    # 2. truncate content...
    empty_content_prompt = langchain_oa_prompt.format(content='')
    empty_content_prompt_token_ids = tokenize(empty_content_prompt, tokenizer, cutoff_len=max_seq_length)['input_ids']
    target_token_ids = tokenize(truncated_title, tokenizer, cutoff_len=max_seq_length)['input_ids']

    if max_seq_length - len(empty_content_prompt_token_ids) - len(target_token_ids) <=0:
        truncated_content = ''
    else:
        content_token_ids_cut_off = tokenize(data_point['content'], tokenizer,
                cutoff_len=max_seq_length - len(empty_content_prompt_token_ids) - len(target_token_ids))['input_ids']
        truncated_content = tokenizer.decode(content_token_ids_cut_off, skip_special_tokens=True)

    # 3. get input
    input_text = langchain_oa_prompt.format(content=truncated_content)

    #instruction = data_point['instruction']
    #input_text = data_point["input"]
    #input_text = "Human: " + instruction + input_text + "\n\nAssistant: "

    input_text = tokenizer.bos_token + input_text if tokenizer.bos_token!=None else input_text


    # 4. get target
    target_text = truncated_title + tokenizer.eos_token

    # 5. put everything together...
    full_prompt = input_text+target_text
    tokenized_full_prompt = tokenize(full_prompt, tokenizer, cutoff_len=max_seq_length)

    train_on_inputs = False
    if not train_on_inputs:

        # 6. set prompt part to -100
        # better user target len as there might be cases multi char maps to one token like "：我"
        #user_prompt = input_text
        tokenized_target= tokenize(target_text, tokenizer, add_eos_token=True, cutoff_len=max_seq_length)
        target_len = len(tokenized_target["input_ids"]) - 1 # bos

        tokenized_full_prompt["labels"] = [
            -100
        ] * (len(tokenized_target["input_ids"]) - target_len) + tokenized_full_prompt["labels"][
            len(tokenized_target["input_ids"]) - target_len:
        ]
    tokenized_full_prompt['full_prompt'] = full_prompt
    return tokenized_full_prompt










if __name__ == '__main__':
    import argparse
    #modelfile = '/cognitive_comp/wuziwei/pretrained_model_hf/medical_v2'
    #datafile = '/home/ubuntu/cloudfs/ghost_data/merge_all_add_1208_1228//merge_all_0108_with_multichoice_scores_and_templates_val_sample_1673194850.csv.tgz'
    data_file='/home/ubuntu/cloudfs/ghost_data/newred_redbook_link_download/api_0305_download/' \
              'merge_all_till0305_with_multichoice_scores_and_templates_val_1679690410.csv.tgz'

    parser = argparse.ArgumentParser(description='hf test', allow_abbrev=False)
    group = parser.add_argument_group(title='test args')
    group.add_argument('--pretrained-model-path', type=str, default="/home/ubuntu/cloudfs/saved_models/bigscience/bloomz-3b",
                       help='Number of transformer layers.')
    group.add_argument('--max-seq-length', type=int, default=1024)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
            "BelleGroup/BELLE-7B-2M", #, add_eos_token=True
            proxies={
                'http':'socks5h://exmpl:abcd1234@43.156.235.42:8128',
                'https':'socks5h://exmpl:abcd1234@43.156.235.42:8128'}
        )

    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"


    df = pd.read_csv(data_file, usecols=df_cols_to_use)
    print(f"df types: {df.dtypes}")
    print(f"df head: {df.head()}")
    remapping_new_tts = {
        '深扒…的小心机': '深扒...的小心机',
        '后悔没早点…': '后悔没早点...',
        '穿出门被问了很多次的…，出门没输过': '...出门没输过',
        '任何人错过这个…我都会伤心的OK?': '任何人错过我都会伤心的...',
        '我怎么没早点发现！': '我怎么没早点发现...',
        '这是谁发明的!': '这是谁发明的...',
        '只有1%知道的…': '只有1%知道的...',
        '当你…的时候人家已经…': '当你...的时候人家已经...',
        '…是怎么狠狠奖励我的': '...是怎么狠狠奖励我的',
        '请大数据把这篇推荐给…': '请大数据把这篇推荐给...',
        '为它疯狂/心动一万次': '为它疯狂/心动一万次',
        '让人一眼就沦陷': '...让人一眼就沦陷',
        '一见钟情的': '一见钟情的...',
        '好看到犯规': '...好看到犯规',
        '可盐可甜': '...可盐可甜...',
        '你的DNA动了吗': '...你的DNA动了吗...',
        '近期风很大的': '近期风很大的...',
        '温柔buff叠满的': '...温柔buff叠满...',
        '氛围感叠满的': '氛围感叠满的...',
        '被惊艳到了': '被惊艳到了...',
        '快来看看你做对了没有': '快来看看你做对了没有...',
        '被夸爆的…在这里': '被夸爆的...',
    }

    df['title_template_name'] = df.title_template_name.apply(
        lambda x: remapping_new_tts[x] if x in remapping_new_tts else x)


    data = Dataset.from_pandas(df).train_test_split(
            test_size=100, shuffle=True, seed=42
        )
    cols = data['train'].column_names
    train_data = data["test"].shuffle().map(partial(generate_and_tokenize_prompt, tokenizer=tokenizer, max_seq_length=1000),
                                            remove_columns = [x for x in cols if x not in ['title', 'content']])


    print(f"\n\n1. going through training dataset, asserting everything")
    for i, batch in enumerate(train_data):
        #print(f"{i}: {batch}")
        decoded = tokenizer.decode(batch['input_ids'], skip_special_tokens=True)
        decoded_labels = tokenizer.decode([x if x!=-100 else 0 for x in batch['labels']], skip_special_tokens=True)
        #print(f"decoded: {decoded}")
        #print(f"decoded label: {decoded_labels}")

        assert len(batch['input_ids']) == len(batch['labels'])
        assert len(batch['input_ids']) == len(batch['attention_mask'])
        assert len(batch['input_ids']) <= 1000
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






    #print(testml[10])
    #print(testml.tokenizer.decode(testml[10]['input_ids']))
    #print(testml.tokenizer.decode(testml[23]['input_ids']))
    #print(testml.tokenizer.decode(testml[17]['input_ids']))
    #print(len(testml))
    #print(f"max len:{testml.max_seq_length}")

    print(f"\n\n2. trying different input len and asserting cut off correctly")

    test_content = '将数据转换成模型训练的输入将数据转换成模型训练的输入将数据转换成模型训练的输入'
    test_title = '将数据转换成模型训练的输入'

    for i in range(200, 40, -1):
        print(f"\nfor max len: {i}")

        tt = tt_df.sample(2).iloc[0]['title_template_name']


        res = partial(generate_and_tokenize_prompt, tokenizer=tokenizer, max_seq_length=i)({
                             "data_type":"redbook_content_title",
                             "title_template_name":tt,
                             "source_category":"newrank_healthcare",
                             'content':test_content,
                             'title':test_title})
        deres = tokenizer.decode(res['input_ids'], skip_special_tokens=True)

        res_no_max = partial(generate_and_tokenize_prompt, tokenizer=tokenizer, max_seq_length=1000000)({
                             "data_type":"redbook_content_title",
                             "title_template_name":tt,
                             "source_category":"newrank_healthcare",
                             'content':test_content,
                             'title':test_title})

        res_min_content_no_max = partial(generate_and_tokenize_prompt, tokenizer=tokenizer, max_seq_length=1000000)({
                             "data_type":"redbook_content_title",
                             "title_template_name":tt,
                             "source_category":"newrank_healthcare",
                             'content':'我',
                             'title':test_title})

        if test_title in deres and test_content in deres:
            assert len(res['input_ids']) <= i, f"deres:{deres}"
            assert len(res['input_ids']) == len(res_no_max['input_ids'])
        elif test_title in deres and test_content not in deres:
            assert len(res['input_ids']) <= i, f"res: {res}, i:{i}, deres:{deres}"
            assert len(res_no_max['input_ids']) > i, f"deres:{deres}"
            assert len(res_min_content_no_max['input_ids']) < i, f"deres:{deres}"
        elif test_title not in deres and test_content not in deres:
            assert len(res['input_ids']) <= i, f"deres:{deres}"
            assert len(res_no_max['input_ids']) > i, f"deres:{deres}"
            assert len(res_min_content_no_max['input_ids']) > i, f"deres:{deres}"
        elif test_title not in deres and test_content in deres:
            assert False

        assert res['input_ids'][0] == tokenizer.bos_token_id, f"deres:{deres}"
        assert res['input_ids'][-1] == tokenizer.eos_token_id,  f"deres:{deres}"
        assert tokenizer.bos_token_id not in res['input_ids'][1:],  f"deres:{deres}"
        assert tokenizer.eos_token_id not in res['input_ids'][:-1], f"deres:{deres}"



    from tqdm import tqdm

    print(f"3. go over whole train data to make sure no exception...")

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