import copy
import math
import os
import warnings
import transformers


def adjust_learning_rate(iter, configs):
    """Decay the learning rate with half-cycle cosine after warmup"""
    warmup_iters = configs['warmup_iters']
    total_iters = configs['iters']
    min_lr_scale = configs['min_lr_scale']

    if iter < configs['warmup_iters']:
        lr_scaler = 1.0 * iter / warmup_iters
    else:
        lr_scaler = min_lr_scale + (1.0 - min_lr_scale) * 0.5 * \
            (1.0 + math.cos(math.pi * (iter - warmup_iters) / (total_iters - warmup_iters)))

    return lr_scaler


def default_tokenizer_config(tokenizer):
    tokenizer_cfg = {
        'BertTokenizer': {
            'type': 'BertTokenizer',
            'pretrained_model_name_or_path': 'bert-base-uncased'
        },
        'BertTokenizerFast': {
            'type': 'BertTokenizerFast',
            'pretrained_model_name_or_path': 'bert-base-uncased'
        },
        'CLIPTokenizer': {
            'type': 'CLIPTokenizer',
            'pretrained_model_name_or_path': "openai/clip-vit-base-patch32"
        },
        'MPT1b': {
            'type': 'AutoTokenizer',
            'pretrained_model_name_or_path': '/mnt/bn/robotics/lxh/mpt-1b-redpajama-200b'
        },
        'MPT1bIFTTokenizer': {
            'type': 'AutoTokenizer',
            'pretrained_model_name_or_path': '/mnt/bn/robotics/lxh/mpt-1b-redpajama-200b-dolly'
        },
        'MPT3bTokenizer': {
            'type': 'AutoTokenizer',
            'pretrained_model_name_or_path': '/mnt/bn/robotics/lxh/RedPajama-INCITE-Base-3B-v1'
        },
        'MPT3bIFTTokenizer': {
            'type': 'AutoTokenizer',
            'pretrained_model_name_or_path': '/mnt/bn/robotics/lxh/RedPajama-INCITE-Instruct-3B-v1'
        },
        'MPT7bTokenizer': {
            'type': 'AutoTokenizer',
            'pretrained_model_name_or_path': '/mnt/bn/robotics/lxh/mpt-7b'
        },
        'LLaMA7bTokenizer': {
            'type': 'AutoTokenizer',
            'pretrained_model_name_or_path': '/mnt/bn/robotics/lxh/.cache/llama-7b-hf-jxu124'
        },
    }

    # synonyms
    tokenizer_cfg['bert'] = tokenizer_cfg['BertTokenizer']
    tokenizer_cfg['clip'] = tokenizer_cfg['CLIPTokenizer']

    tokenizer_cfg['mpt_1b'] = tokenizer_cfg['MPT1bTokenizer']
    tokenizer_cfg['mpt_1b_ift'] = tokenizer_cfg['MPT1bIFTTokenizer']

    tokenizer_cfg['mpt_3b'] = tokenizer_cfg['MPT3bTokenizer']
    tokenizer_cfg['mpt_3b_ift'] = tokenizer_cfg['MPT3bIFTTokenizer']
    
    tokenizer_cfg['mpt_7b'] = tokenizer_cfg['MPT7bTokenizer']

    tokenizer_cfg['llama_7b'] = tokenizer_cfg['LLaMA7bTokenizer']

    return tokenizer_cfg[tokenizer]


def build_tokenizer(tokenizer_config, new_tokens=None):
    if isinstance(tokenizer_config, str):
        tokenizer_config = default_tokenizer_config(tokenizer_config)
    else:
        assert isinstance(tokenizer_config, dict) and 'type' in tokenizer_config

    # prevent changes to the original dict
    tokenizer_config = copy.deepcopy(tokenizer_config)
    tokenizer_type = tokenizer_config.pop('type')
    # tokenizer_path = os.path.join('pretrained/tokenizers', tokenizer_type)
    tokenizer_path = tokenizer_config.pop('pretrained_model_name_or_path')

    try:
        tokenizer = getattr(transformers, tokenizer_type).from_pretrained(tokenizer_path)

    except:
        warnings.warn(f"Tokenizer initialization failed with the given path {tokenizer_path}. "
                      f"The tokenizer will be initialized from scratch with default settings. "
                      f"Please refer to {__file__} for details.")

        tokenizer = getattr(transformers, tokenizer_type).from_pretrained(**tokenizer_config)

        # try:
        #     if os.path.exists(tokenizer_path):
        #         import shutil
        #         shutil.rmtree(tokenizer_path)
        #     os.makedirs(tokenizer_path, exist_ok=True)

        #     tokenizer.save_pretrained(tokenizer_path)
        #     print(f'Tokenizer saved to {tokenizer_path}.')

        # except:
        #     print(f'Saving tokenizer to {tokenizer_path} failed...')

    if new_tokens is not None:
        new_tokens = {'additional_special_tokens': new_tokens}
        tokenizer.add_special_tokens(new_tokens)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.add_special_tokens({"pad_token": "<PAD>"})

    assert tokenizer.vocab_size < 2 ** 16


def get_target_modal_tokens(tok_seq, tok_mask):
    index = tok_mask.nonzero(as_tuple=True)
    return tok_seq[index]


def preprocess_text_flamingo(sample, tokenizer):
    tokenizer.padding_side = "right"
    sample = [
        # (f"{s.strip()}{tokenizer.eos_token}")
        # for s in sample
        (f"<image>{s.strip()}<|endofchunk|>{tokenizer.eos_token}") for s in sample
    ]
    text = tokenizer(
        sample,
        max_length=32,
        padding="longest",
        truncation="only_first",
        return_tensors="pt",
    )
    return text["input_ids"], text["attention_mask"]


def build_text_function(tokenizer, llm_type):
    if llm_type == "flamingo":
        text_fn = preprocess_text_flamingo
    else:
        raise NotImplementedError

# def get_modal_tokens(tok_seq, tok_mask_dict, modal_name):
#     assert modal_name in tok_mask_dict, f"{modal_name} not in token sequence"
#     return get_target_modal_tokens(tok_seq, tok_mask_dict[modal_name])