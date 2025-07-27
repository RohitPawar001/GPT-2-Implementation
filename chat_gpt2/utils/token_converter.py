import torch
import tiktoken

def get_tokenizer():
    tokenizer = tiktoken.get_encoding("gpt2")
    return tokenizer


def text_to_token_ids(text, tokenizer):
    token = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    token_tensor = torch.tensor(token).unsqueeze(0)
    return token_tensor


def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())