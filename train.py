# Single GPU training script using FIM
import os
import torch

import transformers
from accelerate import Accelerator

from mamba_model import MambaLMHeadModel

import lance
import pyarrow as pa

# Params (replace with Arg parser later)
class Args:
    tokenizer_model = "EleutherAI/gpt-neox-20b"
    model_name = 'state-spaces/mamba-1.4b'
    low_cpu_mem_usage = False
    fim_training = True
    truncate_or_pad = True
    fim_prefix_token = "<fim_prefix>"
    fim_middle_token = "<fim_middle_token>"
    fim_suffix_token = "<fim_suffix_token>"
    fim_pad_token = "<fim_pad>"
    pad_factor = 8

# Define Tokenizer and Model
tokenizer = transformers.AutoTokenizer.from_pretrained(Args.tokenizer_model)
tokenizer.pad_token = tokenizer.eos_token

model = MambaLMHeadModel.from_pretrained(
    Args.model_name,
    low_cpu_mem_usage=Args.low_cpu_mem_usage,   
)

# Get the FIM-specific token ids
prefix_tok_id = tokenizer.convert_tokens_to_ids(Args.fim_prefix_token)
middle_tok_id = tokenizer.convert_tokens_to_ids(Args.fim_middle_token)
suffix_tok_id = tokenizer.convert_tokens_to_ids(Args.fim_suffix_token)
pad_tok_id = None

fim_tokens = [prefix_tok_id, middle_tok_id, suffix_tok_id]

# If truncate_or_pad is on, also get pad token id
if Args.truncate_or_pad:
    pad_tok_id = tokenizer.convert_tokens_to_ids(Args.fim_pad_token)
    fim_tokens.append(pad_tok_id)

# Add new tokens and resize model token embeddings according to multivariate normal distribution
tokenizer.add_tokens(fim_tokens)
original_embeddings = model.get_input_embeddings().weight
model.resize_token_embeddings(len(tokenizer), pad_to_mulitple_of=Args.pad_factor)
mean = original_embeddings.mean(dim=0)
n = original_embeddings.size()[0]
sigma = ((original_embeddings - mean).T @ (original_embeddings - mean)) / n
dist = torch.distributions.MultivariateNormal(
    mean,
    covariance_matrix=1e-5*sigma
)
new_token_embeddings = torch.stack(
    tuple((dist.sample() for _ in range(len(fim_tokens)))),
    dim=0
)

# Get updated embedding layer and make a copy of it's weights
embeddings = model.get_input_embeddings()
new_embeddings = embeddings.weight.clone()

# Set the new token' embeddings to the newly sampled embeddings
new_embeddings[-len(fim_tokens):] = new_token_embeddings

# Update the model's embeddings with the new embeddings
embeddings.weight = torch.nn.Parameter(new_embeddings)
model.set_input_embeddings(embeddings)

