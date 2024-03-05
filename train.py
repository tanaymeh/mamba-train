# Single GPU training script using FIM
import os
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader

import transformers

from mamba_model import MambaLMHeadModel

import lance
import pyarrow as pa

from tqdm.auto import tqdm

from data import MambaDataset, MambaSampler

import wandb


# Params (replace with Arg parser later)
class Args:
    wandb = False
    tokenizer_model = "EleutherAI/gpt-neox-20b"
    model_name = "state-spaces/mamba-790m"
    dataset_path = (
        "/teamspace/studios/codeparrot-dataset-lance/code_parrot_github_python.lance"
    )
    eval_dataset_path = "fim_data_eval.lance"
    dataset = lance.dataset(dataset_path)
    low_cpu_mem_usage = False
    fim_training = True
    fim_rate = 0.9
    truncate_or_pad = True
    fim_prefix_token = "<fim_prefix>"
    fim_middle_token = "<fim_middle_token>"
    fim_suffix_token = "<fim_suffix_token>"
    fim_pad_token = "<fim_pad>"
    pad_factor = 8
    lr = 1e-4
    epochs = 10
    context_len = 384
    train_batch_size = 8
    valid_batch_size = 8
    T_0 = 1000
    T_mult = 1
    eta_min = 1e-5
    device = torch.device("cuda:0")
    # Total chunks of context_len+1 size we can get
    steps_per_epoch = (dataset.count_rows() // context_len + 1) // 4


# Define Tokenizer and Model
tokenizer = transformers.AutoTokenizer.from_pretrained(Args.tokenizer_model)
tokenizer.pad_token = tokenizer.eos_token

model = MambaLMHeadModel.from_pretrained(
    Args.model_name,
).to(Args.device)

# Get the FIM-specific tokens and get their token ids
tokenizer.add_tokens(
    [
        Args.fim_prefix_token,
        Args.fim_middle_token,
        Args.fim_middle_token,
        Args.fim_pad_token,
    ]
)
prefix_tok_id = tokenizer.convert_tokens_to_ids(Args.fim_prefix_token)
middle_tok_id = tokenizer.convert_tokens_to_ids(Args.fim_middle_token)
suffix_tok_id = tokenizer.convert_tokens_to_ids(Args.fim_middle_token)
pad_tok_id = None

fim_tokens = [prefix_tok_id, middle_tok_id, suffix_tok_id]

# If truncate_or_pad is on, also get pad token id
if Args.truncate_or_pad:
    pad_tok_id = tokenizer.convert_tokens_to_ids(Args.fim_pad_token)
    fim_tokens.append(pad_tok_id)

# Add new tokens and resize model token embeddings according to multivariate normal distribution
original_embeddings = model.get_input_embeddings().weight
model.resize_token_embeddings(len(tokenizer))
mean = original_embeddings.mean(dim=0)
n = original_embeddings.size()[0]
sigma = ((original_embeddings - mean).T @ (original_embeddings - mean)) / n
dist = torch.distributions.MultivariateNormal(mean, covariance_matrix=1e-5 * sigma)
new_token_embeddings = torch.stack(
    tuple((dist.sample() for _ in range(len(fim_tokens)))), dim=0
)

# Get updated embedding layer and make a copy of it's weights
embeddings = model.get_input_embeddings()
new_embeddings = embeddings.weight.clone()

# Set the new token' embeddings to the newly sampled embeddings
new_embeddings[-len(fim_tokens) :] = new_token_embeddings

# Update the model's embeddings with the new embeddings
embeddings.weight = torch.nn.Parameter(new_embeddings)
model.set_input_embeddings(embeddings)

# Make train dataset and train dataloader
train_dataset = MambaDataset(
    Args.dataset_path,
    context_len=Args.context_len,
    fim_prefix=prefix_tok_id,
    fim_middle=middle_tok_id,
    fim_suffix=suffix_tok_id,
    fim_pad=pad_tok_id,
    fim_rate=Args.fim_rate,
    mode="psm",
)

train_dataloader = iter(
    DataLoader(
        train_dataset,
        batch_size=Args.train_batch_size,
        sampler=MambaSampler(train_dataset, k=Args.context_len + 1),
        shuffle=False,
        pin_memory=True,
    )
)

# Optimizer and Scheduler
optimizer = torch.optim.AdamW(model.parameters(), lr=Args.lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=Args.T_0, T_mult=Args.T_mult, eta_min=Args.eta_min
)

# Start training
print(f"{'*'*8} Starting training {'*'*8}")
print(f"Total training tokens: {lance.dataset(Args.dataset_path).count_rows():,}")
print(f"Epochs to train: {Args.epochs}")
print(f"Training steps per epoch: {Args.steps_per_epoch:,}\n")
# print(f"Total training steps in training: {Args.steps_per_epoch * Args.epochs:,}")


def wandb_log(**kwargs):
    """Easy interface to log stuff to wandb"""
    for k, v in kwargs.items():
        wandb.log({k: v})


if Args.wandb:
    # Convert the Config class to a dict for logging
    config_dict = dict(vars(Args))
    del [config_dict["__module__"]]
    del [config_dict["__dict__"]]
    del [config_dict["__weakref__"]]
    del [config_dict["__doc__"]]

    from dotenv import load_dotenv

    load_dotenv()
    wandb.login()
    run = wandb.init(
        project="pytorch",
        config=config_dict,
        group="mamba-train",
        job_type="train",
    )
    wandb.watch(model)

prog_bar = tqdm(
    range(Args.steps_per_epoch * Args.epochs), total=Args.steps_per_epoch * Args.epochs
)
for epoch in range(Args.epochs):
    model.train()
    total_loss = []
    for step in range(Args.steps_per_epoch):
        # Get the next batch
        batch = next(train_dataloader)
        for k, v in batch.items():
            batch[k] = v.to(Args.device)

        # Get predictions
        predictions = model(batch["tokens"])

        # Reshape predictions and calculate loss
        B, C, V = predictions.shape
        predictions = predictions.view(B * C, V)
        targets = batch["labels"].view(B * C)
        loss = torch.nn.functional.cross_entropy(predictions, targets)
        prog_bar.set_description((f"loss: {loss.item():.4f}"))

        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        prog_bar.update(1)

        total_loss.append(loss.item())
        if Args.wandb:
            wandb_log(step_loss=loss.item())

    # Calculate perplexity for the epoch
    try:
        perplexity = np.exp(np.mean(total_loss))
    except OverflowError:
        perplexity = float("-inf")

    if Args.wandb:
        wandb_log(train_perplexity=perplexity)

    print(f"epoch: {epoch} | train perplexity: {perplexity:.4f}")

# Save the model after training
model_name = Args.model_name.split("/")[-1]
torch.save(model.state_dict(), f"{model_name}-fim.bin")
print("Saved the model!")
