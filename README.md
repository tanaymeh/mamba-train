# mamba-train

A single repo with all scripts and utils to train / fine-tune the Mamba model with or without Fill-in-Middle objective (for code infilling).

### Data
Currently, the `train.py` script only supports training from a Lance and a Huggingface dataset. If you are training using a Huggingface dataset, substitute `MambaDataset` with your Huggingface dataset in the `train.py` file.

In order for the training to run when using the aforementioned huggingface dataset, the data needs to be grouped in groups of 'context length'. That is, each sample in the dataset must have 'context length' number of tokens in it. For more information on how to achieve this, see the [`group_texts`](https://github.com/huggingface/transformers/blob/89c64817ce4172bc8bb58c675c445a63f16d0e38/examples/pytorch/language-modeling/run_clm_no_trainer.py#L459-L472) function. 

Once the data is in the right format, call the `apply_fim` function in the training loop, passing in the samples and all the appropriate parameters with it. If you face any problems, please open an issue!

For the Lance dataset, I will be releasing the 5M samples subset of the Codeparrot dataset soon. For more information on how it was made using Lance, see my [article](https://tanaymeh.github.io/blog/2024/02/08/p7.html).

**A note about `MambaSampler`**: I am training the model on the Lance dataset which is one large contiguous array of tokens. In this setting, it is very hard to distinguish between different samples (each with the size of context length) without altering the dataset creation process. We need to have non-overlapping samples so as to not overfit the model.

My workaround for this was making a new sampler that samples `len(dataset) // context_len` number of samples from the dataset, where each of those sample is atleast `context_len` indices apart from each other. This "emulates" them as individual samples with minimal processing overhead.

### Fill-in-Middle
Both the Lance and HF datasets apply Fill-in-Middle transformation on each 'sample' during the training run. FIM training objectives allows the model to infill the code. FIM trained models are the ones used by code-completion tools like Github Copilot. 
In order to learn more about Fill-in-Middle training objective, see the [OpenAI paper](https://arxiv.org/abs/2207.14255).

In order to adjust what percentage of training samples are transformed using FIM, you can adjust the `fim_rate` parameter in both datasets. By default it is set to 0.9, meaning 90% of all samples will be FIM transformed (this is because I am fine-tuning the model instead of pre-training it).

### Training
Before starting the training run, you need to install all the dependencies from the requirements file

```bash
pip install -r requirements.txt
```

Once that is done, start the training run via:

```bash
python train.py
```