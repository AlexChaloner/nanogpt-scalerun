# GPT-2 124M architecture for an existing GPT-2-tokenized uint16 dataset.
# This is a training preset, not a calibrated public benchmark result.
out_dir = 'runs/out-gpt2'
dataset = 'openwebtext'
n_layer = 12
n_head = 12
n_embd = 768
block_size = 1024
batch_size = 4
gradient_accumulation_steps = 8
max_iters = 10000
learning_rate = 6e-4
weight_decay = 0.1
decay_lr = True
warmup_iters = 100
lr_decay_iters = max_iters
min_lr = 6e-5
device = 'cuda'
dtype = 'bfloat16'
compile = True
checkpoint_interval = 100
