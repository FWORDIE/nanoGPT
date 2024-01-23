import time

out_dir = 'out-reddit-fine'
eval_interval = 5
eval_iters = 40
wandb_log = False # feel free to turn on
wandb_project = 'reddit'
wandb_run_name = 'ft-' + str(time.time())

dataset = 'reddit'
init_from = 'gpt2' # this is the largest GPT-2 model

# only save checkpoints if the validation loss improves
always_save_checkpoint = False

# the number of examples per iter:
# 1 batch_size * 32 grad_accum * 1024 tokens = 32,768 tokens/iter
# shakespeare has 301,966 tokens, so 1 epoch ~= 9.2 iters
batch_size = 1
gradient_accumulation_steps = 32
max_iters = 20

# finetune at constant LR
learning_rate = 3e-5
decay_lr = False

device = 'cpu'
compile = False
eval_iters = 20
log_interval = 1
block_size = 64
batch_size = 12
n_layer = 4
n_head = 4
n_embd = 128
max_iters = 100
lr_decay_iters = 2000
dropout = 0.0