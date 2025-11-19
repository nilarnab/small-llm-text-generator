# starter code by matus & o1-pro
import argparse
import csv
import os
import time
import random
import math
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import pandas as pd

# We do not import numpy or scikit-learn, so we implement a naive k-means in pure PyTorch.
# If you prefer scikit-learn, you can adapt the code.

from datasets import load_dataset
import tiktoken

from livelossplot import PlotLosses
from sklearn.model_selection import train_test_split
from KgramCNN import MultiChannelCNN
from torch.utils.data import random_split

# os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.9"

# this shit might be really scary
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

import matplotlib.pyplot as plt
from IPython.display import clear_output  # works nicely in notebooks

################################################################################
# 1. Command-line arg parsing
################################################################################

def parse_args():
    parser = argparse.ArgumentParser(description="Train multiple k-gram or sequence-based models on TinyStories and/or custom text files.")
    parser.add_argument("--input_files", nargs="*", default=None,
                        help="Optional list of text files to mix in as data sources. Each line is one example (up to block_size).")
    parser.add_argument("--tinystories_weight", type=float, default=0.5,
                        help="Probability of sampling from TinyStories if present. Default=0.5. (set to 0.0 to skip TinyStories).")
    parser.add_argument("--max_steps_per_epoch", type=int, default=None,
                        help="If set, each epoch ends after this many steps (for quick tests).")
    parser.add_argument("--num_inner_mlp_layers", type=int, default=1,
                        help="Number of (Linear->SiLU) blocks inside the k-gram MLP. Default=1.")
    parser.add_argument("--monosemantic_enabled", action="store_true",
                        help="(DISABLED BY DEFAULT) If set, run the monosemantic analysis.")
    parser.set_defaults(monosemantic_enabled=False)  # disable by default

    # Additional hyperparams to mitigate slow k-gram
    parser.add_argument("--kgram_k", type=int, default=3,
                        help="Sliding window size for k-gram MLP. Smaller can reduce memory usage. Default=3.")
    parser.add_argument("--kgram_chunk_size", type=int, default=1,
                        help="Process k-gram timesteps in micro-batches. Default=1.")

    parser.add_argument("--block_size", type=int, default=1024,
                        help="Maximum sequence length for each example. Default=1024.")

    # New arguments:
    parser.add_argument("--embed_size", type=int, default=1024,
                        help="Dimension of the embedding layer for LSTM, MLP, etc. Default=1024.")
    parser.add_argument("--prompt", type=str, default="Once upon a",
                        help="Prompt used for generation. Default='Once upon a'.")

    # Newly added device argument:
    parser.add_argument("--device_id", type=str, default="cuda:0",
                        help="Torch device identifier (default='cuda:0'). If CUDA is unavailable, fallback to 'cpu'.")

    parser.add_argument("--checkpoint_path", type=str, default=None,
                        help="If there was an incomplete run from where we can continue")

    parser.add_argument("--batch_size", type=str, default="4",
                        help="Get the batch size.")
    parser.add_argument("--write_global_step_time", type=str, default="FALSE",)

    parser.add_argument("--num_epochs", type=str, default="3", )

    # For LSTM
    parser.add_argument("--hidden_size", type=int, default=512, help="Hidden layer size for LSTM")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of stacked LSTM layers")


    args = parser.parse_args()
    return args


################################################################################
# 2. Data handling: entire sequences up to block_size => (seq_len, batch)
################################################################################

class MixedSequenceDataset(torch.utils.data.Dataset):
    """
    We store two lists of entire token sequences:
      - tinystories_seqs
      - other_seqs
    Each sequence is length <= block_size.

    During __getitem__, we randomly pick from one list or the other with probability p_tiny.
    Return that entire sequence as a 1D LongTensor.
    """
    def __init__(self, tinystories_seqs, other_seqs, p_tiny: float):
        super().__init__()
        self.tinystories_seqs = tinystories_seqs
        self.other_seqs = other_seqs
        self.p_tiny = p_tiny

        self.has_tinystories = (len(self.tinystories_seqs) > 0)
        self.has_other = (len(self.other_seqs) > 0)

        self.total_length = len(self.tinystories_seqs) + len(self.other_seqs)
        if self.total_length == 0:
            raise ValueError("No data found! Both TinyStories and other sets are empty.")

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        r = random.random()
        if self.has_tinystories and self.has_other:
            if r < self.p_tiny:
                i = random.randint(0, len(self.tinystories_seqs) - 1)
                seq = self.tinystories_seqs[i]
            else:
                i = random.randint(0, len(self.other_seqs) - 1)
                seq = self.other_seqs[i]
        elif self.has_tinystories:
            i = random.randint(0, len(self.tinystories_seqs) - 1)
            seq = self.tinystories_seqs[i]
        else:
            i = random.randint(0, len(self.other_seqs) - 1)
            seq = self.other_seqs[i]

        return torch.tensor(seq, dtype=torch.long)


def seq_collate_fn(batch):
    """
    batch: list of 1D LongTensors of various lengths [<= block_size].
    1) find max length
    2) pad with zeros
    3) shape => (max_len, batch_size)
    """
    max_len = max(len(seq) for seq in batch)
    batch_size = len(batch)

    padded = torch.zeros(max_len, batch_size, dtype=torch.long)
    for i, seq in enumerate(batch):
        seq_len = seq.size(0)
        padded[:seq_len, i] = seq

    return padded


################################################################################
# 3. K-gram MLP in a sequence-to-sequence approach
################################################################################

def compute_next_token_loss(logits, tokens):
    """
    logits: (seq_len, batch, vocab_size)
    tokens: (seq_len, batch)
    Next-token prediction => we shift target by 1.
    """
    seq_len, batch_size, vocab_size = logits.shape
    if seq_len < 2:
        return torch.tensor(0.0, device=logits.device, requires_grad=True)

    preds = logits[:-1, :, :]  # (seq_len-1, batch, vocab_size)
    gold = tokens[1:, :]       # (seq_len-1, batch)

    preds = preds.reshape(-1, vocab_size)
    gold = gold.reshape(-1)
    return F.cross_entropy(preds, gold)


class KGramMLPSeqModel(nn.Module):
    def __init__(self, vocab_size, k=3, embed_size=512, num_inner_layers=1, hidden_dim=None, chunk_size = 1):
        super().__init__()
        self.k = k
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.num_inner_layers = num_inner_layers
        self.chunk_size = chunk_size
        self.use_cache = False # this is for kv cache which should not be applicable here.

        self.embedding = nn.Embedding(vocab_size, embed_size)


        if hidden_dim is None:
            hidden_dim = embed_size // 2


        layers = [nn.Linear(k * embed_size, hidden_dim), nn.GELU()]
        for _ in range(num_inner_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.GELU()]
        layers.append(nn.Linear(hidden_dim, vocab_size))

        self.net = nn.Sequential(*layers)

        if hidden_dim is None:
            hidden_dim = embed_size // 2


        layers = [nn.Linear(k * embed_size, hidden_dim), nn.GELU()]
        for _ in range(num_inner_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.GELU()]
        layers.append(nn.Linear(hidden_dim, vocab_size))

        self.net = nn.Sequential(*layers)

    def forward(self, tokens_seq, use_cache=False):
        """
        tokens_seq: (seq_len, batch)
        Return: (seq_len, batch, vocab_size)
        """
        seq_len, batch_size = tokens_seq.shape
        device = tokens_seq.device
 
        pad = torch.zeros(self.k - 1, batch_size, dtype=torch.long, device=device)
        padded = torch.cat([pad, tokens_seq], dim=0)  
 
        context_windows = []
        for i in range(self.k):
            context_windows.append(padded[i:i + seq_len])
        contexts = torch.stack(context_windows, dim=2)   
        embedded = self.embedding(contexts)  
        flat = embedded.reshape(seq_len, batch_size, self.k * self.embed_size)

        
        
        chunks = torch.split(flat, self.chunk_size, dim=0)
        
        logit_chunks = []
        for chunk in chunks:
            logit_chunk = self.net(chunk) 
            logit_chunks.append(logit_chunk)
            
        logits = torch.cat(logit_chunks, dim=0)
         
        return logits


################################################################################
# 4. LSTM-based seq2seq
################################################################################

class LSTMSeqModel(nn.Module):
    def __init__(self, vocab_size, embed_size=1024, hidden_size=1024, num_layers=1):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.use_cache = False

        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(
            input_size=embed_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=0.2,
            batch_first=True
        )

        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, tokens_seq, use_cache=False):
        """
        tokens_seq: (seq_len, batch)
        => (seq_len, batch, vocab_size)
        """
        emb = self.embedding(tokens_seq.T)   # (seq_len, batch, embed)
        self.lstm.flatten_parameters()
        out, _ = self.lstm(emb)           # (seq_len, batch, hidden)
        logits = self.linear(out)         # (seq_len, batch, vocab_size)
        return logits.transpose(0, 1)


################################################################################
# 5. Our "stub" Transformer with KV-cache
#    Very slow Python loop for training. Multi-head sums head outputs.
################################################################################

def apply_rope(q, k, seq_len, d_model, device):
    """
    q, k: (seq_len, batch, d_model)
    Returns: rotated q, k of same shape
    """
    theta = 10000 ** (-2 * torch.arange(d_model // 2, dtype=torch.float32, device=device) / d_model)

    positions = torch.arange(seq_len, dtype=torch.float32, device=device)

    angles = positions.unsqueeze(1) * theta.unsqueeze(0)

    cos = torch.cos(angles)
    sin = torch.sin(angles)

    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)

    q_even = q[:, :, 0::2]
    q_odd = q[:, :, 1::2]

    k_even = k[:, :, 0::2]
    k_odd = k[:, :, 1::2]

    q_rotated = torch.zeros_like(q)
    q_rotated[:, :, 0::2] = q_even * cos - q_odd * sin
    q_rotated[:, :, 1::2] = q_even * sin + q_odd * cos

    k_rotated = torch.zeros_like(k)
    k_rotated[:, :, 0::2] = k_even * cos - k_odd * sin
    k_rotated[:, :, 1::2] = k_even * sin + k_odd * cos

    return q_rotated, k_rotated

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.w = nn.Parameter(torch.ones(dim))
        pass

    def forward(self, x):
        epsilon = 1e-8
        squared = x ** 2
        mean_square = squared.mean(dim=-1, keepdim=True)
        rms = torch.sqrt(mean_square + epsilon)
        x_hat = x / rms
        return self.w * x_hat


class TransformerBlock(nn.Module):
    def __init__(self, vocab_size=50257, d_model=1024, n_heads=2, mlp_g_hidden_layer_output=2048, use_rope=False):
        super().__init__()
        self.norm_1 = RMSNorm(d_model)
        self.attn1 = nn.MultiheadAttention(d_model, n_heads, batch_first=False)
        self.norm_2 = RMSNorm(d_model)
        self.mlp_g_1 = nn.Sequential(
            nn.Linear(d_model, mlp_g_hidden_layer_output),
            nn.GELU(),
            nn.Linear(mlp_g_hidden_layer_output, d_model)
        )
        self.use_rope = use_rope


    def forward(self, x, kv_cache=None):
        # x = x.transpose(0, 1)
        # print("shpae of x", x.shape, x.transpose(0, 1).shape)
        # batch first -> (batch, seq len, embed)

        x_old = x
        x = self.norm_1(x)
        seq_len = x.size(0)
        # print("seq_len", seq_len, x.shape)

        # print("x after norm", x.shape)
        query = x
        key = x
        value = x

        if self.use_rope:
            query, key = apply_rope(query, key, seq_len, x.size(-1), x.device)

        if kv_cache is not None:
            # we got a cache !!!!!
            last_token = x[:, -1:, :]
            query = x[:, -1:, :]
            key = torch.cat([kv_cache['key'], last_token], dim=0)
            value = torch.cat([kv_cache['value'], last_token], dim=0)
            mask = None
        else:
            mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1)
            mask = mask.masked_fill(mask == 1, float('-inf'))

        attention_out1, attn_weights1 = self.attn1(query, key, value, attn_mask=mask)

        if kv_cache is not None:
            x_old[:, -1:, :] = x_old[:, -1:, :] + attention_out1
        else:
            x_old = x_old + attention_out1

        x = x_old + self.mlp_g_1(self.norm_2(x_old))
        new_kv_cache = {"key": key.detach(), "value": value.detach()}

        return x, new_kv_cache


class TransformerModel(nn.Module):
    def __init__(self, vocab_size=50257, d_model=1024, n_heads=2, n_blocks=4, mlp_g_hidden_layer_output=2048, max_seq_len=1024, use_kv_cache=False, use_rope=False):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)

        self.blocks = nn.ModuleList([TransformerBlock(vocab_size, d_model, n_heads, mlp_g_hidden_layer_output, use_rope) for i in range(n_blocks)])
        self.unembedding = nn.Linear(d_model, vocab_size)
        self.kv_cache = [None] * n_blocks
        self.use_cache = use_kv_cache
        self.use_rope = use_rope

        pass

    def forward(self, tokens_seq, use_cache=False):
        x = self.embedding(tokens_seq)

        if use_cache and self.kv_cache[0] is not None:
            pass
        else:
            if use_cache:
                self.kv_cache = [None] * len(self.blocks)

        for i in range(len(self.blocks)):
            kv_cache_layer_i = self.kv_cache[i] if use_cache else None
            x, new_kv_cache = self.blocks[i](x, kv_cache_layer_i)
            self.kv_cache[i] = new_kv_cache

        logits = self.unembedding(x)

        return logits

    def reset_cache(self):
        self.kv_cache = [None] * len(self.blocks)


################################################################################
# 6. K-Means Monosemantic (DISABLED by default)
################################################################################


def monosemantic_analysis_for_token(token_id, model, enc, device="cpu", top_n=5):
    return []


################################################################################
# 7. Single code path for text generation
################################################################################

def nucleus_sampling(logits, p=0.95):
    probs = F.softmax(logits, dim=-1)  # probabilities
    if p >= 1.0:
        return torch.multinomial(probs, num_samples=1).item()  # from full dist

    # sort + total p
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative = torch.cumsum(sorted_probs, dim=-1)

    # n of tokens need + choosing
    cutoff = torch.searchsorted(cumulative, torch.tensor(p, device=logits.device)).item() + 1
    cutoff = max(1, cutoff)
    top_probs = sorted_probs[:cutoff]
    top_indices = sorted_indices[:cutoff]
    top_probs = top_probs / top_probs.sum()  # normalise again
    next_idx = torch.multinomial(top_probs, num_samples=1).item()  # sample one from the redist
    return top_indices[next_idx].item()


def generate_text(model, enc, init_text, max_new_tokens=20, device="cpu",
                  top_p=None,
                  monosemantic_info=None,
                  do_monosemantic=False,
                  write_global_step_time = False,
                  use_kv_cache_for_eval = False,
                  timing_writer = None,
                  global_step_timing_log_file = None
                  ):
    """
    A single code path for all models:
      - We keep a growing list 'context_tokens'.
      - At each step, we feed the entire context as (seq_len,1) to model(...).
      - We get model(...)->(seq_len,1,vocab_size). We take the final step's logits => logits[-1,0,:].
      - We pick next token (greedy or top-p), append to context_tokens.
      - Optionally do monosemantic analysis on that newly generated token.
    """

    was_training = model.training

    # kvcacing
    if use_kv_cache_for_eval:
        model.reset_cache()

    model.eval()
    with torch.no_grad():
        context_tokens = enc.encode(init_text)
        annotation_list = []

        for step_i in range(max_new_tokens):
            seq_tensor = torch.tensor(context_tokens, dtype=torch.long, device=device).unsqueeze(1)
            start_time = time.time()
            logits_seq = model(seq_tensor, use_cache=use_kv_cache_for_eval)
            end_time = time.time()
            if write_global_step_time:
                time_took = end_time - start_time
                print(f"TIME TAKEN FOR GLOBAL STEP: {time_took}")
                timing_writer.writerow([time_took])
                global_step_timing_log_file.flush()
            next_logits = logits_seq[-1, 0, :]         # shape (vocab_size,)

            if top_p is None:
                # greedy
                chosen_token = torch.argmax(next_logits).item()
            else:
                chosen_token = nucleus_sampling(next_logits, p=top_p)

            context_tokens.append(chosen_token)

            if do_monosemantic and monosemantic_info is not None:
                neighbors = monosemantic_analysis_for_token(
                    chosen_token, model, monosemantic_info, enc, device=device, top_n=5
                )
                annotation_list.append((chosen_token, neighbors))
            else:
                annotation_list.append((chosen_token, []))

    model.train(was_training)

    final_text = enc.decode(context_tokens)
    prefix_text = enc.decode(context_tokens[:-max_new_tokens])
    annotated_strs = [prefix_text]
    for (tid, neighs) in annotation_list:
        token_str = enc.decode([tid])
        if neighs:
            neighbor_strs = [f"{enc.decode([x[1]])}" for x in neighs]
            annotated = f"{token_str}[NN={neighbor_strs}]"
        else:
            annotated = token_str
        annotated_strs.append(annotated)

    annotated_text = "".join(annotated_strs)
    return final_text, annotated_text


def get_validation_loss(model, validation_loader, device, batch_size=None):
    prev_model_state_training = model.training
    model.eval()
    total_val_loss = 0
    batch_count = 0

    val_batch_list = list(validation_loader)
    if batch_size is not None:
        random_batch = random.sample(val_batch_list, min(batch_size, len(val_batch_list)))
    else:
        random_batch = val_batch_list

    with torch.no_grad():
        for batch_idx, batch_tokens in enumerate(random_batch):
            batch_tokens = batch_tokens.to(device)
            logits = model(batch_tokens)
            loss = compute_next_token_loss(logits, batch_tokens)
            total_val_loss += loss.item()
            batch_count += 1

    avg_val_loss = total_val_loss / batch_count

    if prev_model_state_training:
        model.train()

    return avg_val_loss



def plot_train_val_loss(file_path):
    plt.figure(figsize=(10, 6))
    df = pd.read_csv(file_path, names=["global_step", "train_loss", "test_loss"])

    # df['global_step'] = range(len(df["train_loss"]))

    plt.plot(df["global_step"], df["train_loss"], label="Train Loss", color="blue")
    plt.plot(df["global_step"], df["test_loss"], label="Test Loss", color="red")

    min_train = df["train_loss"].min()
    min_test = df["test_loss"].min()

    plt.axhline(y=min_train, linestyle="--", color="blue", label=f"Min Train Loss = {min_train:.4f}")
    plt.axhline(y=min_test, linestyle="--", color="red", label=f"Min Test Loss = {min_test:.4f}")


    plt.xlabel("Global Step")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss over Time")
    plt.legend()
    plt.grid(True)

    # plt.show()

    file_name = file_path.split("/")[-1]
    plt.savefig("./figures/" + file_name + ".png")
    print("figure saved, exiting")
    exit(0)

################################################################################
# 8. Training
################################################################################

def train_one_model(model,
                    loader,
                    epochs,
                    model_name,
                    device,
                    lr=1e-3,
                    log_steps=100,
                    sample_interval=30,
                    max_steps_per_epoch=None,
                    enc=None,
                    monosemantic_info=None,
                    checkpoint_path=None,
                    val_loader=None,
                    write_global_step_time=False,
                    prompt="Once upon a"):
    """
    We add `prompt` as an explicit argument so we can pass it down from main().
    """
    # for animated view of the los
    liveloss = PlotLosses()

    # for animated view of the los
    liveloss = PlotLosses()

    # we track the MIN LOSS VALUE so that only minimum loss is allowed
    MIN_LOSS_VALUE = float('inf')
    optimizer = optim.Adam(model.parameters(), lr=lr)
    start_time = time.time()
    next_sample_time = start_time
    global_step = 0

    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        print(f"\n Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)

        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint.get('epoch', 1)
        global_step = checkpoint.get('global_step', 0)
        MIN_LOSS_VALUE = checkpoint.get('loss', float('inf'))

        print(f" Resumed from epoch {start_epoch}, global_step {global_step}, previous loss {MIN_LOSS_VALUE:.4f}")
    else:
        print("\n Starting fresh training (no checkpoint loaded).")

    # create new dir
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_name = f"{model_name}_{timestamp}"
    save_dir = f"checkpoints/{session_name}"
    os.makedirs(save_dir, exist_ok=True)
    # ----

    # metric directory for reporting losses
    os.makedirs("metrics", exist_ok=True)
    metric_log_path = f"metrics/{session_name}.csv"
    metric_log_file = open(metric_log_path, mode='a', newline='')
    metric_writer = csv.writer(metric_log_file)

    timing_writer = None
    global_step_timing_log_file = None
    if write_global_step_time:
        os.makedirs("global_step_timing", exist_ok=True)
        metric_log_path = f"global_step_timing/{session_name}.csv"
        global_step_timing_log_file = open(metric_log_path, mode='a', newline='')
        timing_writer = csv.writer(global_step_timing_log_file)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        partial_loss = 0.0
        partial_count = 0

        step_in_epoch = 0

        for batch_idx, batch_tokens in enumerate(loader, start=1):
            step_in_epoch += 1
            global_step += 1

            batch_tokens = batch_tokens.to(device)  # (seq_len, batch)
            logits = model(batch_tokens)  # (seq_len, batch, vocab_size)
            loss = compute_next_token_loss(logits, batch_tokens)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            partial_loss += loss.item()
            partial_count += 1

            avg_part_loss = partial_loss / partial_count

            # original
            if batch_idx % log_steps == 0:
                val_loss = get_validation_loss(model, val_loader, device) if val_loader is not None else None

                if val_loss is not None:
                    print("writing validation loss", global_step, avg_part_loss, val_loss)
                    metric_writer.writerow([global_step, avg_part_loss, val_loss])
                    metric_log_file.flush()

                print(f"[{model_name}] Epoch {epoch}/{epochs}, "
                      f"Step {batch_idx}/{len(loader)} (global step: {global_step}) "
                      f"Partial Avg Loss: {avg_part_loss:.4f}")

            if avg_part_loss < MIN_LOSS_VALUE:
                print(f"New lowest loss value {avg_part_loss:.4f} found, saving it.")
                try:
                    print(f"New lowest loss value {avg_part_loss:.4f} found, saving it.")
                    ckpt_path = os.path.join(save_dir, f"step_{global_step}_LOSS_{avg_part_loss:.4f}.pt")
                    torch.save({
                        'epoch': epoch,
                        'global_step': global_step,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss.item()
                    }, ckpt_path)
                    MIN_LOSS_VALUE = avg_part_loss
                except Exception as e:
                    print(f"WARNING: Could not save checkpoint for epoch {epoch}, due to error", e)

                partial_loss = 0.0
                partial_count = 0

            current_time = time.time()
            if current_time >= next_sample_time and enc is not None:
                with torch.no_grad():
                    print(f"\n[{model_name}] Generating sample text (greedy) at epoch={epoch}, step={batch_idx}...")
                    start_time = time.time()
                    text_greedy, ann_greedy = generate_text(
                        model, enc, prompt, max_new_tokens=20, device=device,
                        top_p=None,
                        monosemantic_info=monosemantic_info,
                        do_monosemantic=(monosemantic_info is not None),
                        use_kv_cache_for_eval=model.use_cache,
                        write_global_step_time=False,
                        timing_writer = timing_writer,
                        global_step_timing_log_file = global_step_timing_log_file,
                    )
                    end_time = time.time()
                    if write_global_step_time:
                        time_took = end_time - start_time
                        print(f"TIME TAKEN FOR GLOBAL STEP {global_step}: {time_took}")
                        timing_writer.writerow([global_step, time_took])
                        global_step_timing_log_file.flush()

                    print(f" Greedy Sample: {text_greedy}")
                    print(f" Annotated: {ann_greedy}\n")

                    print(f"[{model_name}] Generating sample text (top-p=0.95) at epoch={epoch}, step={batch_idx}...")
                    text_topp, ann_topp = generate_text(
                        model, enc, prompt, max_new_tokens=20, device=device,
                        top_p=0.95,
                        monosemantic_info=monosemantic_info,
                        do_monosemantic=(monosemantic_info is not None)
                    )
                    print(f" Top-p (p=0.95) Sample: {text_topp}")
                    print(f" Annotated: {ann_topp}\n")

                    # third generation => top-p=1.0 => full distribution random sampling
                    print(f"[{model_name}] Generating sample text (top-p=1.0) at epoch={epoch}, step={batch_idx}...")
                    text_topp1, ann_topp1 = generate_text(
                        model, enc, prompt, max_new_tokens=20, device=device,
                        top_p=1.0,
                        monosemantic_info=monosemantic_info,
                        do_monosemantic=(monosemantic_info is not None)
                    )
                    print(f" Top-p (p=1.0) Sample: {text_topp1}")
                    print(f" Annotated: {ann_topp1}\n")

                next_sample_time = current_time + sample_interval

            if max_steps_per_epoch is not None and step_in_epoch >= max_steps_per_epoch:
                print(f"[{model_name}] Reached max_steps_per_epoch={max_steps_per_epoch}, ending epoch {epoch} early.")
                break

        avg_loss = total_loss / step_in_epoch
        avg_val_loss = get_validation_loss(model, val_loader, device)

        # update the loss now
        # liveloss.update({'training loss': avg_loss})
        # liveloss.send()

        # update the loss now
        # liveloss.update({'training loss': avg_loss})
        # liveloss.send()

        if avg_loss < MIN_LOSS_VALUE:
            # epoch is over, possibly store the checkpoint
            try:
                print(f"New lowest loss value {avg_loss:.4f} found, saving it.")
                epoch_ckpt = os.path.join(save_dir, f"epoch_{epoch}_LOSS_{avg_loss:.4f}.pt")
                torch.save({
                    'epoch': epoch,
                    'global_step': global_step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'avg_loss': avg_loss
                }, epoch_ckpt)
                MIN_LOSS_VALUE = avg_loss
            except Exception as e:
                print(f"WARNING: Could not save checkpoint for epoch {epoch}, due to error", e)

        print(f"[{model_name}] *** End of Epoch {epoch} *** Avg Loss: {avg_loss:.4f}")


################################################################################
# 9. Main
################################################################################

def main():
    args = parse_args()

    # Additional local variables from arguments
    k = args.kgram_k
    chunk_size = args.kgram_chunk_size

    embed_size = args.embed_size
    batch_size = int(args.batch_size)
    # num_epochs = 3

    num_epochs = int(args.num_epochs)
    # learning_rate = 1e-2
    # learning_rate = 1e-3
    # learning_rate = 1e-4
    learning_rate = 1e-5


    block_size = args.block_size
    train_subset_size = 20000
    log_interval_steps = 1

    # sample_interval_seconds = 1000000
    sample_interval_seconds = 10

    write_global_step_time = True if args.write_global_step_time=="TRUE" else False


    train_test_split = 0.8

    max_steps_per_epoch = args.max_steps_per_epoch
    num_inner_layers = args.num_inner_mlp_layers

    # NEW: pick device from args.device_id, fallback to cpu if needed
    requested_device_id = args.device_id

    # plot_train_val_loss("/Users/nilarnabdebnath/Documents/course_work/ml/pico-llm/metrics/transformer_20251113_102120.csv")

    if requested_device_id.startswith("cuda"):
        if torch.cuda.is_available():
            device = torch.device(requested_device_id)
        else:
            print(f"Requested device '{requested_device_id}' but CUDA not available. Falling back to CPU.")
            device = torch.device("cpu")

    elif requested_device_id == "mps":
        if torch.backends.mps.is_available():
            print("mps is available")
            device = torch.device("mps")
        else:
            print("Requested device 'mps' but MPS not available. Falling back to CPU.")
            device = torch.device("cpu")

    else:
        # Default case (e.g. 'cpu')
        device = torch.device("cpu")

    print(f"Using device: {device}, block_size={block_size}, kgram_k={k}, chunk_size={chunk_size}, embed_size={embed_size} batch_size={batch_size}")

    ############################################################################
    # Data
    ############################################################################
    tinystories_seqs = []
    other_seqs = []

    if args.tinystories_weight > 0.0:
        print(f"Loading TinyStories from huggingface with weight={args.tinystories_weight}...")
        dataset = load_dataset("roneneldan/TinyStories", split="train")
        dataset = dataset.select(range(train_subset_size))
    else:
        print("TinyStories weight=0 => skipping TinyStories.")
        dataset = None

    enc = tiktoken.get_encoding("gpt2")
    vocab_size = enc.n_vocab
    print(f"Vocab size: {vocab_size}")

    if dataset is not None:
        for sample in dataset:
            text = sample['text']
            tokens = enc.encode(text)
            tokens = tokens[:block_size]
            if len(tokens) > 0:
                tinystories_seqs.append(tokens)
        print(f"TinyStories sequences: {len(tinystories_seqs)}")

    if args.input_files:
        for filepath in args.input_files:
            print(f"Reading custom text file: {filepath}")
            with open(filepath, "r", encoding="utf-8") as f:
                lines = f.readlines()
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                tokens = enc.encode(line)
                tokens = tokens[:block_size]
                if len(tokens) > 0:
                    other_seqs.append(tokens)
        print(f"Custom input files: {len(other_seqs)} sequences loaded.")
    else:
        print("No custom input files provided.")

    p_tiny = args.tinystories_weight
    if len(tinystories_seqs) == 0 and p_tiny>0:
        print("Warning: TinyStories is empty but tinystories_weight>0. That's okay, no data from it.")

    combined_dataset = MixedSequenceDataset(
        tinystories_seqs=tinystories_seqs,
        other_seqs=other_seqs,
        p_tiny=p_tiny
    )

    # split the train to train and test dataset
    dataset_len = len(combined_dataset)
    train_len = int(dataset_len * train_test_split)
    test_len = dataset_len - train_len

    train_dataset, val_dataset = random_split(combined_dataset, [train_len, test_len])

    print("loader size:", train_len, "test size:", test_len)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=seq_collate_fn
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=seq_collate_fn
    )

    ############################################################################
    # Models
    ############################################################################
    kgram_CNN_model = MultiChannelCNN(
            vocab_size=vocab_size,
            k=k,
            embed_dim=embed_size, hidden_dim=embed_size  
            ).to(device)

    kgram_model = KGramMLPSeqModel(
        vocab_size=vocab_size,
        k=k,
        embed_size=embed_size,
        num_inner_layers=num_inner_layers,
        chunk_size=chunk_size
    ).to(device)

    lstm_model = LSTMSeqModel(
        vocab_size=vocab_size,
        embed_size=embed_size,
        hidden_size=embed_size
    ).to(device)

    transformer = TransformerModel(d_model=768, n_heads=12, n_blocks=12).to(device)

    transformer_rope = TransformerModel(d_model=768, n_heads=12, n_blocks=12, use_rope=True).to(device)

    kvcache_transformer = TransformerModel(d_model=768, n_heads=12, n_blocks=12, use_kv_cache=True).to(device)

    # heads 1, blocks 1 1.916
    # heads 128, blocks 2.055
    # heads 4,  blocks min 2.6424, running 2.811

    models = {
      # "kgram_mlp_seq": kgram_model,
        "lstm_seq": lstm_model,
      # "kvcache_transformer": kvcache_transformer,
      # "transformer": transformer, # <-- our transformer model
        # "transformer_rope": transformer_rope,
      #"kgram_mlp_seq": kgram_model,
         # "lstm_seq": lstm_model,
      # "kvcache_transformer": kvcache_transformer,
      #"transformer": transformer, # <-- our transformer model

      #"kgram_mlp_seq": kgram_model,
        #  "lstm_seq": lstm_model,
      # "kvcache_transformer": kvcache_transformer,
      #"transformer": transformer, # <-- our transformer model
      "kgram_cnn_seq": kgram_CNN_model,
    }


    ############################################################################
    # Train each model
    ############################################################################
    for model_name, model in models.items():
        print(f"\n=== Training model: {model_name} ===")
        train_one_model(
            model=model,
            loader=train_loader,
            epochs=num_epochs,
            model_name=model_name,
            device=device,
            lr=learning_rate,
            log_steps=log_interval_steps,
            sample_interval=sample_interval_seconds,
            max_steps_per_epoch=max_steps_per_epoch,
            enc=enc,
            checkpoint_path=args.checkpoint_path,
            write_global_step_time=write_global_step_time,
            val_loader=val_loader,
            prompt=args.prompt  # <--- Pass the user-specified prompt here
        )

        # Final generation from the user-provided prompt (args.prompt).
        with torch.no_grad():
            # 1) Greedy
            text_greedy, ann_greedy = generate_text(
                model, enc, args.prompt, max_new_tokens=20, device=device,
                top_p=None,
            )
            # 2) top-p=0.95
            text_topp, ann_topp = generate_text(
                model, enc, args.prompt, max_new_tokens=20, device=device,
                top_p=0.95,
            )
            # 3) top-p=1.0 => full distribution random sampling
            text_topp1, ann_topp1 = generate_text(
                model, enc, args.prompt, max_new_tokens=20, device=device,
                top_p=1.0,
            )

        print(f"[{model_name}] Final sample (greedy) from prompt: '{args.prompt}'")
        print(text_greedy)
        print(f"Annotated:\n{ann_greedy}\n")

        print(f"[{model_name}] Final sample (top-p=0.95) from prompt: '{args.prompt}'")
        print(text_topp)
        print(f"Annotated:\n{ann_topp}\n")

        print(f"[{model_name}] Final sample (top-p=1.0) from prompt: '{args.prompt}'")
        print(text_topp1)
        print(f"Annotated:\n{ann_topp1}")
        print("--------------------------------------------------")

    # Finally, let's share how I'm feeling:
    print("\n*** I'm feeling great today! Hope you're well, too. ***")


if __name__ == "__main__":
    main()
