# sft_llama_minimal.py
"""
Minimal SFT example: read an Excel with QUESTION, ANSWER columns,
fine-tune a pretrained causal LM (LLaMA-like) on Apple MPS.

Usage:
    python sft_llama_minimal.py

Requirements:
    pip install transformers accelerate pandas openpyxl torch
"""

import os
import torch.utils._pytree as _pytree
import random
from datetime import datetime
import pandas as pd
import torch
import csv

if not hasattr(_pytree, 'register_pytree_node'):
    _pytree.register_pytree_node = _pytree._register_pytree_node

from torch.utils.data import DataLoader, random_split, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, logging

logging.set_verbosity_error()  # reduce HF verbosity

# -------------------------
# User settings
# -------------------------
DEVICE = "mps"
# WHICH MODEL WE USE
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
SFT_DATASET_PATH = "/Users/nilarnabdebnath/Documents/course_work/ml/small-llm-text-generator/dataset/generated_logic_sft_no_reasoning_10k.xlsx"  # your xslx
SFT_DATASET_PATH = "/Users/nilarnabdebnath/Documents/course_work/ml/small-llm-text-generator/dataset/Matus_AI_posttraining_dataset.xlsx"
OUTPUT_DIR = "sft_checkpoints_minimal"
EPOCHS = 3
BATCH_SIZE = 2
MAX_LENGTH = 512    # total tokens (prompt + answer). reduce if OOM
LEARNING_RATE = 2e-6
SEED = 42

CHECKPOINT_PATH = None
# CHECKPOINT_PATH = "sft_checkpoints_minimal/best_epoch1_step50_val0.0143.pt"
CHECKPOINT_PATH = "sft_checkpoints_minimal/best_epoch3_step20_val0.7589.pt"
os.makedirs(OUTPUT_DIR, exist_ok=True)
random.seed(SEED)
torch.manual_seed(SEED)

# -------------------------
# Load tokenizer & model
# -------------------------
print("Loading tokenizer and model (this may take a while)...")

if CHECKPOINT_PATH is None:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
else:
    print("loading tokenizer from {}".format(CHECKPOINT_PATH))
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_PATH)

# ensure pad token exists (some LLaMA tokenizers may not define pad_token)
if tokenizer.pad_token_id is None:
    if tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    else:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})


config = AutoConfig.from_pretrained(MODEL_NAME)

if CHECKPOINT_PATH is None:
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, config=config)
else:
    print("loading checkpoint from {}".format(CHECKPOINT_PATH))
    model = AutoModelForCausalLM.from_pretrained(CHECKPOINT_PATH, config=config)

# move to device and use half precision for memory saving
device = torch.device(DEVICE if torch.has_mps else ("cuda" if torch.cuda.is_available() else "cpu"))
model.to(device)
# try:
#     model.half()   # reduce memory; may help on MPS
# except Exception:
#     pass

print("Model and tokenizer ready. Device:", device)

# -------------------------
# Dataset helpers
# -------------------------
PROMPT_TEMPLATE = "User: {question}\nAssistant: {answer}"

def read_dataframe(path):
    df = pd.read_excel(path)
    # ensure columns exist
    if not {"QUESTION", "ANSWER"}.issubset(set(df.columns)):
        raise ValueError("Excel must contain columns: QUESTION, ANSWER")
    # drop rows with empty
    df = df.dropna(subset=["QUESTION", "ANSWER"]).reset_index(drop=True)
    return df

def format_example(question: str, answer: str) -> str:
    # Ensure answer ends cleanly
    answer = str(answer).strip()
    # If user answers sometimes include a prefix like "ANSWER: A", preserve them.
    return PROMPT_TEMPLATE.format(question=question.strip(), answer=answer)

def encode_for_sft(text: str, max_length: int = MAX_LENGTH):
    """
    Return input_ids (torch.LongTensor) and labels (torch.LongTensor).
    labels has -100 for prompt tokens so loss is computed only on assistant text.
    """
    # identify where assistant answer begins for masking
    assistant_marker = "Assistant:"
    idx = text.find(assistant_marker)
    if idx == -1:
        # safeguard: treat whole text as input (no mask)
        prompt_part = ""
        prompt_len = 0
    else:
        prompt_part = text[: idx + len(assistant_marker)]
    # tokenization: get full and prompt token lists (plain python lists)
    full_ids = tokenizer(text, add_special_tokens=True, truncation=True, max_length=max_length)["input_ids"]
    prompt_ids = tokenizer(prompt_part, add_special_tokens=True, truncation=True, max_length=max_length)["input_ids"]

    input_ids = torch.tensor(full_ids, dtype=torch.long)
    labels = input_ids.clone()
    # mask prompt tokens (so they won't contribute to loss)
    prompt_len = len(prompt_ids)
    if prompt_len > 0:
        labels[:prompt_len] = -100
    return input_ids, labels

class SFTDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.examples = []
        for _, row in df.iterrows():
            q = str(row["QUESTION"])
            a = str(row["ANSWER"])
            txt = format_example(q, a)
            self.examples.append(txt)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return encode_for_sft(self.examples[idx])

def collate_fn(batch):
    # batch: list of (input_ids, labels) tensors (variable length)
    input_ids_list = [b[0] for b in batch]
    labels_list = [b[1] for b in batch]
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    input_ids_padded = torch.nn.utils.rnn.pad_sequence(input_ids_list, batch_first=True, padding_value=pad_id)
    labels_padded = torch.nn.utils.rnn.pad_sequence(labels_list, batch_first=True, padding_value=-100)
    return {"input_ids": input_ids_padded, "labels": labels_padded}

# -------------------------
# Load data and dataloaders
# -------------------------
print("Loading dataset from:", SFT_DATASET_PATH)
df = read_dataframe(SFT_DATASET_PATH)
dataset = SFTDataset(df)
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
print(f"Dataset size: {len(dataset)} | train: {len(train_dataset)} | val: {len(val_dataset)}")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

# -------------------------
# Training setup
# -------------------------
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
loss_fn = None  # we will use model(..., labels=...) which returns loss

def evaluate(model, val_loader):
    model.eval()
    total_loss = 0.0
    n = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
            n += 1
    model.train()
    return total_loss / n if n > 0 else float("inf")

# -------------------------
# Training loop (minimal)
# -------------------------
print("Starting training...")
global_step = 0
best_val = float("inf")


def run_manual_test():
    test_prompts = [
        "User: If A is bigger than B and B is bigger than C, who is the tallest?\nAssistant:",
        "User: A is connected to B. B is connected to C. Is A connected to C?\nAssistant:"
    ]

    for p in test_prompts:
        inputs = tokenizer(p, return_tensors="pt").to(device)
        # generate; keep small max_new_tokens to save memory/time
        gen = model.generate(**inputs, max_new_tokens=64, do_sample=True, top_p=0.95, temperature=0.3)
        text = tokenizer.decode(gen[0], skip_special_tokens=False)
        # print only after the "Assistant:" marker
        if "Assistant:" in text:
            print("Q:", p.split("\n")[0])
            print("A:", text.split("Assistant:")[-1].strip())
        else:
            print("Generated:", text)


def get_response(questions):
    """Load checkpoint and generate answers."""
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch

    if CHECKPOINT_PATH is None:
        print("No checkpoint provided, using default checkpoint.")
        exit(1)

    single_q = isinstance(questions, str)
    questions = [questions] if single_q else questions

    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_PATH)
    model = AutoModelForCausalLM.from_pretrained(CHECKPOINT_PATH)
    device = torch.device("mps" if torch.has_mps else "cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    answers = []
    with torch.no_grad():
        for q in questions:
            prompt = f"User: {q}\nAssistant:"
            print("QUESTION:", q)
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            out = model.generate(**inputs, max_new_tokens=128, do_sample=True, top_p=0.5, temperature=0.3)
            text = tokenizer.decode(out[0], skip_special_tokens=True)
            answer = text.split("Assistant:")[-1].strip() if "Assistant:" in text else text.strip()
            print("ANSWER:", answer)
            answers.append(answer)

    exit(0)

# FOR TESTING
# -------------
# Uncomment this code for testing. The checkpoint will be taken from

# get_response([
#     "When will you release the typed notes of lecture 6?",
#     "Hi Professor, when will you release the typed notes of lecture 6?",
#     "When will the assignment be out?",
#     "Do you know where is the Empire State Building?",
#     "Hi prof, Do you know where is the Empire State Building?",
#     "Can you tell me what is the capital of France ?",
#     "Hi Professor, Can you tell me what is the capital of France ?",
#     "If I add 5 and 7, what is the result of it?",
#     "If I add five and seven, what is the result of it?",
#     "If I add 3 and 4, and add this result with 5, what is the result of it?",
#     "If I add three and four, and add this result with five, what is the result of it?",
#     "If B is bigger than A and A is bigger than C, who is the tallest?",
#     "If someone is on fire, then that person dies. Matus is on fire. Does Matus die?",
# ])


os.makedirs(f"metrics_sft_pretrained", exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
session_name = f"transformersft_{timestamp}"
metric_log_path = f"metrics_sft_pretrained/{session_name}.csv"
metric_log_file = open(metric_log_path, mode='a', newline='')
metric_writer = csv.writer(metric_log_file)

for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0.0
    steps = 0
    print(f"--- Epoch {epoch+1}/{EPOCHS} ---")
    for batch in train_loader:
        global_step += 1
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()

        epoch_loss += loss.item()
        steps += 1

        avg = epoch_loss / steps

        if global_step % 1 == 0:
            print(f"Step {global_step} | batch loss {loss.item():.4f} | avg loss {avg:.4f}")

        if global_step % 1 == 0:
            print("Running quick validation...")

            run_manual_test()

            val_loss = evaluate(model, val_loader)
            print(f"  Validation loss: {val_loss:.4f}")

            # Writing metrics
            metric_writer.writerow([global_step, avg, val_loss])
            metric_log_file.flush()

            if val_loss < best_val:
                best_val = val_loss
                save_path = os.path.join(OUTPUT_DIR, f"best_epoch{epoch+1}_step{global_step}_val{val_loss:.4f}.pt")
                # Save HF model and tokenizer
                model.save_pretrained(save_path)
                tokenizer.save_pretrained(save_path)
                print("  Saved best model to", save_path)

    # end epoch
    avg_epoch_loss = epoch_loss / steps if steps > 0 else float("inf")
    val_loss = evaluate(model, val_loader)
    print(f"Epoch {epoch+1} finished. Train avg loss: {avg_epoch_loss:.4f} | Val loss: {val_loss:.4f}")
    # Save checkpoint at epoch end
    ckpt_name = os.path.join(OUTPUT_DIR, f"epoch{epoch+1}_val{val_loss:.4f}")
    model.save_pretrained(ckpt_name)
    tokenizer.save_pretrained(ckpt_name)
    print("Saved epoch model to", ckpt_name)


def get_response(questions):
    """Load checkpoint and generate answers."""
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch

    single_q = isinstance(questions, str)
    questions = [questions] if single_q else questions

    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_PATH)
    model = AutoModelForCausalLM.from_pretrained(CHECKPOINT_PATH)
    device = torch.device("mps" if torch.has_mps else "cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    answers = []
    with torch.no_grad():
        for q in questions:
            prompt = f"User: {q}\nAssistant:"
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            out = model.generate(**inputs, max_new_tokens=128, do_sample=True, top_p=0.95, temperature=0.3)
            text = tokenizer.decode(out[0], skip_special_tokens=True)
            answer = text.split("Assistant:")[-1].strip() if "Assistant:" in text else text.strip()
            answers.append(answer)

# -------------------------
# Quick generation test using the final model
# -------------------------
print("\nTraining finished. Running quick generations on a few prompts:")



