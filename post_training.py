import os
from datetime import datetime

import pandas as pd
import tiktoken
import torch
from torch.utils.data import DataLoader, random_split
from pico_llm import TransformerModel, generate_text
import csv


enc = tiktoken.get_encoding("gpt2")
EOS_TOKEN_ID = enc.eot_token if hasattr(enc, "eot_token") else enc.encode("<|end|>")[0]

DEVICE = "mps"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

# CHANGE HERE: How often vlalidation is to be calculated, also how often graph points are made
VAL_INTERVAL = 1

def test_post_trained_model(pessage, question, checkpoint_path):

    model = TransformerModel(d_model=768, n_heads=12, n_blocks=12).to(DEVICE)
    print("Loading the model...")


    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    print("Model loaded successfully!")

    model.eval()
    sample_question = f"Passage: {pessage}\nUser: {question}"
    sample_prompt = f"{sample_question}\nAssistant:"
    with torch.no_grad():
        out, _ = generate_text(
                        model=model,
                        top_p=0.15,
                        enc=enc,
                        init_text=sample_prompt,
                        max_new_tokens=100,
                        device=DEVICE,
                        use_kv_cache_for_eval=None,
                        truncate_on="<|endoftext|>"

                    )
        print(f"    Q: {sample_question}")
        print(f"    A (0.15): {out.split('Assistant:')[-1].strip()}")
        print()
        out, _ = generate_text(
                        model=model,
                        top_p=0.50,
                        enc=enc,
                        init_text=sample_prompt,
                        max_new_tokens=100,
                        device=DEVICE,
                        use_kv_cache_for_eval=None,
                        truncate_on="<|endoftext|>"
                    )

        print(f"    Q: {sample_question}")
        print(f"    A (0.50): {out.split('Assistant:')[-1].strip()}")
        print()

        out, _ = generate_text(
                        model=model,
                        top_p=0.95,
                        enc=enc,
                        init_text=sample_prompt,
                        max_new_tokens=100,
                        device=DEVICE,
                        use_kv_cache_for_eval=None,
                        truncate_on="<|endoftext|>"

                    )

        print(f"    Q: {sample_question}")
        print(f"    A (0.95): {out.split('Assistant:')[-1].strip()}")
        print()
    print("voluntarily exiting after test...")
    exit(0)


# TRAIN OR TEST MODE
# Uncomment this code for testing. The checkpoint will be taken from
# test_post_trained_model(
#     pessage="He redressed and left the side room, boiling with concern. However, he found Denkmal at a desk, reading Emily’s medical chart; she was off, he realized, with a female nurse, so everything was all right.",
#     question="was Denkmal with a female nurse",
#     checkpoint_path="/Users/nilarnabdebnath/Documents/course_work/ml/small-llm-text-generator/sft_checkpoints/transformersft_20251206_191844/step_3350_LOSS_0.3269.pt"
# )

def format_row(row):
    question = str(row["QUESTION"]).strip()
    answer = str(row["ANSWER"]).strip()
    return f"User: {question}\nAssistant: {answer}"


def format_row_pessage_question_answer(row):
    passage = str(row["passage"]).strip()
    question = str(row["question"]).strip()
    answer = str(row["answer"]).strip()
    answer = answer.replace("True", "yes").replace("False", "no")
    print("Formatted:", passage, question, "answer:", answer)
    return f"Passage: {passage}\nUser: {question}\nAssistant: {answer}"



def encode_for_sft(text):
    # find where answer begins
    answer_marker = "Assistant:"
    split_idx = text.index(answer_marker) + len(answer_marker)

    prompt = text[:split_idx]
    full_text = text

    input_ids = enc.encode(full_text)
    prompt_ids = enc.encode(prompt)

    input_ids.append(EOS_TOKEN_ID)

    # labels = copy(input_ids)
    labels = input_ids.copy()
    labels[:len(prompt_ids)] = [-100] * len(prompt_ids)

    return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


class SFTDataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.texts = [format_row(r) for _, r in df.iterrows()]
        # self.texts = [format_row_pessage_question_answer(r) for _, r in df.iterrows()]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return encode_for_sft(self.texts[idx])


def collate(batch):
    inputs = [b[0] for b in batch]
    labels = [b[1] for b in batch]

    inputs = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=0)
    labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)

    return inputs, labels


def evaluate(model, dataloader, device):
    """Evaluate model on validation set"""
    model.eval()
    total_loss = 0
    total_batches = 0
    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)

    with torch.no_grad():
        for input_ids, labels in dataloader:
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            logits = model(input_ids)
            shift_logits = logits[:, :-1].contiguous()
            shift_labels = labels[:, 1:].contiguous()

            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
            total_loss += loss.item()
            total_batches += 1

    model.train()
    return total_loss / total_batches if total_batches > 0 else float('inf')


# Load and split dataset
# df = pd.read_excel('generated_logic_sft_10k.xlsx')
# df = pd.read_excel('dataset/boolq_train_sft.xlsx')
# CHANGE HERE: For choosing which datset to choose from
df = pd.read_excel('dataset/graph_sft_dataset.xlsx')

full_dataset = SFTDataset(df)

# Split into train/val (90/10)
train_size = int(0.90 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

print(f"Train size: {train_size}, Validation size: {val_size}")

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, collate_fn=collate)

# Load model
model = TransformerModel(d_model=768, n_heads=12, n_blocks=12).to(DEVICE)

# CHANGE HERE: For choosing which checkpoint the model should choose training from
# checkpoint_path = "/Users/nilarnabdebnath/Documents/course_work/ml/pico-llm/checkpoints/transformer_20251202_223936/step_37327_LOSS_0.3591.pt"
# checkpoint_path = "/Users/nilarnabdebnath/Documents/course_work/ml/pico-llm/checkpoints/transformer_rope_20251201_004731/step_225_LOSS_4.4066.pt"
# checkpoint_path = "trained_models/ojaswi_step_38990_LOSS_0.4860.pt"
# checkpoint_path = "/Users/nilarnabdebnath/Documents/course_work/ml/small-llm-text-generator/checkpoints/transformer_graph_pretrain_20251207_103827/step_136_LOSS_0.1972.pt"
checkpoint_path = "/Users/nilarnabdebnath/Documents/course_work/ml/small-llm-text-generator/checkpoints/transformer_graph3_pretrain_20251207_111022/step_130_LOSS_0.2467.pt"
print("Loading the model...")
checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
model.load_state_dict(checkpoint['model_state_dict'], strict=False)
print("Model loaded successfully!")

# Setup training
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
session_name = f"transformersft_{timestamp}"
save_dir = f"sft_checkpoints/{session_name}"
os.makedirs(save_dir, exist_ok=True)

loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)

# Training parameters
NUM_EPOCHS = 100
SAVE_EVERY_N_STEPS = 500
best_val_loss = float('inf')
global_step = 0

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)


print(f"\nStarting training for {NUM_EPOCHS} epochs...")
print(f"Checkpoints will be saved to: {save_dir}\n")

os.makedirs("metrics_sft", exist_ok=True)
metric_log_path = f"metrics_sft/{session_name}.csv"
metric_log_file = open(metric_log_path, mode='a', newline='')
metric_writer = csv.writer(metric_log_file)

# Training loop
for epoch in range(NUM_EPOCHS):
    model.train()
    epoch_loss = 0
    num_batches = 0

    print(f"{'=' * 60}")
    print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")
    print(f"{'=' * 60}")

    for batch_idx, (input_ids, labels) in enumerate(train_loader):
        global_step += 1

        input_ids = input_ids.to(DEVICE)
        labels = labels.to(DEVICE)

        # Forward pass
        logits = model(input_ids)

        # Shift for next token prediction
        shift_logits = logits[:, :-1].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # NEW
        optimizer.step()
        optimizer.zero_grad()

        epoch_loss += loss.item()
        num_batches += 1

        if (batch_idx + 1) % 1000 == 0:
            avg_loss = epoch_loss / num_batches
            print(
                f"  Step {global_step} | Batch {batch_idx + 1}/{len(train_loader)} | Loss: {loss.item():.4f} | Avg Loss: {avg_loss:.4f}")

            # Generate sample text at end of epoch
            print(f"\n  Sample generation:")
            # sample_questions = ["If A is bigger than B and B is Bigger than C, who is tallest?", "Alice has 2 items and buys 2 more. How many items does she have?"]
            # sample_questions = [
            #     "Passage: Good Samaritan laws offer legal protection to people who give reasonable assistance to those who are, or who they believe to be, injured, ill, in peril, or otherwise incapacitated. The protection is intended to reduce bystanders' hesitation to assist, for fear of being sued or prosecuted for unintentional injury or wrongful death. An example of such a law in common-law areas of Canada: a good Samaritan doctrine is a legal principle that prevents a rescuer who has voluntarily helped a victim in distress from being successfully sued for wrongdoing. Its purpose is to keep people from being reluctant to help a stranger in need for fear of legal repercussions should they make some mistake in treatment. By contrast, a duty to rescue law requires people to offer assistance and holds those who fail to do so liable.\nUser: do good samaritan laws protect those who help at an accident?",
            #     "Passage: The series premiered in the United States on Starz on 12 April 2013, and its second season premiered on 22 March 2014. The series was renewed for a third season, which premiered on 24 October 2015. On 23 July 2015, Starz announced that the third season would be the show's last. However Goyer has left it open for a miniseries return. \nUser: will there be a season 4 of da vinci's demons?",
            # ]
            # CHANGE HERE: to chagne the prompt that will be tested during training
            sample_questions = [
                "User: A is conncted to B. B is connected to A.",
                "User: A is connected to B. B is connected to C.",
            ]
            for sample_question in sample_questions:
                sample_prompt = f"{sample_question}\nAssistant:"

                model.eval()
                with torch.no_grad():
                    out, _ = generate_text(
                        model=model,
                        top_p=0.15,
                        enc=enc,
                        init_text=sample_prompt,
                        max_new_tokens=20,
                        device=DEVICE,
                        use_kv_cache_for_eval=None,
                        truncate_on="<|endoftext|>"

                    )
                    print(f"    Q: {sample_question}")
                    print(f"    A (0.15): {out.split('Assistant:')[-1].strip()}")
                    print()
                    out, _ = generate_text(
                        model=model,
                        top_p=0.50,
                        enc=enc,
                        init_text=sample_prompt,
                        max_new_tokens=20,
                        device=DEVICE,
                        use_kv_cache_for_eval=None,
                        truncate_on="<|endoftext|>"
                    )

                    print(f"    Q: {sample_question}")
                    print(f"    A (0.50): {out.split('Assistant:')[-1].strip()}")
                    print()

                    out, _ = generate_text(
                        model=model,
                        top_p=0.95,
                        enc=enc,
                        init_text=sample_prompt,
                        max_new_tokens=20,
                        device=DEVICE,
                        use_kv_cache_for_eval=None,
                        truncate_on="<|endoftext|>"

                    )

                    print(f"    Q: {sample_question}")
                    print(f"    A (0.95): {out.split('Assistant:')[-1].strip()}")
                    print()

                model.train()

        if (batch_idx + 1) % VAL_INTERVAL == 0:
            avg_loss = epoch_loss / num_batches
            print("getting validatino loss...")
            val_loss = evaluate(model, val_loader, DEVICE)

            # Writing metrics
            metric_writer.writerow([global_step, avg_loss, val_loss])
            metric_log_file.flush()

            print(
                f"  Step {global_step} | Batch {batch_idx + 1}/{len(train_loader)} | Loss: {loss.item():.4f} | Avg Loss: {avg_loss:.4f} | Val Loss: {val_loss:.4f}")


        if global_step % SAVE_EVERY_N_STEPS == 0:
            step_ckpt = os.path.join(save_dir, f"step_{global_step}_LOSS_{loss.item():.4f}.pt")
            torch.save({
                'epoch': epoch,
                'global_step': global_step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss.item(),
                'scheduler_state_dict': scheduler.state_dict(),  # NEW

            }, step_ckpt)
            print(f"  → Checkpoint saved: step_{global_step}")


    # CHANGE HERE: After each eopch, which questions shoul be answered right after epoch ocmplete
    sample_questions = [
                "User: A is connected to B. B is connected to A.",
                "User: A is connected to B. B is connected to C.",
            ]

    for sample_question in sample_questions:
                sample_prompt = f"{sample_question}\nAssistant:"

                model.eval()
                with torch.no_grad():
                    out, _ = generate_text(
                        model=model,
                        top_p=0.15,
                        enc=enc,
                        init_text=sample_prompt,
                        max_new_tokens=100,
                        device=DEVICE,
                        use_kv_cache_for_eval=None,
                        truncate_on="<|endoftext|>"

                    )
                    print(f"    Q: {sample_question}")
                    print(f"    A (0.15): {out.split('Assistant:')[-1].strip()}")
                    print()
                    # out, _ = generate_text(
                    #     model=model,
                    #     top_p=0.50,
                    #     enc=enc,
                    #     init_text=sample_prompt,
                    #     max_new_tokens=100,
                    #     device=DEVICE,
                    #     use_kv_cache_for_eval=None,
                    #     truncate_on="<|endoftext|>"
                    # )

                    # print(f"    Q: {sample_question}")
                    # print(f"    A (0.50): {out.split('Assistant:')[-1].strip()}")
                    # print()

                    # out, _ = generate_text(
                    #     model=model,
                    #     top_p=0.95,
                    #     enc=enc,
                    #     init_text=sample_prompt,
                    #     max_new_tokens=100,
                    #     device=DEVICE,
                    #     use_kv_cache_for_eval=None,
                    #     truncate_on="<|endoftext|>"

                    # )

                    print(f"    Q: {sample_question}")
                    print(f"    A (0.95): {out.split('Assistant:')[-1].strip()}")
                    print()

                model.train()
        


    # Calculate average training loss for epoch
    avg_train_loss = epoch_loss / num_batches

    # Validation
    print(f"\n  Running validation...")
    val_loss = evaluate(model, val_loader, DEVICE)

    print(f"\n  Epoch {epoch + 1} Summary:")
    print(f"    Train Loss: {avg_train_loss:.4f}")
    print(f"    Val Loss:   {val_loss:.4f}")

    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_ckpt = os.path.join(save_dir, f"best_model_epoch_{epoch + 1}_VAL_{val_loss:.4f}.pt")
        torch.save({
            'epoch': epoch,
            'global_step': global_step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': avg_train_loss,
            'val_loss': val_loss,
            'scheduler_state_dict': scheduler.state_dict(),  # NEW

        }, best_ckpt)
        print(f"  ✓ New best model saved! (Val Loss: {val_loss:.4f})")

    scheduler.step()



print(f"\n{'=' * 60}")
print("Training completed!")
print(f"Best validation loss: {best_val_loss:.4f}")
print(f"Checkpoints saved to: {save_dir}")
print(f"{'=' * 60}\n")

# Final test generation
print("Final model test:")

test_questions = [
                "User: A is conncted to B. B is connected to A.",
                "User: A is connected to B. B is connected to C. Is A connected to C?",
            ]

model.eval()
for question in test_questions:
    prompt = f"User: {question}\nAssistant:"
    with torch.no_grad():
        out, _ = generate_text(
            model=model,
            enc=enc,
            init_text=prompt,
            max_new_tokens=80,
            device=DEVICE,
            use_kv_cache_for_eval=None
        )
    print(f"\nQ: {question}")
    print(f"A: {out.split('Assistant:')[-1].strip()}")
    print("-" * 60)