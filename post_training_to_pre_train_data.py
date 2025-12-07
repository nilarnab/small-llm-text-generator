import pandas as pd

# === 1. Load your generated SFT dataset ===
df = pd.read_excel("generated_logic_sft_10k.xlsx")

# Assumes the dataset has columns: "question", "answer"
# If yours are named differently, change here:
QUESTION_COL = "QUESTION"
ANSWER_COL = "ANSWER"

# === 2. Prepare warm-up text output ===
output_lines = []

for i, row in df.iterrows():
    q = str(row[QUESTION_COL]).strip()
    a = str(row[ANSWER_COL]).strip()

    # Format sample (NO masking, pure next-token)
    block = (
        f"User: {q}\n"
        f"Assistant: {a}\n\n"
    )
    output_lines.append(block)

# === 3. Save to txt ===
output_text = "".join(output_lines)

with open("dataset/warmup_pretrain_data.txt", "w", encoding="utf-8") as f:
    f.write(output_text)

print("Done! Saved as warmup_pretrain_data.txt")