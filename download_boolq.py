import pandas as pd

splits = {
    'train': 'data/train-00000-of-00001.parquet',
    'validation': 'data/validation-00000-of-00001.parquet'
}

df = pd.read_parquet("hf://datasets/google/boolq/" + splits["train"])

# Save as .xlsx
df.to_excel("dataset/boolq_train_sft.xlsx", index=False)