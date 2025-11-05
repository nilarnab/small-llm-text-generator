# This script downloads the TriviaQA dataset and saves
# the first n question-answer pairs into a text file.

from datasets import load_dataset
from tqdm import tqdm

print("Downloading the dataset")
dataset = load_dataset("trivia_qa", "rc", split="train")

qa_pairs = []

for item in tqdm(dataset):
    question = item.get("question", "").strip()
    answer_info = item.get("answer", {})
    answer = answer_info.get("value", "").strip()

    if question and answer:
        qa_pairs.append((question, answer))

    # choose the num of pairs
    if len(qa_pairs) >= 5000:
        break

print("Total question-answer pairs collected:", len(qa_pairs))

output_file = "qa_data_triviaqa.txt"
print("Saving data to", output_file)

with open(output_file, "w", encoding="utf-8") as f:
    for q, a in qa_pairs:
        f.write("Question: " + q + " Answer: " + a + "\n")

print("Saved", len(qa_pairs), "question-answer pairs.")
