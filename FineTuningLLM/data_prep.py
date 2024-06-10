from datasets import load_dataset
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
import faiss
from datasets import Dataset, DatasetDict
from tqdm.autonotebook import tqdm
import numpy as np


def tokenize_combined(dataset: Dataset):
    instruction_token_counts = [
    len(tokenizer.tokenize(example["0"])) for example in dataset["train"]
    ]
    output_token_counts = [
        len(tokenizer.tokenize(example["1"])) for example in dataset["train"]
    ]
    combined_token_counts = [
        instruction + output
        for instruction, output in zip(instruction_token_counts, output_token_counts)
    ]
    return combined_token_counts

def deduplicate_dataset(dataset: Dataset, model: str, threshold: float):
    sentence_model = SentenceTransformer(model, device="cuda")
    outputs = [example["1"] for example in dataset["train"]]

    print("Converting text to embeddings...")
    embeddings = sentence_model.encode(outputs, show_progress_bar=True)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    normalized_embeddings = embeddings / np.linalg.norm(
        embeddings, axis=1, keepdims=True
    )
    index.add(normalized_embeddings)

    print("Filtering out near-duplicates...")
    D, I = index.search(normalized_embeddings, k=2)
    to_keep = []

    for i in tqdm(range(len(embeddings)), desc="Filtering"):
        if D[i, 1] >= threshold:
            nearest_neighbor = I[i, 1]
            if i not in to_keep and nearest_neighbor not in to_keep:
                to_keep.append(i)
        else:
            to_keep.append(i)

    dataset = dataset["train"].select(to_keep)
    return DatasetDict({"train": dataset})


def get_top_k_rows(dataset, token_counts, k):
    sorted_indices = sorted(
        range(len(token_counts)), key=lambda i: token_counts[i], reverse=True
    )
    top_k_indices = sorted_indices[:k]

    top_k_data = {
        "0": [dataset["train"][i]["0"] for i in top_k_indices],
        "1": [dataset["train"][i]["1"] for i in top_k_indices],
    }

    return Dataset.from_dict(top_k_data)


def chat_template(example):
    example["0"] = f"### Pertanyaan:\n{example['0']}\n\n### Jawaban:\n"
    return example


import huggingface_hub
huggingface_hub.login("token_read")

dataset = load_dataset("genta-tech/squad_pairs_indo")
tokenizer = AutoTokenizer.from_pretrained("Yellow-AI-NLP/komodo-7b-base")
combined_token_counts = tokenize_combined(dataset)

valid_indices = [i for i, count in enumerate(combined_token_counts) if count <= 100]
print(f"Number of valid rows: {len(valid_indices)}")
print(f"Removing {len(dataset['train']) - len(valid_indices)} rows...")

dataset["train"] = dataset["train"].select(valid_indices)

token_counts = [combined_token_counts[i] for i in valid_indices]

deduped_dataset = deduplicate_dataset(dataset, "firqaaa/indo-sentence-bert-base", 0.95)

combined_token_counts = tokenize_combined(deduped_dataset)

k = 1000
top_k_dataset = get_top_k_rows(deduped_dataset, combined_token_counts, k)

dataset = DatasetDict({"train": top_k_dataset})
dataset = dataset.map(chat_template)
access_token_write = "token_write"
dataset.push_to_hub("squad_pairs_indo_1000", token=access_token_write)