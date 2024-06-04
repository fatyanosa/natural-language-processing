from datasets import load_dataset
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
import faiss
from datasets import Dataset, DatasetDict
from tqdm.autonotebook import tqdm
import numpy as np
import pickle

dataset = load_dataset("genta-tech/squad_pairs_indo")
# 1. Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("NousResearch/Llama-2-7b-hf")

# 2. Tokenize each row and count the number of tokens
instruction_token_counts = [len(tokenizer.tokenize(example["0"])) for example in dataset['train']]
output_token_counts = [len(tokenizer.tokenize(example["1"])) for example in dataset['train']]
combined_token_counts = [instruction + output for instruction, output in zip(instruction_token_counts, output_token_counts)]

def deduplicate_dataset(dataset: Dataset, model: str, threshold: float):
    sentence_model = SentenceTransformer(model, device='cuda')
    outputs = [example["1"] for example in dataset['train']]

    print("Converting text to embeddings...")
    embeddings = sentence_model.encode(outputs, show_progress_bar=True)
    
    with open("embeddings.pkl", "wb") as fOut:
        pickle.dump({'sentences': outputs, 'embeddings': embeddings},fOut)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    normalized_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    index.add(normalized_embeddings)

    print("Filtering out near-duplicates...")
    D, I = index.search(normalized_embeddings, k=2)
    to_keep = []

    for i in tqdm(range(len(embeddings)), desc="Filtering"):
        # If the second closest vector (D[i, 1]) has cosine similarity above the threshold
        if D[i, 1] >= threshold:
            # Check if either the current item or its nearest neighbor is already in the to_keep list
            nearest_neighbor = I[i, 1]
            if i not in to_keep and nearest_neighbor not in to_keep:
                # If not, add the current item to the list
                to_keep.append(i)
        else:
            # If the similarity is below the threshold, always keep the current item
            to_keep.append(i)

    dataset = dataset['train'].select(to_keep)
    return DatasetDict({"train": dataset})


# Get the top k rows with the most tokens
def get_top_k_rows(dataset, token_counts, k):
    # Sort by descending token count and get top k indices
    sorted_indices = sorted(range(len(token_counts)), key=lambda i: token_counts[i], reverse=True)
    top_k_indices = sorted_indices[:k]

    # Extract top k rows
    top_k_data = {
        "0": [dataset['train'][i]["0"] for i in top_k_indices],
        "1": [dataset['train'][i]["1"] for i in top_k_indices]
    }

    return Dataset.from_dict(top_k_data)

def chat_template(example):
    example["instruction"] = f"### Instruction:\n{example['0']}\n\n### Response:\n"
    return example

valid_indices = [i for i, count in enumerate(combined_token_counts) if count <= 100]
print(f"Number of valid rows: {len(valid_indices)}")
print(f"Removing {len(dataset['train']) - len(valid_indices)} rows...")

# Extract valid rows based on indices
dataset['train'] = dataset['train'].select(valid_indices)

# Get token counts for valid rows
token_counts = [combined_token_counts[i] for i in valid_indices]

deduped_dataset = deduplicate_dataset(dataset, "firqaaa/indo-sentence-bert-base", 0.95)

# k = 1000  # You can adjust this value as needed
# top_k_dataset = get_top_k_rows(deduped_dataset, combined_token_counts, k)

# Save these rows in a Dataset object with a 'train' split
dataset = DatasetDict({"train": deduped_dataset})

dataset = dataset.map(chat_template)
access_token_write = "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"
dataset.push_to_hub("squad_pairs_indo_1000", token=access_token_write)
# login(token = access_token_write)
