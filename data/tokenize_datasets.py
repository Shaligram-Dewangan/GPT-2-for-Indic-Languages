import os
import numpy as np
from datasets import load_dataset
from tokenizers import Tokenizer
from tqdm import tqdm
from itertools import chain

# Load datasets
print("--- Loading Datasets ---")
fw_eng = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train")
fw_hi = load_dataset("HuggingFaceFW/fineweb-2", name="hin_Deva", split="train")

local_dir = "datasets/FW_tokenized_20B" 

DATA_CACHE_DIR = os.path.join(os.getcwd(), local_dir)
os.makedirs(DATA_CACHE_DIR, exist_ok=True)


# Get dataset sizes for handling exhaustion
eng_dataset_size = len(fw_eng)
hi_dataset_size = len(fw_hi)
print(f"English dataset size: {eng_dataset_size} documents")
print(f"Hindi dataset size: {hi_dataset_size} documents")


tokenizer = Tokenizer.from_file("../tokenizer/tokenizer.json")


def tokenize(docs: list, is_english_sum):
    eot = tokenizer.encode("<|endoftext|>").ids
    encoded_doc_list = tokenizer.encode_batch(docs)
    tokenized_doc_list = [eot + doc.ids for doc in encoded_doc_list]

    english_docs = tokenized_doc_list[:is_english_sum]
    hindi_docs = tokenized_doc_list[is_english_sum:]

    english_index = 0
    hindi_index = 0

    batch_eng_token_count = 0
    batch_hi_token_count = 0

    reordered_docs = [None] * batch_size

    i = 0
    for boolean in is_english:
        if boolean:
            reordered_docs[i] = english_docs[english_index]

            batch_eng_token_count += len(english_docs[english_index])
            english_index += 1
            i += 1
        else:
            reordered_docs[i] = hindi_docs[hindi_index]

            batch_hi_token_count += len(hindi_docs[hindi_index])
            hindi_index += 1
            i += 1

    tokens = list(chain.from_iterable(reordered_docs))
    tokens_np = np.array(tokens, dtype=np.uint16)
    
    return tokens_np, batch_eng_token_count, batch_hi_token_count


def save_datafile(filename, tokens_np):
    np.save(filename, tokens_np)


# --- Probability Calculation --- 
mean_token_count_per_doc_eng = 1235.2
mean_token_count_per_doc_hi = 743.6

print(f"\nAverage Token-Count / Doucument, English: {mean_token_count_per_doc_eng}")
print(f"Average Token-Count / Doucument, Hindi: {mean_token_count_per_doc_hi}")

# Calculate probability of picking English and Hindi Document
# So that they are equal in the resulting tokenized dataset
p_select_english = mean_token_count_per_doc_hi / (mean_token_count_per_doc_eng + mean_token_count_per_doc_hi)
p_select_hindi = mean_token_count_per_doc_eng / (mean_token_count_per_doc_eng + mean_token_count_per_doc_hi)

print(f"Probability of selecting English document: {p_select_english:.4f}")
print(f"Probability of selecting Hindi document: {p_select_hindi:.4f}\n")


total_desired_tokens = int(20e9)    # 20 Billion
shard_size = int(1e8)    # 100 Million

global_eng_token_count = 0
global_hi_token_count = 0
global_processed_tokens = 0

shard_token_count = 0 # Tokens in current buffer
all_tokens_np_buffer = np.empty((shard_size,), dtype=np.uint16)

shard_index = 0
doc_eng_counter = 0
doc_hi_counter = 0

batch_size = 512

np.random.seed(64)

with tqdm(total=total_desired_tokens, unit=" tokens") as progress_bar:
    
    while global_processed_tokens < total_desired_tokens \
    or global_eng_token_count < (total_desired_tokens / 2) \
        or global_hi_token_count < (total_desired_tokens / 2):

        # --- Select Language and Document ---
        is_english = np.random.rand(batch_size) < p_select_english
        is_english_sum = int(sum(is_english))
        

        if (doc_eng_counter + is_english_sum) >= eng_dataset_size:
            print("Warning: Reached end of English dataset. Resetting counter.")
            print(f"English tokens processed so far: {global_eng_token_count}")
            doc_eng_counter = 0

        doc_eng = fw_eng[doc_eng_counter : doc_eng_counter + is_english_sum]['text']
        doc_eng_counter += is_english_sum

        if (doc_hi_counter + (batch_size - is_english_sum)) >= hi_dataset_size:
            print("Warning: Reached end of Hindi dataset. Resetting counter.")
            print(f"Hindi tokens processed so far: {global_hi_token_count}")
            doc_hi_counter = 0

        doc_hi = fw_hi[doc_hi_counter : doc_hi_counter + (batch_size - is_english_sum)]['text']
        doc_hi_counter += (batch_size - is_english_sum)


        # --- Tokenize ---
        docs = doc_eng + doc_hi
        tokens, batch_eng_token_count, batch_hi_token_count = tokenize(docs, is_english_sum)

        global_eng_token_count += batch_eng_token_count
        global_hi_token_count += batch_hi_token_count
        

        # --- Buffer and Save Logic ---
        num_tokens_to_add = len(tokens) # Number of tokens to add from current document(s)
        tokens_added_from_doc = 0 # Tokens from this doc(s) added to current buffer/shard

        while num_tokens_to_add > 0:
            remaining_space_in_buffer = shard_size - shard_token_count
            take_from_doc = min(num_tokens_to_add, remaining_space_in_buffer)

            # Add tokens to the current buffer
            start_idx_doc = len(tokens) - num_tokens_to_add
            end_idx_doc = start_idx_doc + take_from_doc
            all_tokens_np_buffer[shard_token_count : shard_token_count + take_from_doc] = tokens[start_idx_doc:end_idx_doc]

            shard_token_count += take_from_doc
            num_tokens_to_add -= take_from_doc
            tokens_added_from_doc += take_from_doc

            # If buffer is full, save the shard
            if shard_token_count == shard_size:
                split = "val" if shard_index == 0 else "train"
                filename = os.path.join(DATA_CACHE_DIR, f"data_{split}_{shard_index:03d}.npy")

                save_datafile(filename, all_tokens_np_buffer)
                # print(f"Saved shard {filename} with {shard_size} tokens")

                shard_index += 1
                shard_token_count = 0

        # Update progress bar
        progress_bar.update(tokens_added_from_doc)
        global_processed_tokens += tokens_added_from_doc


# --- Save the last partial shard (if any) ---
if shard_token_count > 0:
    split = "val" if shard_index == 0 else "train"
    filename = os.path.join(DATA_CACHE_DIR, f"data_{split}_{shard_index:03d}.npy")
    save_datafile(filename, all_tokens_np_buffer[:shard_token_count])
    # print(f"Saved final partial shard {filename} with {shard_token_count} tokens")

print("\nData generation complete.")
print(f"English tokens: {global_eng_token_count}")
print(f"Hindi tokens: {global_hi_token_count}")
print(f"Total tokens saved: {global_processed_tokens}")