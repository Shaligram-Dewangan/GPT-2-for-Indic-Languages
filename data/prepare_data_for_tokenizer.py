from datasets import load_dataset
from tqdm import tqdm

print(f"--- Loading / Downloading Datasets ---\n")

fw_eng = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train")
print(f"FineWeb-Edu 10B token (for English) has been downloaded, len: {len(fw_eng)}\n")

fw_hi = load_dataset("HuggingFaceFW/fineweb-2", name="hin_Deva", split="train")
print(f"Fineweb-2 hin_Deva ~10.6B words (for Hindi) has been downloaded, len: {len(fw_hi)}")

print("--------------------------------------\n")

print("Creating dataset for BPE tokenizer training with 1 Billion English and Hindi words")

SAVE_PATH = "../tokenizer/data_1B.txt"

desired_wc_per_lang = 50e7
avg_wc_per_doc_eng = 772 
avg_wc_per_doc_hi = 520

chunks = 10

eng_word_count = 0
hi_word_count = 0

with open(SAVE_PATH, "w", encoding="utf-8") as f:
    for i in tqdm(range(chunks), desc="Writing file in chunks"):

        subset_eng = fw_eng.select(range(i * int((desired_wc_per_lang/chunks) / avg_wc_per_doc_eng), 
                                        (i+1) * int((desired_wc_per_lang/chunks) / avg_wc_per_doc_eng)))["text"] # ~50 million words
        
        for line in subset_eng:
            eng_word_count += len(line.split())
            f.write(line + "\n")
        f.flush()

        subset_hi = fw_hi.select(range(i * int((desired_wc_per_lang/chunks) / avg_wc_per_doc_hi), 
                                        (i+1) * int((desired_wc_per_lang/chunks) / avg_wc_per_doc_hi)))["text"] # ~50 million words
        
        for line in subset_hi:
            hi_word_count += len(line.split())
            f.write(line + "\n")
        f.flush()

print(f"\nEnglish word count: {eng_word_count}, Hindi word count: {hi_word_count}")