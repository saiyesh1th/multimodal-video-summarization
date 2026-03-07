import os
import re
import json
from collections import defaultdict

# -------------------------------------------------------------
# PHASE 1: DATA PREPARATION & CUSTOM TOKENIZER TRAINING
# -------------------------------------------------------------
# This script trains a custom Byte-Pair Encoding (BPE) tokenizer
# entirely from scratch. No pre-trained models or weights are used.
# -------------------------------------------------------------

DATA_DIR = "./data"
MODEL_DIR = "./model_weights"

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# 1. Provide some sample training data
# In a real scenario, you'd download CNN/DailyMail or SAMSum here.
# For demonstration of the *from scratch* pipeline, we'll generate
# a synthetic corpus if one doesn't exist, or load a small dummy one.
# You will want to replace this with a real, large dataset download
# using something like `datasets` library, but we'll stick to raw 
# text files to keep it 100% custom and dependency-free where possible.

def download_or_create_dummy_data():
    raw_text_path = os.path.join(DATA_DIR, "raw_corpus.txt")
    if not os.path.exists(raw_text_path):
        print("[!] Creating synthetic raw corpus for Tokenizer Training...")
        sample_texts = [
            "hello world . this is a test transcript .",
            "the speaker in the audio says hello world .",
            "we are building a deep learning model from scratch .",
            "this model will summarize transcripts .",
            "transformers use self attention mechanisms .",
            "byte pair encoding is a compression technique used for tokenization .",
            "hello , this is another audio transcript test .",
            "summarizing audio is a complex natural language processing task .",
            "learning from scratch requires a lot of data ."
        ] * 100 # Multiply to get more frequency counts
        
        with open(raw_text_path, "w", encoding="utf-8") as f:
            for text in sample_texts:
                f.write(text + "\n")
    return raw_text_path


# 2. Basic BPE Tokenizer Implementation From Scratch
class CustomBPETokenizer:
    def __init__(self, vocab_size=500):
        self.vocab_size = vocab_size
        self.merges = {}
        self.vocab = {}
        self.inverse_vocab = {}
        
        # Special Tokens
        self.PAD_TOKEN = "<PAD>"
        self.UNK_TOKEN = "<UNK>"
        self.BOS_TOKEN = "<BOS>" # Beginning of sequence
        self.EOS_TOKEN = "<EOS>" # End of sequence
        
        # Initialize with standard byte/ascii characters
        self.initial_vocab = {chr(i): i for i in range(256)}
        self.initial_vocab[self.PAD_TOKEN] = 256
        self.initial_vocab[self.UNK_TOKEN] = 257
        self.initial_vocab[self.BOS_TOKEN] = 258
        self.initial_vocab[self.EOS_TOKEN] = 259
        
    def get_stats(self, ids):
        """Count frequencies of adjacent pairs."""
        counts = defaultdict(int)
        for pair in zip(ids, ids[1:]):
            counts[pair] += 1
        return counts
        
    def merge(self, ids, pair, idx):
        """Replace all occurrences of pair in ids with idx."""
        newids = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
                newids.append(idx)
                i += 2
            else:
                newids.append(ids[i])
                i += 1
        return newids

    def train(self, text):
        """Train BPE on raw text."""
        print(f"[*] Training BPE Tokenizer (Target Vocab Size: {self.vocab_size})...")
        
        # 1. Convert text to initial byte/character IDs
        # We encode as UTF-8 bytes to handle any character
        tokens = list(text.encode("utf-8"))
        
        num_merges = self.vocab_size - len(self.initial_vocab)
        if num_merges < 0:
            print("[!] Vocab size too small for initial bytes + specials.")
            return

        vocab_id = len(self.initial_vocab)
        merges = {}
        
        for i in range(num_merges):
            stats = self.get_stats(tokens)
            if not stats: 
                break
            
            # Find the most frequent pair
            best_pair = max(stats, key=stats.get)
            
            # Merge it
            tokens = self.merge(tokens, best_pair, vocab_id)
            merges[best_pair] = vocab_id
            
            if (i+1) % 50 == 0:
                print(f"  - Merge {i+1}/{num_merges}: {best_pair} -> {vocab_id}")
                
            vocab_id += 1
            
        self.merges = merges
        
        # Build final vocab dictionaries
        self.vocab = self.initial_vocab.copy()
        
        # In a generic BPE, the vocab string representations can be complex.
        # We just store the merges for encoding/decoding.
        print("[*] Tokenizer training complete!")
        
    def save(self, save_path):
        os.makedirs(save_path, exist_ok=True)
        # Convert tuple keys to string for JSON
        merges_str = {f"{k[0]},{k[1]}": v for k, v in self.merges.items()}
        
        data = {
            "vocab_size": self.vocab_size,
            "merges": merges_str
        }
        with open(os.path.join(save_path, "tokenizer_config.json"), "w") as f:
            json.dump(data, f, indent=2)
        print(f"[*] Tokenizer saved to {save_path}")

if __name__ == "__main__":
    print("=========================================")
    print(" NLP PHASE 1: Data Prep & Tokenizer")
    print("=========================================")
    
    # 1. Prepare Data
    corpus_file = download_or_create_dummy_data()
    
    with open(corpus_file, "r", encoding="utf-8") as f:
        raw_text = f.read()
        
    # 2. Train Custom Tokenizer
    # We use a tiny vocab size for the custom dummy dataset.
    # For a real dataset like CNN/DM, you'd use 30000+.
    tokenizer = CustomBPETokenizer(vocab_size=350)
    tokenizer.train(raw_text)
    
    # 3. Save Tokenizer
    tokenizer.save(MODEL_DIR)
    
    print("\n[+] Phase 1 Complete. Ready for Phase 2 (Dataset pipeline).")
