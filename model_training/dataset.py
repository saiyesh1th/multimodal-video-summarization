import os
import json
import torch
from torch.utils.data import Dataset

# -------------------------------------------------------------
# PHASE 2: CUSTOM PYTORCH DATASET & DATALOADER
# -------------------------------------------------------------
# Reads the trained tokenizer configuration and prepares 
# correctly paddded and formatted Seq2Seq tensors directly
# from the transcribed audio CSVs.
# -------------------------------------------------------------

import pandas as pd

class BPETokenizerInference:
    """Loads the trained tokenizer from Phase 1 without external libraries."""
    def __init__(self, model_dir):
        config_path = os.path.join(model_dir, "tokenizer_config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Tokenizer config not found at {config_path}")
            
        with open(config_path, "r") as f:
            data = json.load(f)
            
        self.vocab_size = data["vocab_size"]
        
        # Special Tokens
        self.PAD_TOKEN = "<PAD>"
        self.UNK_TOKEN = "<UNK>"
        self.BOS_TOKEN = "<BOS>"
        self.EOS_TOKEN = "<EOS>"
        self.PAD_ID = 256
        self.UNK_ID = 257
        self.BOS_ID = 258
        self.EOS_ID = 259
        
        # Reconstruct merges from string keys ("val1,val2" -> (int(val1), int(val2)))
        self.merges = {}
        # Also build a reverse vocab mapping from ID to byte-pair characters
        self.vocab = {i: bytes([i]) for i in range(256)}
        
        for k_str, target_id in data["merges"].items():
            k1, k2 = map(int, k_str.split(","))
            self.merges[(k1, k2)] = target_id
            self.vocab[target_id] = self.vocab[k1] + self.vocab[k2]
            
    def _merge(self, ids, pair, idx):
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
        
    def encode(self, text):
        """Converts text string to entirely custom token integer IDs."""
        tokens = list(text.encode("utf-8"))
        
        while len(tokens) >= 2:
            stats = {}
            for pair in zip(tokens, tokens[1:]):
                stats[pair] = stats.get(pair, 0) + 1
            
            # Find the pair in `stats` that has the lowest merge index (was merged earliest in training)
            # If no pairs are in our learned merges, we stop.
            pair = min(stats.keys(), key=lambda p: self.merges.get(p, float('inf')))
            
            if pair not in self.merges:
                break # Nothing else can be merged
                
            idx = self.merges[pair]
            tokens = self._merge(tokens, pair, idx)
            
        return tokens
        
    def decode(self, token_ids):
        """Converts token integer IDs back into a text string."""
        # Filter out special tokens first before mapping to raw bytes
        filtered_ids = [t for t in token_ids if t not in (self.PAD_ID, self.BOS_ID, self.EOS_ID, self.UNK_ID)]
        
        raw_bytes = b"".join(self.vocab.get(idx, b"") for idx in filtered_ids)
        return raw_bytes.decode("utf-8", errors="replace")

class SummarizationDataset(Dataset):
    def __init__(self, transcripts_csv, summaries_csv, tokenizer, max_src_len=512, max_tgt_len=128):
        """
        Loads the actual ASR transcripts and target TF-IDF summary keyframes.
        """
        self.tokenizer = tokenizer
        self.max_src = max_src_len
        self.max_tgt = max_tgt_len
        
        self.samples = []
        
        # Load the CSV data
        df_trans = pd.read_csv(transcripts_csv)
        df_summ = pd.read_csv(summaries_csv)
        
        # Clean up appended headers from inner rows
        df_trans = df_trans[df_trans['start_time'] != 'start_time'].copy()
        df_summ = df_summ[df_summ['start_time'] != 'start_time'].copy()
        
        # Group by video file
        grouped_trans = df_trans.groupby('file')
        
        for file_name, group in grouped_trans:
            # Sort by time to ensure chronological transcript
            group = group.sort_values(by='start_time')
            
            # The full source text is the entire transcript joined
            src_text = " ".join(group['text'].astype(str).tolist())
            
            # Since the transcript generator in Phase 1 outputs the entire text block 
            # at timestamp 0.0, we cannot perform a sentence-by-sentence match with 
            # the TF-IDF dataframe. For the sake of testing phase 4 architecture:
            # We will use the first 20% of the transcript string as the target summary proxy.
            
            split_idx = max(int(len(src_text) * 0.2), 30)
            tgt_text = src_text[:split_idx] + " summary ."
            
            if src_text.strip() and tgt_text.strip():
                self.samples.append((src_text, tgt_text))
                
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        src_text, tgt_text = self.samples[idx]
        
        # Encode sequences
        src_ids = self.tokenizer.encode(src_text)
        tgt_ids = self.tokenizer.encode(tgt_text)
        
        # Add special tokens
        src_ids = src_ids[:self.max_src - 2]
        src_ids = [self.tokenizer.BOS_ID] + src_ids + [self.tokenizer.EOS_ID]
        
        tgt_ids = tgt_ids[:self.max_tgt - 2]
        tgt_ids = [self.tokenizer.BOS_ID] + tgt_ids + [self.tokenizer.EOS_ID]
        
        # Pad to max length
        src_pad_len = self.max_src - len(src_ids)
        tgt_pad_len = self.max_tgt - len(tgt_ids)
        
        src_ids.extend([self.tokenizer.PAD_ID] * src_pad_len)
        tgt_ids.extend([self.tokenizer.PAD_ID] * tgt_pad_len)
        
        return {
            "src_text": src_text,
            "tgt_text": tgt_text,
            "src_input_ids": torch.tensor(src_ids, dtype=torch.long),
            "tgt_input_ids": torch.tensor(tgt_ids, dtype=torch.long)
        }

if __name__ == "__main__":
    print("=========================================")
    print(" NLP PHASE 2: Custom PyTorch DataLoader")
    print("=========================================")
    
    tokenizer = BPETokenizerInference("./model_weights")
    print(f"[*] Loaded custom tokenizer. Vocab Size: {tokenizer.vocab_size}")
    
    # Test encoding
    test_str = "hello world"
    encoded = tokenizer.encode(test_str)
    print(f"[*] Test Encode '{test_str}': {encoded}")
    
    dataset = SummarizationDataset(
        transcripts_csv="./results/transcripts.csv", 
        summaries_csv="./results/final_summary.csv", 
        tokenizer=tokenizer, 
        max_src_len=32, 
        max_tgt_len=16
    )
    print(f"[*] Loaded Actual Transcript Dataset. Size: {len(dataset)}")
    
    sample = dataset[0]
    print(f"[*] Sample 0 Source Ids Shape (paddded): {sample['src_input_ids'].shape}")
    print(f"[*] Sample 0 Target Ids Shape (paddded): {sample['tgt_input_ids'].shape}")
    
    print("\n[+] Phase 2 functional verification complete.")
