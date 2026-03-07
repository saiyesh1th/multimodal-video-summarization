import os
import json
import re
import math
import torch
from torch.utils.data import Dataset
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


# Common English stopwords to exclude from TF-IDF scoring
_STOPWORDS = {
    'i','me','my','myself','we','our','ours','ourselves','you','your','yours',
    'yourself','yourselves','he','him','his','himself','she','her','hers',
    'herself','it','its','itself','they','them','their','theirs','themselves',
    'what','which','who','whom','this','that','these','those','am','is','are',
    'was','were','be','been','being','have','has','had','having','do','does',
    'did','doing','a','an','the','and','but','if','or','because','as','until',
    'while','of','at','by','for','with','about','against','between','into',
    'through','during','before','after','above','below','to','from','up','down',
    'in','out','on','off','over','under','again','further','then','once','here',
    'there','when','where','why','how','all','both','each','few','more','most',
    'other','some','such','no','nor','not','only','own','same','so','than',
    'too','very','s','t','can','will','just','don','should','now','let','go',
    'get','got','going','come','came','see','know','like','make','put','take',
    'us','oh','ok','yeah','yes','alright','well','hey','hi'
}

def extractive_summary(text, num_sentences=5):
    """
    Improved custom TF-IDF extractive summarizer.
    Key improvements over the naive version:
    - Stopword filtering: common words don't pollute scores
    - Minimum word count: eliminates extremely short filler sentences
    - Sum-based TF-IDF (not average per unique word): longer, richer
      sentences naturally score higher than short isolated ones
    - Sublinear TF: log(1+tf) dampens the effect of repetition
    - Position weighting: slight bonus for sentences appearing early
      (they tend to establish the main topic)
    No pretrained models -- fully from scratch.
    """
    # 1. Tokenize into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
    if len(sentences) <= num_sentences:
        return ' '.join(sentences)

    # 2. Filter very short sentences (< 8 words) as they're usually filler
    MIN_WORDS = 8
    valid_indices = [i for i, s in enumerate(sentences)
                     if len(re.findall(r'[a-z]+', s.lower())) >= MIN_WORDS]
    # Fall back to all sentences if filtering is too aggressive
    if len(valid_indices) < num_sentences:
        valid_indices = list(range(len(sentences)))

    def content_words(s):
        """Extract meaningful content words (no stopwords)."""
        return [w for w in re.findall(r'[a-z]+', s.lower())
                if w not in _STOPWORDS and len(w) > 2]

    # 3. Compute sublinear TF per sentence (log(1 + count))
    tf_raw = []
    for s in sentences:
        w = content_words(s)
        counts = {}
        for word in w:
            counts[word] = counts.get(word, 0) + 1
        # Sublinear TF: log(1 + count) / len(sentence in words)
        n = max(len(re.findall(r'[a-z]+', s.lower())), 1)
        tf_raw.append({k: math.log(1 + v) / n for k, v in counts.items()})

    # 4. Compute IDF across all sentences
    N = len(sentences)
    df_counts = {}
    for sent_tf in tf_raw:
        for word in sent_tf:
            df_counts[word] = df_counts.get(word, 0) + 1
    idf = {w: math.log((N + 1) / (1 + df)) for w, df in df_counts.items()}

    # 5. Score = SUM of TF-IDF over all content words in the sentence
    #    (not average -- longer, information-dense sentences score higher)
    scores = []
    for i, sent_tf in enumerate(tf_raw):
        if sent_tf:
            # Sum-based score
            score = sum(sent_tf[w] * idf.get(w, 0) for w in sent_tf)
            # Position boost: reward earlier sentences slightly
            # Sentences in the first 30% of the transcript get +10% bonus
            position_ratio = i / N
            if position_ratio < 0.30:
                score *= 1.10
        else:
            score = 0.0
        scores.append(score)

    # 6. Only consider valid (long enough) sentences, pick top-n
    #    sorted by position to preserve reading order
    valid_scores = [(scores[i], i) for i in valid_indices]
    valid_scores.sort(key=lambda x: x[0], reverse=True)
    top_indices = sorted([idx for _, idx in valid_scores[:num_sentences]])
    return ' '.join(sentences[i] for i in top_indices)



class SummarizationDataset(Dataset):
    def __init__(self, transcripts_csv, summaries_csv, tokenizer, max_src_len=512, max_tgt_len=150):
        """
        Loads the actual ASR transcripts and builds extractive target summaries.
        """
        self.tokenizer = tokenizer
        self.max_src = max_src_len
        self.max_tgt = max_tgt_len
        
        self.samples = []
        
        # Load the CSV data
        df_trans = pd.read_csv(transcripts_csv)
        
        # Clean up appended headers from inner rows
        df_trans = df_trans[df_trans['start_time'] != 'start_time'].copy()
        
        # Group by video file
        grouped_trans = df_trans.groupby('file')
        
        for file_name, group in grouped_trans:
            # Sort by time to ensure chronological transcript
            group = group.sort_values(by='start_time')
            
            # The full source text is the entire transcript joined
            src_text = " ".join(group['text'].astype(str).tolist())
            
            # Build an extractive 5-sentence summary using our custom TF-IDF scorer
            tgt_text = extractive_summary(src_text, num_sentences=5)
            
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
        max_src_len=256, 
        max_tgt_len=150
    )
    print(f"[*] Loaded Actual Transcript Dataset. Size: {len(dataset)}")
    
    sample = dataset[0]
    print(f"[*] Sample 0 Source Ids Shape (paddded): {sample['src_input_ids'].shape}")
    print(f"[*] Sample 0 Target Ids Shape (paddded): {sample['tgt_input_ids'].shape}")
    
    print("\n[+] Phase 2 functional verification complete.")
