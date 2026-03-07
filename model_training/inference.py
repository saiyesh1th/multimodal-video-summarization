import os
import torch

from dataset import BPETokenizerInference
from model import Seq2SeqTransformer

# -------------------------------------------------------------
# PHASE 5: AUTOREGRESSIVE INFERENCE ENGINE
# -------------------------------------------------------------
# This script loads the trained custom Transformer weights 
# and runs the autoregressive loop to generate a summary
# word-by-word without the use of teacher forcing.
# -------------------------------------------------------------

def generate_summary(model, tokenizer, text, device, max_len=200, min_len=60):
    model.eval()
    
    # 1. Encode Source Text
    src_ids = tokenizer.encode(text)
    src_ids = [tokenizer.BOS_ID] + src_ids[:256-2] + [tokenizer.EOS_ID]
    src_tensor = torch.tensor(src_ids, dtype=torch.long).unsqueeze(0).to(device) # [1, src_len]
    
    # Create source mask
    PAD_ID = 256
    src_pad_mask = (src_tensor != PAD_ID).unsqueeze(1).unsqueeze(2)
    
    # 2. Start Decoder with BOS token
    tgt_ids = [tokenizer.BOS_ID]
    tgt_tensor = torch.tensor(tgt_ids, dtype=torch.long).unsqueeze(0).to(device) # [1, 1]
    
    print(f"[*] Generating summary for (truncated): '{text[:60]}...'")
    
    with torch.no_grad():
        for step in range(max_len):
            # Forward pass
            tgt_pad_mask = (tgt_tensor != PAD_ID)
            output = model(src_tensor, tgt_tensor, src_pad_mask=src_pad_mask, tgt_pad_mask=tgt_pad_mask)
            
            # Get the prediction for the last token position
            next_token_logits = output[0, -1, :] # [vocab_size]
            
            # Apply repetition penalty for the last few tokens
            if len(tgt_ids) > 1:
                for prev_tok in set(tgt_ids[-8:]):
                    next_token_logits[prev_tok] -= 2.0
            
            # Suppress EOS before minimum length is reached
            if step < min_len:
                next_token_logits[tokenizer.EOS_ID] = -1e9
                
            temperature = 0.9
            probs = torch.nn.functional.softmax(next_token_logits / temperature, dim=-1)
            
            # Top-K Sampling (K=20)
            top_k = 20
            top_k_probs, top_k_indices = torch.topk(probs, top_k)
            sampled_idx = torch.multinomial(top_k_probs, 1).item()
            next_token_id = top_k_indices[sampled_idx].item()
            
            # Append predicted token
            tgt_ids.append(next_token_id)
            tgt_tensor = torch.tensor(tgt_ids, dtype=torch.long).unsqueeze(0).to(device)
            
            # Stop if EOS is predicted (only after min_len)
            if next_token_id == tokenizer.EOS_ID and step >= min_len:
                break
                
    # 3. Decode the generated IDs back to text
    decoded_text = tokenizer.decode(tgt_ids)
    print(f"\n[+] Generated Token Sequence: {tgt_ids}")
    print(f"[+] Predicted Summary: '{decoded_text}'")
    return tgt_ids

def main():
    print("=========================================")
    print(" NLP PHASE 5: Autoregressive Inference")
    print("=========================================")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Tokenizer
    tokenizer = BPETokenizerInference("./model_weights")
    
    # 2. Load Model Architecture
    model = Seq2SeqTransformer(
        vocab_size=tokenizer.vocab_size, 
        d_model=128, 
        num_heads=4, 
        num_layers=2, 
        d_ff=512,
        max_seq_len=512,
        dropout=0.1
    ).to(device)
    
    # 3. Load Trained Weights
    weights_path = "./model_weights/custom_summarizer.pt"
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device))
        print(f"[*] Loaded trained weights from {weights_path}")
    else:
        print("[!] Warning: Trained weights not found. Using random initialization.")
        
    # 4. Generate over actual video transcripts
    transcripts_path = "./results/transcripts.csv"
    output_path = "./results/model_summaries.csv"
    
    if os.path.exists(transcripts_path):
        import pandas as pd
        df = pd.read_csv(transcripts_path)
        df_clean = df[df['start_time'] != 'start_time']
        grouped = df_clean.groupby('file')
        
        results = []
        print("[*] Generating summaries for all pipeline videos...")
        for file_name, group in grouped:
            # Sort by time to ensure chronological transcript
            group = group.sort_values(by='start_time')
            src_text = " ".join(group['text'].astype(str).tolist())
            
            # Predict
            print(f"\n--- {file_name} ---")
            tgt_ids = generate_summary(model, tokenizer, src_text, device)
            decoded_text = tokenizer.decode(tgt_ids)
            
            results.append({
                "file": file_name,
                "generated_summary": decoded_text
            })
            
        summary_df = pd.DataFrame(results)
        summary_df.to_csv(output_path, index=False)
        print(f"\n[+] Saved generated summaries to: {output_path}")
    else:
        print("[!] Transcripts dataset not found. Running on fallback sample.")
        sample_text = "hello world . this is a test transcript ."
        generate_summary(model, tokenizer, sample_text, device)

if __name__ == "__main__":
    main()
