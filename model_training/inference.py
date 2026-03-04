import os
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# -------------------------------------------------------------
# PHASE 5: ADVANCED INFERENCE ENGINE (HUGGINGFACE)
# -------------------------------------------------------------
# This script loads an advanced pretrained Abstractive Summarization 
# framework (DistilBART) trained on CNN/DailyMail to generate coherent 
# multi-line human-readable summaries dynamically.
# -------------------------------------------------------------

def main():
    print("=========================================")
    print(" NLP PHASE 5: Abstractive Summarization Inference")
    print("=========================================")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Loading HuggingFace Pretrained Summarization Pipeline (Device: {device})...")
    
    # Load model and tokenizer directly to avoid pipeline alias deprecation issues
    model_name = "sshleifer/distilbart-cnn-12-6"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    
    transcripts_path = "./results/transcripts.csv"
    output_path = "./results/model_summaries.csv"
    
    if os.path.exists(transcripts_path):
        df = pd.read_csv(transcripts_path)
        df_clean = df[df['start_time'] != 'start_time']
        grouped = df_clean.groupby('file')
        
        results = []
        print("[*] Generating 5-6 sentence abstractive summaries for all pipeline videos...")
        for file_name, group in grouped:
            group = group.sort_values(by='start_time')
            src_text = " ".join(group['text'].astype(str).tolist())
            
            # Constrain to DistilBART maximums (~1024 tokens)
            src_text = src_text[:4000] 
            
            print(f"\n--- {file_name} ---")
            print(f"[*] Processing {len(src_text.split())} words of context...")
            
            input_length = len(src_text.split())
            if input_length < 40:
                print("[!] Transcript too short for extensive summarization. Returning raw text.")
                summary_text = src_text
            else:
                # DistilBART parameters scaled to generate ~4 to 6 lines of narrative summary
                # (100 to 150 Tokens roughly translates to 4 to 6 long sentences).
                max_len = min(150, int(input_length * 0.8))
                min_len = min(100, int(input_length * 0.3))
                
                inputs = tokenizer(src_text, return_tensors="pt", max_length=1024, truncation=True).to(device)
                summary_ids = model.generate(
                    inputs["input_ids"], 
                    num_beams=4, 
                    min_length=min_len, 
                    max_length=max_len, 
                    early_stopping=True
                )
                summary_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            
            results.append({
                "file": file_name,
                "generated_summary": summary_text
            })
            
        summary_df = pd.DataFrame(results)
        summary_df.to_csv(output_path, index=False)
        print(f"\n[+] Saved advanced generated summaries to: {output_path}")
        print("[+] Phase 5 executed successfully.")
    else:
        print("[!] Transcripts dataset not found. Please ensure Phase 2 has completed.")

if __name__ == "__main__":
    main()
