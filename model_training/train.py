import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import BPETokenizerInference, SummarizationDataset
from model import Seq2SeqTransformer

# -------------------------------------------------------------
# PHASE 4: TRAINING LOOP
# -------------------------------------------------------------
# Trains the model from scratch on the loaded dataset using
# Cross Entropy Loss and Teacher Forcing strategies.
# -------------------------------------------------------------

def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    
    for batch_idx, batch in enumerate(dataloader):
        src = batch["src_input_ids"].to(device)
        tgt = batch["tgt_input_ids"].to(device)
        
        # Teacher Forcing: We feed the decoder everything EXCEPT the last token
        # And we calculate loss against everything EXCEPT the first token (<BOS>)
        tgt_input = tgt[:, :-1]
        tgt_expected = tgt[:, 1:]
        
        # We need padding masks so attention ignores <PAD> tokens
        # (Assuming PAD_ID = 256 based on our tokenizer)
        PAD_ID = 256
        src_pad_mask = (src != PAD_ID).unsqueeze(1).unsqueeze(2) # [batch, 1, 1, src_len]
        tgt_pad_mask = (tgt_input != PAD_ID) # [batch, tgt_len]
        
        optimizer.zero_grad()
        
        # Forward pass
        output = model(src, tgt_input, src_pad_mask=src_pad_mask, tgt_pad_mask=tgt_pad_mask)
        
        # Calculate loss (ignoring PAD tokens)
        # Output shape: [batch, tgt_len, vocab_size]
        # Expected shape: [batch, tgt_len]
        output = output.reshape(-1, output.shape[-1])
        tgt_expected = tgt_expected.reshape(-1)
        
        loss = criterion(output, tgt_expected)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients in Transformers
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        total_loss += loss.item()
        
        if batch_idx % 10 == 0:
            print(f"   [Batch {batch_idx}/{len(dataloader)}] Loss: {loss.item():.4f}")
            
    return total_loss / len(dataloader)

def main():
    print("=========================================")
    print(" NLP PHASE 4: Transformer Training")
    print("=========================================")
    
    # 1. Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Proceeding with device: {device}")
    
    # 2. Load Tokenizer & Dataset
    tokenizer = BPETokenizerInference("./model_weights")
    dataset = SummarizationDataset(
        transcripts_csv="./results/transcripts.csv", 
        summaries_csv="./results/final_summary.csv", 
        tokenizer=tokenizer, 
        max_src_len=32, 
        max_tgt_len=16
    )
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    # 3. Instantiate Model Architecture (Empty weights)
    print("[*] Initializing empty Transformer Architecture...")
    model = Seq2SeqTransformer(
        vocab_size=tokenizer.vocab_size, 
        d_model=128, 
        num_heads=4, 
        num_layers=2, 
        d_ff=512,
        max_seq_len=64, # Matches max lengths
        dropout=0.1
    ).to(device)
    
    # 4. Setup Optimization Loop
    PAD_ID = 256
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    
    epochs = 5
    print(f"[*] Commencing Training for {epochs} Epochs...")
    
    for epoch in range(1, epochs + 1):
        print(f"\n--- Epoch {epoch} ---")
        avg_loss = train_epoch(model, dataloader, optimizer, criterion, device)
        print(f"[*] Epoch {epoch} Completed. Average Loss = {avg_loss:.4f}")
        
    # 5. Save the compiled model weights
    save_path = "./model_weights/custom_summarizer.pt"
    torch.save(model.state_dict(), save_path)
    print(f"\n[+] Training Complete. Weights saved to: {save_path}")

if __name__ == "__main__":
    main()
