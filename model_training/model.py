import torch
import torch.nn as nn
import math

# -------------------------------------------------------------
# PHASE 3: CUSTOM TRANSFORMER ARCHITECTURE
# -------------------------------------------------------------
# Every piece of the model is implemented here from scratch:
# Attention, Positional Encoding, Encoders, Decoders.
# -------------------------------------------------------------

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        attn = torch.softmax(scores, dim=-1)
        output = torch.matmul(attn, V)
        return output, attn
        
    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)
        
        Q = self.W_q(Q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(K).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(V).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        x, attn = self.scaled_dot_product_attention(Q, K, V, mask)
        
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.W_o(x)

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        return self.linear2(self.dropout(self.relu(self.linear1(x))))

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, x, mask):
        attn_out = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout1(attn_out))
        ff_out = self.feed_forward(x)
        x = self.norm2(x + self.dropout2(ff_out))
        return x

class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        
    def forward(self, x, enc_out, src_mask, tgt_mask):
        # 1. Masked Self-Attention (Cannot look ahead at future words)
        attn1 = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout1(attn1))
        
        # 2. Cross-Attention (Look at encoder output/source transcript)
        attn2 = self.cross_attn(x, enc_out, enc_out, src_mask)
        x = self.norm2(x + self.dropout2(attn2))
        
        # 3. Feed Forward
        ff_out = self.feed_forward(x)
        x = self.norm3(x + self.dropout3(ff_out))
        return x

class Seq2SeqTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=256, num_heads=8, num_layers=3, d_ff=1024, max_seq_len=512, dropout=0.1):
        super().__init__()
        
        # Custom Embeddings learned from scratch
        self.src_emb = nn.Embedding(vocab_size, d_model)
        self.tgt_emb = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=max_seq_len)
        
        # Stacks
        self.encoder_layers = nn.ModuleList([TransformerEncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([TransformerDecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        
        # Generator back to vocabulary probability distribution
        self.generator = nn.Linear(d_model, vocab_size)
        self.d_model = d_model
        
    def generate_square_subsequent_mask(self, sz):
        """Prevents decoder from looking into the future"""
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        # For our custom Attention, we expect 0 to be blocked, 1 to be allowed:
        custom_mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1).long()
        return custom_mask
        
    def forward(self, src, tgt, src_pad_mask=None, tgt_pad_mask=None):
        """
        src: [batch_size, src_len]
        tgt: [batch_size, tgt_len]
        """
        # (Assuming src is shape [batch, len])
        # We need to reshape for PyTorch convention if desired, but our 
        # custom MultiHeadAttention currently assumes [batch, len, dim]
        
        # Embeddings
        src_emb = self.pos_encoder(self.src_emb(src) * math.sqrt(self.d_model))
        tgt_emb = self.pos_encoder(self.tgt_emb(tgt) * math.sqrt(self.d_model))
        
        # TGT Mask (No look-ahead)
        tgt_seq_len = tgt.size(1)
        tgt_mask = self.generate_square_subsequent_mask(tgt_seq_len).to(tgt.device)
        
        # If pad masks provided, combine them
        if tgt_pad_mask is not None:
            if tgt_pad_mask.dim() == 2:
                tgt_pad_mask = tgt_pad_mask.unsqueeze(1).unsqueeze(2)
            tgt_mask = tgt_mask.unsqueeze(0).unsqueeze(1).bool() & tgt_pad_mask.bool()
            
        if src_pad_mask is not None and src_pad_mask.dim() == 2:
            src_pad_mask = src_pad_mask.unsqueeze(1).unsqueeze(2)

        # Encoder
        enc_out = src_emb
        for layer in self.encoder_layers:
            enc_out = layer(enc_out, src_pad_mask)
            
        # Decoder
        dec_out = tgt_emb
        for layer in self.decoder_layers:
            dec_out = layer(dec_out, enc_out, src_pad_mask, tgt_mask)
            
        output = self.generator(dec_out)
        return output

if __name__ == "__main__":
    print("=========================================")
    print(" NLP PHASE 3: Custom Seq2Seq Transformer")
    print("=========================================")
    
    # Simple test of the architecture
    dummy_vocab = 500
    model = Seq2SeqTransformer(vocab_size=dummy_vocab, d_model=128, num_heads=4, num_layers=2)
    
    # Dummy Tensors: [batch_size, seq_len]
    src = torch.randint(0, dummy_vocab, (2, 32)) 
    tgt = torch.randint(0, dummy_vocab, (2, 16))
    
    out = model(src, tgt)
    
    print(f"[*] Custom Model Instantiated successfully.")
    print(f"[*] Batch Input shapes  -> src: {src.shape}, tgt: {tgt.shape}")
    print(f"[*] Batch Output shape -> generator out: {out.shape} (Expected: batch, tgt_len, vocab_size)")
    print("\n[+] Phase 3 architecture verification complete.")
