import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=10):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x, iter_idx):
        # x is the input tensor, iter_idx is the current iteration number
        # self.pe is (1, max_len, d_model)
        # We select the positional encoding for the current iteration and add it to x
        return x + self.pe[:, iter_idx, :].unsqueeze(1) # Add to sequence dimension

class SVDTransformer(nn.Module):
    def __init__(self, input_dim=64, embed_dim=64, rank=32, nhead=8, num_encoder_layers=6, dim_feedforward=256, dropout=0.1, max_iters=4):
        super(SVDTransformer, self).__init__()
        self.input_dim = input_dim
        self.rank = rank
        self.embed_dim = embed_dim
        self.max_iters = max_iters

        # 1. Input Encoder
        # Encodes each of the 64 rows (64x2) of the input matrix into a token.
        self.h_encoder = nn.Linear(input_dim * 2, embed_dim)

        # 2. SVD component embeddings
        self.u_embed = nn.Linear(input_dim * 2, embed_dim) # U row is (64, 2) -> 128
        self.v_embed = nn.Linear(input_dim * 2, embed_dim) # V row is (64, 2) -> 128
        self.s_embed = nn.Linear(rank, embed_dim) # S vector is (32) -> embed_dim

        # 3. Positional Encoding for iterations
        self.pos_encoder = PositionalEncoding(embed_dim, max_len=max_iters)

        # 4. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # 5. Output Decoders
        self.u_decoder = nn.Linear(embed_dim, input_dim * 2)
        self.v_decoder = nn.Linear(embed_dim, input_dim * 2)
        self.s_decoder = nn.Linear(embed_dim, rank)

        # 6. Learnable initial parameters for the first iteration
        self.initial_U = nn.Parameter(torch.randn(1, rank, input_dim, 2))
        self.initial_V = nn.Parameter(torch.randn(1, rank, input_dim, 2))
        self.initial_S = nn.Parameter(torch.randn(1, rank))

    def forward(self, x, iterations=4):
        # x: (B, 64, 64, 2)
        batch_size = x.shape[0]

        # Encode the input matrix H
        # Reshape H to treat each of the 64 rows as a sample to be encoded.
        # x -> (B, 64, 64, 2) -> h_reshaped (B, 64, 128)
        h_reshaped = x.view(batch_size, self.input_dim, self.input_dim * 2)
        encoded_h = self.h_encoder(h_reshaped) # (B, 64, embed_dim)

        # Initialize SVD components
        pred_U = self.initial_U.expand(batch_size, -1, -1, -1)
        pred_V = self.initial_V.expand(batch_size, -1, -1, -1)
        pred_S = self.initial_S.expand(batch_size, -1)

        # Iterative refinement
        for i in range(iterations):
            # --- Tokenization of current SVD estimates ---
            # pred_U is (B, 32, 64, 2) -> flatten to (B, 32, 128)
            u_flat = pred_U.flatten(start_dim=2)
            u_tokens = self.u_embed(u_flat) # (B, 32, embed_dim)

            # pred_V is (B, 32, 64, 2) -> flatten to (B, 32, 128)
            v_flat = pred_V.flatten(start_dim=2)
            v_tokens = self.v_embed(v_flat) # (B, 32, embed_dim)

            # pred_S is (B, 32)
            s_tokens = self.s_embed(pred_S).unsqueeze(1) # (B, 1, embed_dim)

            # --- Token Concatenation and Transformer Pass ---
            # Total tokens: 64 (from H) + 32 (from U) + 32 (from V) + 1 (from S) = 129
            tokens = torch.cat([encoded_h, u_tokens, v_tokens, s_tokens], dim=1)

            # Add positional encoding for the current iteration
            tokens = self.pos_encoder(tokens, i)

            # Pass through transformer
            output_tokens = self.transformer_encoder(tokens)

            # --- Decoding and Residual Prediction ---
            # Split the output tokens back into their respective parts
            h_out_len, u_out_len, v_out_len = self.input_dim, self.rank, self.rank
            u_start, v_start = h_out_len, h_out_len + u_out_len
            
            u_out = output_tokens[:, u_start:v_start, :]
            v_out = output_tokens[:, v_start:v_start + v_out_len, :]
            s_out = output_tokens[:, -1, :] # Last token is for S

            # Decode to get the residual values
            delta_U_flat = self.u_decoder(u_out) # (B, 32, 128)
            delta_U = delta_U_flat.view(batch_size, self.rank, self.input_dim, 2)

            delta_V_flat = self.v_decoder(v_out)
            delta_V = delta_V_flat.view(batch_size, self.rank, self.input_dim, 2)

            delta_S = self.s_decoder(s_out) # (B, rank)

            # --- Update SVD components ---
            pred_U = pred_U + delta_U
            pred_V = pred_V + delta_V
            pred_S = pred_S + delta_S

        return pred_U, pred_S, pred_V