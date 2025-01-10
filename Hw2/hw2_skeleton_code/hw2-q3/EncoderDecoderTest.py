import torch
import torch.nn as nn
from models import Encoder, Decoder, reshape_state, BahdanauAttention

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Test parameters
BATCH_SIZE = 32
MAX_SRC_LEN = 20
MAX_TGT_LEN = 15
SRC_VOCAB_SIZE = 100
TGT_VOCAB_SIZE = 80
HIDDEN_SIZE = 128
PADDING_IDX = 0
DROPOUT = 0.3

# Dummy input data
src = torch.randint(1, SRC_VOCAB_SIZE, (BATCH_SIZE, MAX_SRC_LEN)).to(device)  # Random source sequences
tgt = torch.randint(1, TGT_VOCAB_SIZE, (BATCH_SIZE, MAX_TGT_LEN)).to(device)  # Random target sequences
lengths = torch.randint(5, MAX_SRC_LEN, (BATCH_SIZE,)).to(device)  # Random sequence lengths

# Initialize Encoder
encoder = Encoder(SRC_VOCAB_SIZE, HIDDEN_SIZE, PADDING_IDX, DROPOUT).to(device)

# Forward pass through the Encoder
try:
    encoder_outputs, final_hidden = encoder(src, lengths)
    print(f"Encoder outputs shape: {encoder_outputs.shape}")
    print(f"Encoder final hidden state shape: {final_hidden[0].shape}, {final_hidden[1].shape}")
except Exception as e:
    print(f"Encoder forward pass error: {e}")

# Initialize Decoder (without attention for simplicity)
decoder = Decoder(HIDDEN_SIZE, TGT_VOCAB_SIZE, None, PADDING_IDX, DROPOUT).to(device)

# Forward pass through the Decoder
try:
    dec_outputs, dec_state = decoder(tgt, final_hidden, encoder_outputs, lengths)
    print(f"Decoder outputs shape: {dec_outputs.shape}")
    print(f"Decoder final hidden state shape: {dec_state[0].shape}, {dec_state[1].shape}")
except Exception as e:
    print(f"Decoder forward pass error: {e}")
