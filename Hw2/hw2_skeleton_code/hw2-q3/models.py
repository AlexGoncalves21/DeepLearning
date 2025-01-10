import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack


def reshape_state(state):
    h_state = state[0]
    c_state = state[1]
    new_h_state = torch.cat([h_state[:-1], h_state[1:]], dim=2)
    new_c_state = torch.cat([c_state[:-1], c_state[1:]], dim=2)
    return (new_h_state, new_c_state)


class BahdanauAttention(nn.Module):
    """
    Bahdanau attention mechanism:
    score(h_i, s_j) = v^T * tanh(W_h h_i + W_s s_j)
    """

    def __init__(self, hidden_size):
        super(BahdanauAttention, self).__init__()
        self.Wh = nn.Linear(hidden_size, hidden_size, bias=False)  # Encoder hidden state projection
        self.Ws = nn.Linear(hidden_size, hidden_size, bias=False)  # Decoder hidden state projection
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, query, encoder_outputs, src_lengths):
        """
        query:          (batch_size, max_tgt_len, hidden_size) - decoder hidden states
        encoder_outputs:(batch_size, max_src_len, hidden_size) - encoder hidden states
        src_lengths:    (batch_size) - lengths of source sequences
        Returns:
            attn_out:   (batch_size, max_tgt_len, hidden_size) - attended vector
        """
        # Expand query and encoder_outputs to shape compatibility
        query_proj = self.Ws(query).unsqueeze(2)  # (batch_size, max_tgt_len, 1, hidden_size)
        encoder_proj = self.Wh(encoder_outputs).unsqueeze(1)  # (batch_size, 1, max_src_len, hidden_size)
        
        # Compute alignment scores
        alignment_scores = self.v(torch.tanh(query_proj + encoder_proj)).squeeze(-1)  # (batch_size, max_tgt_len, max_src_len)

        # Create mask for valid source lengths
        mask = self.sequence_mask(src_lengths).unsqueeze(1)  # (batch_size, 1, max_src_len)
        alignment_scores = alignment_scores.masked_fill(~mask, float('-inf'))  # Mask padding positions

        # Compute attention weights
        attn_weights = torch.softmax(alignment_scores, dim=-1)  # (batch_size, max_tgt_len, max_src_len)

        # Compute context vectors as weighted sum of encoder outputs
        context = torch.bmm(attn_weights, encoder_outputs)  # (batch_size, max_tgt_len, hidden_size)

        return context

    def sequence_mask(self, lengths):
        """
        Creates a boolean mask from sequence lengths.
        True for valid positions, False for padding.
        """
        batch_size = lengths.numel()
        max_len = lengths.max()
        return (torch.arange(max_len, device=lengths.device)
                .unsqueeze(0)
                .repeat(batch_size, 1)
                .lt(lengths.unsqueeze(1)))


class Encoder(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        hidden_size,
        padding_idx,
        dropout,
    ):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size // 2
        self.dropout = dropout

        self.embedding = nn.Embedding(
            src_vocab_size,
            hidden_size,
            padding_idx=padding_idx,
        )
        self.lstm = nn.LSTM(
            hidden_size,
            self.hidden_size,
            bidirectional=True,
            batch_first=True,
        )
        self.dropout = nn.Dropout(self.dropout)

    def forward(
        self,
        src,
        lengths,
    ):
        # src: (batch_size, max_src_len)
        # lengths: (batch_size)
        #############################################
        # TODO: Implement the forward pass of the encoder
        # Hints:
        # - Use torch.nn.utils.rnn.pack_padded_sequence to pack the padded sequences
        #   (before passing them to the LSTM)
        # - Use torch.nn.utils.rnn.pad_packed_sequence to unpack the packed sequences
        #   (after passing them to the LSTM)
        #############################################
        
        # Convert token indices into embeddings
        embedded = self.embedding(src)  # Shape: (batch_size, max_src_len, hidden_size)
        
        # Pack padded sequences to ignore padding during LSTM processing
        packed_embedded = pack(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        
        # Pass through bidirectional LSTM
        packed_output, (hidden, cell) = self.lstm(packed_embedded)
        
        # Unpack the sequence
        output, _ = unpack(packed_output, batch_first=True)  # Shape: (batch_size, max_src_len, hidden_size)
        
        # Return encoder outputs and final hidden/cell states
        return output, (hidden, cell)

        #############################################
        # END OF YOUR CODE
        #############################################
        # enc_output: (batch_size, max_src_len, hidden_size)
        # final_hidden: tuple with 2 tensors
        # each tensor is (num_layers * num_directions, batch_size, hidden_size)
        


class Decoder(nn.Module):
    def __init__(
        self,
        hidden_size,
        tgt_vocab_size,
        attn,
        padding_idx,
        dropout,
    ):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.tgt_vocab_size = tgt_vocab_size
        self.dropout = dropout

        self.embedding = nn.Embedding(
            self.tgt_vocab_size, self.hidden_size, padding_idx=padding_idx
        )

        self.dropout = nn.Dropout(self.dropout)
        self.lstm = nn.LSTM(
            self.hidden_size,
            self.hidden_size,
            batch_first=True,
        )

        self.attn = attn

        if self.attn is not None:
            self.attn_proj = nn.Linear(self.hidden_size * 2, self.hidden_size)


    def forward(
        self,
        tgt,
        dec_state,
        encoder_outputs,
        src_lengths,
    ):
        # tgt: (batch_size, max_tgt_len)
        # dec_state: tuple with 2 tensors
        # each tensor is (num_layers * num_directions, batch_size, hidden_size)
        # encoder_outputs: (batch_size, max_src_len, hidden_size)
        # src_lengths: (batch_size)
        # bidirectional encoder outputs are concatenated, so we may need to
        # reshape the decoder states to be of size (num_layers, batch_size, 2*hidden_size)
        # if they are of size (num_layers*num_directions, batch_size, hidden_size)
        if dec_state[0].shape[0] == 2:
            dec_state = reshape_state(dec_state)

        #############################################
        # TODO: Implement the forward pass of the decoder
        # Hints:
        # - the input to the decoder is the previous target token,
        #   and the output is the next target token
        # - New token representations should be generated one at a time, given
        #   the previous token representation and the previous decoder state
        # - Add this somewhere in the decoder loop when you implement the attention mechanism in 3.2:
        # if self.attn is not None:
        #     output = self.attn(
        #         output,
        #         encoder_outputs,
        #         src_lengths,
        #     )
        #############################################
        
        # Convert target token indices to embeddings
       # Convert target token indices to embeddings
        embedded = self.dropout(self.embedding(tgt))  # (batch_size, max_tgt_len, embedding_dim)

        # Initialize a list to collect outputs at each time step
        outputs = []

        # Process each time step in the target sequence
        for t in range(embedded.size(1)):
            # Get the embedding for the current time step
            input_t = embedded[:, t, :].unsqueeze(1)  # (batch_size, 1, embedding_dim)

            # Pass through LSTM for the current step
            lstm_out, dec_state = self.lstm(input_t, dec_state)  # (batch_size, 1, hidden_size)

            if self.attn is not None:
                # Apply attention mechanism
                context = self.attn(lstm_out, encoder_outputs, src_lengths)  # (batch_size, 1, hidden_size)
                # Combine LSTM output and attention context
                lstm_out = torch.cat((context, lstm_out), dim=-1)  # (batch_size, 1, 2*hidden_size)
                
                # Dynamically add attn_proj if it doesn't exist
                if not hasattr(self, "attn_proj"):
                    self.attn_proj = nn.Linear(lstm_out.size(-1), self.hidden_size).to(lstm_out.device)
                
                lstm_out = torch.tanh(self.attn_proj(lstm_out))  # Project back to hidden_size

            outputs.append(lstm_out)

        # Concatenate all time step outputs
        outputs = torch.cat(outputs, dim=1)  # (batch_size, max_tgt_len, hidden_size)

        # Remove the first timestep output corresponding to <SOS>
        if outputs.shape[1] > 1:  # Ensure there's more than one timestep
            outputs = outputs[:, :-1, :]  # Remove the <SOS> token's output

        return outputs, dec_state


        #############################################
        # END OF YOUR CODE
        #############################################
        # outputs: (batch_size, max_tgt_len, hidden_size)
        # dec_state: tuple with 2 tensors
        # each tensor is (num_layers, batch_size, hidden_size)
    


class Seq2Seq(nn.Module):
    def __init__(
        self,
        encoder,
        decoder,
    ):
        super(Seq2Seq, self).__init__()

        self.encoder = encoder
        self.decoder = decoder

        self.generator = nn.Linear(decoder.hidden_size, decoder.tgt_vocab_size)

        self.generator.weight = self.decoder.embedding.weight

    def forward(
        self,
        src,
        src_lengths,
        tgt,
        dec_hidden=None,
    ):

        encoder_outputs, final_enc_state = self.encoder(src, src_lengths)

        if dec_hidden is None:
            dec_hidden = final_enc_state

        output, dec_hidden = self.decoder(
            tgt, dec_hidden, encoder_outputs, src_lengths
        )

        return self.generator(output), dec_hidden
