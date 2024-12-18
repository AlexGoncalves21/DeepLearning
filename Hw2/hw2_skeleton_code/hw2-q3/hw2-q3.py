import argparse
import random
from functools import partial
from os.path import join

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

from data import collate_samples, Seq2SeqDataset, PAD_IDX, SOS_IDX, EOS_IDX
from models import Encoder, Decoder, Seq2Seq, BahdanauAttention, reshape_state

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def configure_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True


def distance(str1, str2):
    """Simple Levenshtein implementation for evalm."""
    m = np.zeros([len(str2) + 1, len(str1) + 1], dtype=int)
    for x in range(1, len(str2) + 1):
        m[x, 0] = m[x - 1, 0] + 1
    for y in range(1, len(str1) + 1):
        m[0, y] = m[0, y - 1] + 1
    for x in range(1, len(str2) + 1):
        for y in range(1, len(str1) + 1):
            if str1[y - 1] == str2[x - 1]:
                dg = 0
            else:
                dg = 1
            m[x, y] = min(
                m[x - 1, y] + 1, m[x, y - 1] + 1, m[x - 1, y - 1] + dg
            )
    return m[len(str2), len(str1)]


def train(data, model, lr, n_epochs, checkpoint_name, max_len=50):
    model.train()

    train_iter, val_iter, test_iter = data

    # Loss and Optimizer
    # Softmax is internally computed.
    # Set parameters to be updated.
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    val_err_rates = []
    min_err_rate = float("inf")

    # Training the model
    for epoch in range(n_epochs):
        model.train()
        for src, tgt in train_iter:
            src_lengths = (src != PAD_IDX).sum(1)
            src, tgt = src.to(device), tgt.to(device)
            src_lengths = src_lengths.to(device)

            optimizer.zero_grad()
            outputs, _ = model(src, src_lengths, tgt)
            loss = criterion(
                outputs.reshape(-1, outputs.shape[-1]), tgt[:, 1:].reshape(-1)
            )
            loss.backward()
            optimizer.step()

        print("Epoch: [%d/%d], Loss: %.4f" % (epoch + 1, n_epochs, loss))

        # validation is always greedy
        val_err_rate, _ = test(model, val_iter, max_len=max_len)
        print("Validation error rate: %.4f" % (val_err_rate))

        if val_err_rate < min_err_rate:
            min_err_rate = val_err_rate
            print("New best error rate found: {:.4f}".format(min_err_rate))
            print("Saving model")
            torch.save(model.state_dict(), checkpoint_name)

        val_err_rates.append(val_err_rate)

    return min_err_rate, val_err_rates


def generate(model, data_iter, max_len=50, p=None):
    # Test the Model
    assert data_iter.batch_size == 1

    model.eval()

    if p is None:
        next_token_func = greedy_next_token
    else:
        next_token_func = partial(nucleus_sampling, p=p)

    predictions = []
    with torch.no_grad():
        for src, tgt in data_iter:
            src_lengths = (src != PAD_IDX).sum(1)
            src, tgt = src.to(device), tgt.to(device)
            src_lengths = src_lengths.to(device)

            # Initially, the generated sequence consists of only a start-of-sequence
            # token. Each subsequent generated token will be appended to this sequence.
            sos_token = torch.full(
                [data_iter.batch_size, 1],
                SOS_IDX,
                dtype=torch.long,
                device=device,
            )
            predicted_sequence = [sos_token]

            encoder_outputs, final_enc_state = model.encoder(src, src_lengths)
            dec_state = final_enc_state

            if dec_state[0].shape[0] == 2:
                dec_state = reshape_state(dec_state)

            for _ in range(max_len):
                prev_token = predicted_sequence[-1]

                output, dec_state = model.decoder(
                    prev_token, dec_state, encoder_outputs, src_lengths
                )
                logits = model.generator(output.view(-1))

                next_token = next_token_func(logits)

                predicted_sequence.append(next_token.view(1, 1))

                # Stop symbol index
                if int(next_token) == EOS_IDX:
                    break

            predicted_sequence = torch.cat(predicted_sequence).squeeze()
            predictions.append(predicted_sequence)

    return [data_iter.dataset.tgt_vocab.tensor2string(p) for p in predictions]


def evaluate(predictions, gold_data_iter):
    """
    Return
        - character error rate (the levenshtein distance normalized by the true length)
        - word error rate (rate of examples where the prediction does not exactly match the true target)
    """
    gold_len = 0
    total_distance = 0
    incorrect = 0
    for pred, (src, tgt) in zip(predictions, gold_data_iter.dataset.pairs):
        tgt = "".join(tgt)

        d_list = distance(tgt, pred)

        total_distance += d_list
        gold_len += len(tgt)
        if tgt != pred:
            incorrect += 1

    cer = total_distance / gold_len
    wer = incorrect / len(predictions)
    return cer, wer


def test(model, data_iter, max_len=50, p=None):
    predictions = generate(model, data_iter, max_len=max_len, p=p)
    cer, wer = evaluate(predictions, data_iter)
    return cer, wer


def compute_wer_at_k(model, gold_data_iter, max_len=50, p=None, k=1, ex_to_print=10):
    multipreds = []
    true_targets = []
    for i in range(k):
        predictions = generate(model, gold_data_iter, max_len=max_len, p=p)
        multipreds.append(predictions)
    pred_sets = [set(p) for p in zip(*multipreds)]

    incorrect = 0
    for pred_set, (src, tgt) in zip(pred_sets, gold_data_iter.dataset.pairs):
        tgt = "".join(tgt)
        true_targets.append(tgt)

        if tgt not in pred_set:
            incorrect += 1

    wer_at_k = incorrect / len(predictions)
    examples = [(t, p) for t, p in zip(true_targets, pred_sets) if len(p) > 1]
    if ex_to_print > 0:
        print(f"Printing first {ex_to_print} examples with multiple predictions:")
    for ex in examples[: ex_to_print]:
        print(ex)
    return wer_at_k


def greedy_next_token(logits):
    """
    Performs greedy decoding.
    logits: 1d tensor of unnormalized scores for each class. Shape: (vocab_size,)

    Returns:
        next_token: index of the next predicted token. Shape: (1,)
    """
    return logits.argmax(-1, keepdim=True)


def nucleus_sampling(logits, p=0.8):
    """
    Performs nucleus (top-p) sampling
    logits: 1d tensor of unnormalized scores for each class. Shape: (vocab_size,)
    p: Cumulative probability threshold to be used. (scalar)

    Returns:
        next_token: index of the next predicted token. Shape: (1,)
    """
    # TODO: Top-p (nucleus) sampling  (https://arxiv.org/pdf/1904.09751 - Section 3.1)
    # You are asked to implement the following steps:
    # 1. Transform the given logits into probabilities.
    # 2. Select the smallest set of tokens whose cumulative probability mass exceeds p.
    # This is equivalent to selecting the tokens with highest probabilities, whose cumulative probability mass equals or exceeds p.
    # 3. Rescale the distribution and sample from the resulting set of tokens.
    # Implementation of the steps as described above:

    raise NotImplementedError("Add your implementation.")


def main(args):

    configure_seed(args.seed)

    print("Loading data...")
    train_dataset = Seq2SeqDataset(join(args.data_dir, "train.tsv"))
    valid_dataset = Seq2SeqDataset(
        join(args.data_dir, "valid.tsv"),
        src_vocab=train_dataset.src_vocab,
        tgt_vocab=train_dataset.tgt_vocab
    )
    test_dataset = Seq2SeqDataset(
        join(args.data_dir, "test.tsv"),
        src_vocab=train_dataset.src_vocab,
        tgt_vocab=train_dataset.tgt_vocab
    )

    collate_fn = partial(collate_samples, padding_idx=PAD_IDX)

    train_iter = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    val_iter = DataLoader(valid_dataset, batch_size=1, shuffle=False)
    test_iter = DataLoader(test_dataset, batch_size=1, shuffle=False)

    data_iters = (train_iter, val_iter, test_iter)

    src_vocab_size = train_dataset.src_vocab.vocab_size
    tgt_vocab_size = train_dataset.tgt_vocab.vocab_size

    encoder = Encoder(
        src_vocab_size,
        args.hidden_size,
        PAD_IDX,
        args.dropout,
    )

    if args.use_attn:
        attn = BahdanauAttention(args.hidden_size)
    else:
        attn = None

    decoder = Decoder(
        args.hidden_size,
        tgt_vocab_size,
        attn,
        PAD_IDX,
        args.dropout,
    )

    model = Seq2Seq(encoder, decoder).to(device)

    if args.checkpoint_name is not None:
        checkpoint_name = args.checkpoint_name
    elif args.use_attn:
        checkpoint_name = "model-attn.pt"
    else:
        checkpoint_name = "model.pt"

    if args.mode == "train":
        print("Training...")
        min_val_err, val_errs = train(
            data_iters,
            model,
            args.lr,
            args.n_epochs,
            checkpoint_name
        )

        print("Best validation error rate: %.4f" % (min_val_err))
        plt.plot(np.arange(1, args.n_epochs + 1), val_errs, label="Validation Set")

        plt.xticks(np.arange(0, args.n_epochs + 1, step=2))
        plt.grid(True)
        plt.xlabel("Epochs")
        plt.ylabel("Error Rate")
        plt.legend()
        plt.savefig(
            "attn_%s_err_rate.pdf" % (str(args.use_attn),),
            bbox_inches="tight",
        )
    else:
        print("Testing...")

        model.load_state_dict(torch.load(checkpoint_name, weights_only=True))

        test_cer, test_wer = test(model, test_iter, p=args.topp)
        print("Test CER: %.4f, Test WER: %.4f" % (test_cer, test_wer))

        if args.topp is not None:
            test_wer_at_k = compute_wer_at_k(model, test_iter, p=args.topp, k=args.k)
            print("Test WER@{}: {:.4f}".format(args.k, test_wer_at_k))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("mode", choices=["train", "test"])
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--n_epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_attn", action="store_true")
    parser.add_argument("--data_dir", default="data/")
    parser.add_argument("--checkpoint_name", default=None)
    parser.add_argument("--topp", type=float, default=None)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--max_len", type=int, default=50)
    args = parser.parse_args()
    main(args)
