import torch
from torch.utils.data import Dataset

PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 3
MAX_LENGTH = 20


class Vocabulary:
    def __init__(self):
        self.word2index = {}
        self.index2word = {0: "PAD", 1: "SOS", 2: "EOS", 3: "UNK"}

    @property
    def vocab_size(self):
        return len(self.index2word)

    def add_sentence(self, sentence):
        for word in sentence:
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word2index:
            current_vocab_size = self.vocab_size
            self.word2index[word] = current_vocab_size
            self.index2word[current_vocab_size] = word

    def string2tensor(self, word, add_specials=False):

        ix = [
            self.word2index[x] if x in self.word2index else UNK_IDX for x in word
        ]
        if add_specials:
            ix = [SOS_IDX] + ix + [EOS_IDX]
        return torch.tensor(ix).long()

    def tensor2tokens(self, t):
        ix = []
        for i in t.tolist():
            if i != 1 and i != 2:
                ix.append(self.index2word[i])
        return ix

    def tensor2string(self, t):
        tokens = self.tensor2tokens(t)
        return "".join(tokens)


class Seq2SeqDataset(Dataset):
    """
    src_tokenizer and tgt_tokenizer defaults are designed with phoneme-to-grapheme
    in mind.
    """
    def __init__(
        self,
        data_tsv,
        src_vocab=None,
        tgt_vocab=None,
        src_tokenizer=str.split,
        tgt_tokenizer=list,
        max_length=None
    ):
        self.pairs = read_tsv_corpus(
            data_tsv,
            max_length=max_length,
            src_tokenizer=src_tokenizer,
            tgt_tokenizer=tgt_tokenizer
        )

        if src_vocab is not None:
            self.src_vocab = src_vocab
        else:
            self.src_vocab = Vocabulary()
            for p in self.pairs:
                self.src_vocab.add_sentence(p[0])

        if tgt_vocab is not None:
            self.tgt_vocab = tgt_vocab
        else:
            self.tgt_vocab = Vocabulary()
            for p in self.pairs:
                self.tgt_vocab.add_sentence(p[1])

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        src = self.pairs[idx][0]
        tgt = self.pairs[idx][1]

        src_ix = self.src_vocab.string2tensor(src)
        tgt_ix = self.tgt_vocab.string2tensor(tgt, add_specials=True)
        return src_ix, tgt_ix


def read_tsv_corpus(path, max_length=None, src_tokenizer=list, tgt_tokenizer=list):
    # previous function readLangs takes two language labels and a split and
    # returns two vocab objects with nothing in them and a set of pairs
    # This one returns just a set of pairs based on the path

    pairs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            src, tgt = line.strip().split("\t")[:2]
            # todo: handle max length
            src_toks = src_tokenizer(src)
            tgt_toks = tgt_tokenizer(tgt)
            if max_length is not None and (len(src_toks) > max_length or len(tgt_toks) > max_length):
                continue
            pairs.append((src_toks, tgt_toks))

    return pairs


def collate_samples(samples, padding_idx):
    batch_size = len(samples)
    max_seq_length_x = max([x.shape[0] for x, _ in samples])
    max_seq_length_y = max([y.shape[0] for _, y in samples])
    X_shape = (batch_size, max_seq_length_x)
    y_shape = (batch_size, max_seq_length_y)
    X = torch.zeros(X_shape, dtype=torch.long).fill_(padding_idx)
    Y = torch.zeros(y_shape, dtype=torch.long).fill_(padding_idx)
    for i, (x, y) in enumerate(samples):
        seq_len_x = x.shape[0]
        seq_len_y = y.shape[0]
        X[i, :seq_len_x] = x
        Y[i, :seq_len_y] = y
    return X, Y
