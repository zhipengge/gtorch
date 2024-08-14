# -*- coding: utf-8 -*-
"""
@author: gehipeng @ 20240712
@file: tmp.py
@brief: tmp
"""
import torch
import torch.nn as nn
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.datasets import WMT21
import math
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

SRC_LANGUAGE = "en"
TGT_LANGUAGE = "zh"

# Tokenizers
token_transform = {}
vocab_transform = {}

token_transform[SRC_LANGUAGE] = get_tokenizer("spacy", language="en_core_web_sm")
token_transform[TGT_LANGUAGE] = get_tokenizer("spacy", language="zh_core_web_sm")

# Build Vocabulary
def yield_tokens(data_iter, language):
    for data_sample in data_iter:
        yield token_transform[language](
            data_sample[0 if language == SRC_LANGUAGE else 1]
        )


train_iter = WMT21(
    root=".data", split="train", language_pair=(SRC_LANGUAGE, TGT_LANGUAGE)
)

vocab_transform[SRC_LANGUAGE] = build_vocab_from_iterator(
    yield_tokens(train_iter, SRC_LANGUAGE),
    min_freq=2,
    specials=["<unk>", "<pad>", "<bos>", "<eos>"],
)
vocab_transform[TGT_LANGUAGE] = build_vocab_from_iterator(
    yield_tokens(train_iter, TGT_LANGUAGE),
    min_freq=2,
    specials=["<unk>", "<pad>", "<bos>", "<eos>"],
)

for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
    vocab_transform[ln].set_default_index(vocab_transform[ln]["<unk>"])


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class TransformerModel(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        d_model=512,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
    ):
        super(TransformerModel, self).__init__()
        self.transformer = nn.Transformer(
            d_model,
            nhead,
            num_encoder_layers,
            num_decoder_layers,
            dim_feedforward,
            dropout,
        )
        self.generator = nn.Linear(d_model, tgt_vocab_size)

        self.src_tok_emb = nn.Embedding(src_vocab_size, d_model)
        self.tgt_tok_emb = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout)

    def forward(
        self,
        src,
        tgt,
        src_mask,
        tgt_mask,
        src_padding_mask,
        tgt_padding_mask,
        memory_key_padding_mask,
    ):
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(tgt))
        memory = self.transformer(
            src_emb,
            tgt_emb,
            src_mask,
            tgt_mask,
            None,
            src_padding_mask,
            tgt_padding_mask,
            memory_key_padding_mask,
        )
        return self.generator(memory)


def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = (
        mask.float()
        .masked_fill(mask == 0, float("-inf"))
        .masked_fill(mask == 1, float(0.0))
    )
    return mask


def collate_fn(batch):
    src_batch, tgt_batch = [], []
    for src_sample, tgt_sample in batch:
        src_batch.append(
            torch.tensor(
                [
                    vocab_transform[SRC_LANGUAGE][token]
                    for token in token_transform[SRC_LANGUAGE](src_sample)
                ],
                dtype=torch.long,
            )
        )
        tgt_batch.append(
            torch.tensor(
                [
                    vocab_transform[TGT_LANGUAGE][token]
                    for token in token_transform[TGT_LANGUAGE](tgt_sample)
                ],
                dtype=torch.long,
            )
        )
    src_batch = pad_sequence(
        src_batch, padding_value=vocab_transform[SRC_LANGUAGE]["<pad>"]
    )
    tgt_batch = pad_sequence(
        tgt_batch, padding_value=vocab_transform[TGT_LANGUAGE]["<pad>"]
    )
    return src_batch, tgt_batch


train_iter = WMT21(
    root=".data", split="train", language_pair=(SRC_LANGUAGE, TGT_LANGUAGE)
)
train_dataloader = DataLoader(train_iter, batch_size=32, collate_fn=collate_fn)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = TransformerModel(
    len(vocab_transform[SRC_LANGUAGE]), len(vocab_transform[TGT_LANGUAGE])
)
model.to(device)

optimizer = optim.Adam(model.parameters(), lr=0.0001)
criterion = nn.CrossEntropyLoss(ignore_index=vocab_transform[TGT_LANGUAGE]["<pad>"])


def train_epoch(model, optimizer, criterion, dataloader):
    model.train()
    losses = 0
    for src, tgt in dataloader:
        src, tgt = src.to(device), tgt.to(device)
        tgt_input = tgt[:-1, :]

        src_mask = torch.zeros((src.size(0), src.size(0)), device=device).type(
            torch.bool
        )
        tgt_mask = generate_square_subsequent_mask(tgt_input.size(0)).to(device)

        src_padding_mask = (src == vocab_transform[SRC_LANGUAGE]["<pad>"]).transpose(
            0, 1
        )
        tgt_padding_mask = (
            tgt_input == vocab_transform[TGT_LANGUAGE]["<pad>"]
        ).transpose(0, 1)
        memory_key_padding_mask = src_padding_mask

        logits = model(
            src,
            tgt_input,
            src_mask,
            tgt_mask,
            src_padding_mask,
            tgt_padding_mask,
            memory_key_padding_mask,
        )

        optimizer.zero_grad()

        tgt_out = tgt[1:, :]
        loss = criterion(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        loss.backward()

        optimizer.step()
        losses += loss.item()
    return losses / len(dataloader)


def train():
    for epoch in range(10):
        loss = train_epoch(model, optimizer, criterion, train_dataloader)
        print(f"Epoch {epoch}, Loss: {loss}")


def greedy_decode(model, src, src_mask, max_len, start_symbol):
    src = src.to(device)
    src_mask = src_mask.to(device)

    memory = model.transformer.encoder(
        model.positional_encoding(model.src_tok_emb(src)), src_mask
    )
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(device)
    for i in range(max_len - 1):
        tgt_mask = (generate_square_subsequent_mask(ys.size(0)).type(torch.bool)).to(
            device
        )
        out = model.transformer.decoder(
            model.positional_encoding(model.tgt_tok_emb(ys)), memory, tgt_mask
        )
        out = model.generator(out)
        prob = out[-1, :].softmax(dim=-1)
        next_word = prob.argmax(dim=-1).item()

        ys = torch.cat(
            [ys, torch.ones(1, 1).fill_(next_word).type(torch.long).to(device)], dim=0
        )
        if next_word == vocab_transform[TGT_LANGUAGE]["<eos>"]:
            break
    return ys


model.eval()
src_sentence = "The quick brown fox jumps over the lazy dog."
src = torch.tensor(
    [
        vocab_transform[SRC_LANGUAGE][token]
        for token in token_transform[SRC_LANGUAGE](src_sentence)
    ],
    dtype=torch.long,
).view(-1, 1)
num_tokens = src.shape[0]
src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)

tgt_tokens = greedy_decode(
    model,
    src,
    src_mask,
    max_len=60,
    start_symbol=vocab_transform[TGT_LANGUAGE]["<bos>"],
)
translation = " ".join(
    [
        vocab_transform[TGT_LANGUAGE].lookup_token(idx)
        for idx in tgt_tokens.squeeze().tolist()
    ]
)
print(translation)
