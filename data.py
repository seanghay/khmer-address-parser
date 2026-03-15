import torch
from torch.utils.data import Dataset
from torch import Tensor


def build_vocab(samples: list[tuple[str, list[int]]]) -> dict[str, int]:
  vocab: dict[str, int] = {"<PAD>": 0, "<UNK>": 1}
  for text, _ in samples:
    for ch in text:
      if ch not in vocab:
        vocab[ch] = len(vocab)
  return vocab


class AddressDataset(Dataset):
  def __init__(self, samples: list[tuple[str, list[int]]], vocab: dict[str, int]):
    self.samples = samples
    self.vocab = vocab

  def __len__(self):
    return len(self.samples)

  def __getitem__(self, idx):
    text, labels = self.samples[idx]
    unk = self.vocab["<UNK>"]
    char_ids = [self.vocab.get(ch, unk) for ch in text]
    return char_ids, labels


def collate_fn(batch) -> tuple[Tensor, Tensor, Tensor]:
  lengths = [len(item[0]) for item in batch]
  t_max = max(lengths)
  chars_padded = []
  tags_padded = []
  masks = []
  for (char_ids, label_ids), length in zip(batch, lengths):
    pad_len = t_max - length
    chars_padded.append(char_ids + [0] * pad_len)
    tags_padded.append(label_ids + [0] * pad_len)
    masks.append([True] * length + [False] * pad_len)
  return (
    torch.tensor(chars_padded, dtype=torch.long),
    torch.tensor(tags_padded, dtype=torch.long),
    torch.tensor(masks, dtype=torch.bool),
  )
