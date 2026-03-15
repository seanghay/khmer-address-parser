import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torchcrf import CRF


class CharCNN(nn.Module):
  def __init__(
    self,
    vocab_size: int,
    char_emb_dim: int = 128,
    num_filters: int = 128,
  ):
    super().__init__()
    self.embedding = nn.Embedding(vocab_size, char_emb_dim, padding_idx=0)
    self.kernel_sizes = (2, 3, 4)

    self.convs = nn.ModuleList(
      [
        nn.Conv1d(char_emb_dim, num_filters, kernel_size=k, padding=k // 2)
        for k in self.kernel_sizes
      ]
    )
    self.out_dim = num_filters * len(self.kernel_sizes)

  def forward(self, x):
    emb = self.embedding(x)
    t = x.size(1)
    emb_t = emb.transpose(1, 2)
    parts = []
    for conv in self.convs:
      out = conv(emb_t)[:, :, :t]
      parts.append(out.transpose(1, 2))
    return torch.cat(parts, dim=-1)


class NERModel(nn.Module):
  def __init__(
    self,
    vocab_size: int,
    num_tags: int = 14,
    char_emb_dim: int = 128,
    num_filters: int = 128,
    hidden: int = 256,
    lstm_layers: int = 2,
    dropout: float = 0.3,
  ):
    super().__init__()
    self.cnn = CharCNN(vocab_size, char_emb_dim, num_filters)
    self.dropout = nn.Dropout(dropout)
    self.lstm = nn.LSTM(
      self.cnn.out_dim,
      hidden,
      num_layers=lstm_layers,
      batch_first=True,
      bidirectional=True,
    )
    self.linear = nn.Linear(hidden * 2, num_tags)
    self.crf = CRF(num_tags, batch_first=True)

  def _emit(self, chars, mask):
    cnn_out = self.cnn(chars)
    cnn_out = self.dropout(cnn_out)
    lengths = mask.sum(dim=1).cpu()
    packed = pack_padded_sequence(
      cnn_out, lengths, batch_first=True, enforce_sorted=False
    )
    lstm_out, _ = self.lstm(packed)
    lstm_out, _ = pad_packed_sequence(
      lstm_out, batch_first=True, total_length=chars.size(1)
    )
    lstm_out = self.dropout(lstm_out)
    return self.linear(lstm_out)

  def forward(self, chars, tags, mask):
    emissions = self._emit(chars, mask)
    return -self.crf(emissions, tags, mask=mask.byte(), reduction="mean")

  def decode(self, chars, mask):
    emissions = self._emit(chars, mask)
    return self.crf.decode(emissions, mask=mask.byte())
