import argparse
import torch

from model import NERModel
from utils import decode_entities


def load_model(path: str, device) -> tuple[NERModel, dict[str, int]]:
  ckpt = torch.load(path, map_location=device)
  vocab = ckpt["vocab"]
  model = NERModel(vocab_size=len(vocab), num_tags=14).to(device)
  model.load_state_dict(ckpt["model_state"])
  model.eval()
  return model, vocab


def parse_address(text: str, model: NERModel, vocab: dict[str, int], device) -> dict:
  unk = vocab.get("<UNK>", 1)
  char_ids = [vocab.get(ch, unk) for ch in text]
  chars = torch.tensor([char_ids], dtype=torch.long, device=device)
  mask = torch.ones(1, len(char_ids), dtype=torch.bool, device=device)
  with torch.no_grad():
    tag_ids = model.decode(chars, mask)[0]
  return decode_entities(text, tag_ids)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--checkpoint", required=False, default="checkpoints/best.pt")
  parser.add_argument(
    "--text",
    required=False,
    default="ឃុព្រះនេត្រព្រះស្រុកព្រះនេត្រព្រះបន្ទយមានជ",
  )
  parser.add_argument("--device", default="cpu")
  args = parser.parse_args()

  device = torch.device(args.device)
  model, vocab = load_model(args.checkpoint, device)
  result = parse_address(args.text, model, vocab, device)
  print(result)


if __name__ == "__main__":
  main()
