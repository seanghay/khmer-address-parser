import argparse
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from generate import generate_address
from data import build_vocab, AddressDataset, collate_fn
from model import NERModel
from utils import entity_f1


def train_epoch(model, loader, optimizer, clip, device):
  model.train()
  total_loss = 0.0
  pbar = tqdm(loader, desc="train", leave=False)
  for chars, tags, mask in pbar:
    chars, tags, mask = chars.to(device), tags.to(device), mask.to(device)
    loss = model(chars, tags, mask)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
    optimizer.step()
    total_loss += loss.item()
    pbar.set_postfix(loss=f"{loss.item():.4f}")
  return total_loss / len(loader)


def eval_epoch(model, loader, device):
  model.eval()
  all_pred, all_gold = [], []
  with torch.no_grad():
    for chars, tags, mask in loader:
      chars, tags, mask = chars.to(device), tags.to(device), mask.to(device)
      preds = model.decode(chars, mask)
      lengths = mask.sum(dim=1).tolist()
      for pred, gold, length in zip(preds, tags.tolist(), lengths):
        all_pred.append(pred[:length])
        all_gold.append(gold[:length])
  return entity_f1(all_pred, all_gold)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--epochs", type=int, default=30)
  parser.add_argument("--batch_size", type=int, default=64)
  parser.add_argument("--lr", type=float, default=1e-3)
  parser.add_argument("--clip", type=float, default=5.0)
  parser.add_argument("--checkpoint", type=str, default="checkpoints/")
  parser.add_argument("--resume", type=str, default=None)
  parser.add_argument("--device", type=str, default="cpu")
  args = parser.parse_args()

  device = torch.device(args.device)
  os.makedirs(args.checkpoint, exist_ok=True)

  clean = list(generate_address(0, 0, 0, 0, 0))

  if args.resume:
    ckpt = torch.load(args.resume, map_location=device)
    vocab = ckpt["vocab"]
  else:
    vocab = build_vocab(clean)
    ckpt = None

  val_ds = AddressDataset(clean, vocab)

  train_samples = []
  for _ in range(3):
    train_samples += list(
      generate_address(
        remove_space_prob=0.1,
        noise_prob=0.05,
        drop_char_prob=0.05,
        drop_prefix_prob=0.5,
        drop_component_prob=0.2,
        insert_word_prob=0.05,
      )
    )
  train_ds = AddressDataset(train_samples, vocab)

  train_loader = DataLoader(
    train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn
  )
  val_loader = DataLoader(
    val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn
  )

  model = NERModel(vocab_size=len(vocab), num_tags=14).to(device)
  optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
  scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="max", factor=0.5, patience=3
  )

  start_epoch = 0
  best_f1 = 0.0

  if ckpt is not None:
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    scheduler.load_state_dict(ckpt["scheduler_state"])
    start_epoch = ckpt["epoch"] + 1
    best_f1 = ckpt["best_f1"]
    print(f"Resumed from epoch {ckpt['epoch']}, best_f1={best_f1:.4f}")

  epoch_pbar = tqdm(range(start_epoch, args.epochs), desc="epochs")
  for epoch in epoch_pbar:
    loss = train_epoch(model, train_loader, optimizer, args.clip, device)
    metrics = eval_epoch(model, val_loader, device)
    val_f1 = metrics["f1"]
    scheduler.step(val_f1)
    epoch_pbar.set_postfix(
      loss=f"{loss:.4f}",
      P=f"{metrics['precision']:.4f}",
      R=f"{metrics['recall']:.4f}",
      F1=f"{val_f1:.4f}",
    )
    tqdm.write(
      f"Epoch {epoch:3d} | loss={loss:.4f} | P={metrics['precision']:.4f} R={metrics['recall']:.4f} F1={val_f1:.4f}"
    )

    state = {
      "epoch": epoch,
      "model_state": model.state_dict(),
      "optimizer_state": optimizer.state_dict(),
      "scheduler_state": scheduler.state_dict(),
      "best_f1": best_f1,
      "vocab": vocab,
    }
    torch.save(state, os.path.join(args.checkpoint, "last.pt"))

    if val_f1 > best_f1:
      best_f1 = val_f1
      state["best_f1"] = best_f1
      torch.save(state, os.path.join(args.checkpoint, "best.pt"))
      tqdm.write(f"  -> new best F1={best_f1:.4f}")


if __name__ == "__main__":
  main()
