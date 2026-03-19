import argparse
import json
from difflib import SequenceMatcher

import numpy as np
import onnxruntime as ort

from utils import decode_entities

PREFIXES = {
  "province": ["ខេត្ត", "រាជធានី"],
  "district": ["ស្រុក", "ខណ្ឌ", "ក្រុង"],
  "commune": ["ឃុំ", "សង្កាត់"],
  "village": ["ភូមិ"],
}


def strip_prefix(text: str, prefixes: list[str]) -> str:
  changed = True
  while changed:
    changed = False
    for p in prefixes:
      if text.startswith(p):
        text = text[len(p) :]
        changed = True
  return text


def best_match(
  query: str, candidates: list[str], level: str, threshold: float
) -> str | None:
  prefixes = PREFIXES[level]
  q = strip_prefix(query, prefixes)
  best_score = -1.0
  best_candidate = None
  for c in candidates:
    c_stripped = strip_prefix(c, prefixes)
    score = SequenceMatcher(None, q, c_stripped).ratio()
    if score > best_score:
      best_score = score
      best_candidate = c
  if best_score >= threshold:
    return best_candidate
  return None


class Autocorrector:
  def __init__(self, data_path: str):
    with open(data_path, encoding="utf-8") as f:
      data = json.load(f)
    self.provinces = data

  def correct(self, result: dict, thresholds: dict) -> dict:
    out = result.copy()

    matched_province = None
    if out.get("province"):
      province_names = [p["name"]["km"] for p in self.provinces]
      corrected = best_match(
        out["province"], province_names, "province", thresholds["province"]
      )
      if corrected:
        out["province"] = corrected
        matched_province = next(
          p for p in self.provinces if p["name"]["km"] == corrected
        )

    matched_district = None
    if out.get("district"):
      district_pool = (
        matched_province["districts"]["values"]
        if matched_province
        else [d for p in self.provinces for d in p["districts"]["values"]]
      )
      district_names = [d["name"]["km"] for d in district_pool]
      corrected = best_match(
        out["district"], district_names, "district", thresholds["district"]
      )
      if corrected:
        out["district"] = corrected
        matched_district = next(
          d for d in district_pool if d["name"]["km"] == corrected
        )

    matched_commune = None
    if out.get("commune"):
      commune_pool = (
        matched_district["communes"]["values"]
        if matched_district
        else [
          c
          for p in self.provinces
          for d in p["districts"]["values"]
          for c in d["communes"]["values"]
        ]
      )
      commune_names = [c["name"]["km"] for c in commune_pool]
      corrected = best_match(
        out["commune"], commune_names, "commune", thresholds["commune"]
      )
      if corrected:
        out["commune"] = corrected
        matched_commune = next(c for c in commune_pool if c["name"]["km"] == corrected)

    if out.get("village"):
      village_pool = (
        matched_commune["villages"]["values"]
        if matched_commune
        else [
          v
          for p in self.provinces
          for d in p["districts"]["values"]
          for c in d["communes"]["values"]
          for v in c["villages"]["values"]
        ]
      )
      village_names = [v["name"]["km"] for v in village_pool]
      corrected = best_match(
        out["village"], village_names, "village", thresholds["village"]
      )
      if corrected:
        out["village"] = corrected

    return out


def viterbi_decode(
  emissions: np.ndarray,
  transitions: np.ndarray,
  start_transitions: np.ndarray,
  end_transitions: np.ndarray,
) -> list[int]:
  """Viterbi decoding in numpy. emissions: (seq_len, num_tags)."""
  seq_len, num_tags = emissions.shape
  score = start_transitions + emissions[0]  # (num_tags,)
  history = []

  for t in range(1, seq_len):
    # broadcast: score[prev] + transitions[prev, next] + emissions[t, next]
    broadcast = score[:, None] + transitions  # (num_tags, num_tags)
    best_prev = broadcast.argmax(axis=0)  # (num_tags,)
    score = broadcast[best_prev, np.arange(num_tags)] + emissions[t]
    history.append(best_prev)

  score += end_transitions
  best_last = int(score.argmax())

  tags = [best_last]
  for best_prev in reversed(history):
    best_last = int(best_prev[best_last])
    tags.append(best_last)
  tags.reverse()
  return tags


class ONNXPredictor:
  def __init__(
    self,
    onnx_path: str,
    crf_path: str,
    vocab_path: str,
    data_path: str | None = None,
    thresholds: dict | None = None,
  ):
    self.sess = ort.InferenceSession(onnx_path)
    crf = np.load(crf_path)

    self.transitions = crf["transitions"]  # (num_tags, num_tags)
    self.start_transitions = crf["start_transitions"]  # (num_tags,)
    self.end_transitions = crf["end_transitions"]  # (num_tags,)

    with open(vocab_path, encoding="utf-8") as f:
      self.vocab: dict[str, int] = json.load(f)
    self.unk = self.vocab.get("<UNK>", 1)

    self.corrector = Autocorrector(data_path) if data_path else None
    self.thresholds = thresholds or {
      "province": 0.6,
      "district": 0.6,
      "commune": 0.6,
      "village": 0.6,
    }

  def predict(self, text: str) -> dict:
    char_ids = np.array([[self.vocab.get(ch, self.unk) for ch in text]], dtype=np.int64)
    emissions = self.sess.run(["emissions"], {"chars": char_ids})[0][
      0
    ]  # (seq_len, num_tags)
    tag_ids = viterbi_decode(
      emissions, self.transitions, self.start_transitions, self.end_transitions
    )
    result = decode_entities(text, tag_ids)
    if self.corrector:
      result = self.corrector.correct(result, self.thresholds)
    return result


def main():
  parser = argparse.ArgumentParser(
    description="ONNX inference for Khmer address parser"
  )
  parser.add_argument("--onnx", default="checkpoints/best.onnx")
  parser.add_argument("--crf", default="checkpoints/best_crf_params.npz")
  parser.add_argument("--vocab", default="checkpoints/best_vocab.json")
  parser.add_argument(
    "--data", default="data.json", help="path to data.json for autocorrect"
  )
  parser.add_argument(
    "--no-autocorrect", action="store_true", help="disable autocorrect"
  )
  parser.add_argument("--threshold-province", type=float, default=0.6)
  parser.add_argument("--threshold-district", type=float, default=0.6)
  parser.add_argument("--threshold-commune", type=float, default=0.6)
  parser.add_argument("--threshold-village", type=float, default=0.6)
  parser.add_argument(
    "--text",
    default="ផ្ទះលេខ២ ផ្លូវ២៣ ភូមិប៉ុស្តិ័ចាស់ព្រះនេត្រព្រះព្រះនេត្រព្រះបន្ទាយមានជ",
  )
  args = parser.parse_args()

  thresholds = {
    "province": args.threshold_province,
    "district": args.threshold_district,
    "commune": args.threshold_commune,
    "village": args.threshold_village,
  }

  data_path = None if args.no_autocorrect else args.data
  predictor = ONNXPredictor(
    args.onnx, args.crf, args.vocab, data_path=data_path, thresholds=thresholds
  )
  result = predictor.predict(args.text)
  print(result)


if __name__ == "__main__":
  main()


