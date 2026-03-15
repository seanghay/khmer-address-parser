from generate import TAGS_2_ID

ID_2_TAGS: dict[int, str] = {v: k for k, v in TAGS_2_ID.items()}
ID_2_TAGS[3] = "0"


def extract_spans(tag_ids: list[int]) -> list[tuple[str, int, int]]:
  spans = []
  cur_type = None
  cur_start = None

  for i, tid in enumerate(tag_ids):
    tag = ID_2_TAGS.get(tid, "0")
    if tag == "0":
      if cur_type is not None:
        spans.append((cur_type, cur_start, i))
        cur_type, cur_start = None, None
    elif tag.startswith("B_"):
      if cur_type is not None:
        spans.append((cur_type, cur_start, i))
      cur_type = tag[2:]
      cur_start = i
    elif tag.startswith("I_"):
      entity = tag[2:]
      if cur_type != entity:
        if cur_type is not None:
          spans.append((cur_type, cur_start, i))
        cur_type = entity
        cur_start = i
    else:
      if cur_type is not None:
        spans.append((cur_type, cur_start, i))
        cur_type, cur_start = None, None

  if cur_type is not None:
    spans.append((cur_type, cur_start, len(tag_ids)))

  return spans


def entity_f1(pred_seqs: list[list[int]], gold_seqs: list[list[int]]) -> dict[str, float]:
  tp = fp = fn = 0
  for pred, gold in zip(pred_seqs, gold_seqs):
    pred_spans = set(extract_spans(pred))
    gold_spans = set(extract_spans(gold))
    tp += len(pred_spans & gold_spans)
    fp += len(pred_spans - gold_spans)
    fn += len(gold_spans - pred_spans)
  precision = tp / (tp + fp) if tp + fp > 0 else 0.0
  recall = tp / (tp + fn) if tp + fn > 0 else 0.0
  f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
  return {"precision": precision, "recall": recall, "f1": f1}


def decode_entities(text: str, tag_ids: list[int]) -> dict[str, str | None]:
  result: dict[str, str | None] = {
    "province": None,
    "district": None,
    "commune": None,
    "village": None,
    "house": None,
    "road": None,
  }
  for entity_type, start, end in extract_spans(tag_ids):
    key = entity_type.lower()
    if key in result and result[key] is None:
      result[key] = text[start:end]
  return result
