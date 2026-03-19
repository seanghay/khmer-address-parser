# Khmer Neural Address Parser

Named entity recognition model for parsing unstructured Khmer address strings into structured fields: province, district, commune, village, house number, and road.

## How it works

Training data is generated synthetically from a structured address database (`data.json`). Each address is assembled from real place names at all four administrative levels, with optional house and road components. Augmentation applies random noise: dropped characters, removed prefixes, missing spaces, dropped components, and injected dictionary words — producing varied and realistic inputs.

The model is a character-level NER pipeline:

1. Each character in the input string is mapped to an integer via a vocabulary built from training data.
2. A CharCNN extracts local character features using parallel convolutions (kernel sizes 2, 3, 4).
3. A bidirectional LSTM reads the sequence and captures long-range context.
4. A linear layer projects to tag logits over 14 BIO tags.
5. A CRF layer enforces valid tag transitions during decoding.

For inference, the CNN + LSTM + linear layers are exported to ONNX. The CRF parameters (transition matrices) are saved separately as `.npz` and decoded in NumPy using the Viterbi algorithm, avoiding any PyTorch dependency at runtime.

After entity extraction, an optional autocorrect step fuzzy-matches each predicted span against the known address database, correcting OCR-style errors or partial names.

## Setup

```shell
pip install -r requirements.txt
```

## Train

```shell
python train.py
```

Checkpoints are saved to `checkpoints/`. Use `--resume checkpoints/last.pt` to continue training.

## Export

```shell
python export_onnx.py
```

Writes `checkpoints/best.onnx`, `best_vocab.json`, and `best_crf_params.npz`.

## Inference

```shell
python inference_onnx.py --text "ផ្ទះលេខ២ ផ្លូវ២៣ ភូមិប៉ុស្តិ័ចាស់ព្រះនេត្រព្រះបន្ទាយមានជ"
```

```
{
  "province": "ខេត្តបន្ទាយមានជ័យ",
  "district": "ស្រុកព្រះនេត្រព្រះ",
  "commune":  "ឃុំព្រះនេត្រព្រះ",
  "village":  "ប៉ុស្ដិចាស់",
  "house":    "ផ្ទះលេខ២",
  "road":     "ផ្លូវ២៣"
}
```

Pass `--no-autocorrect` to skip fuzzy matching. Autocorrect thresholds per level can be tuned with `--threshold-province`, `--threshold-district`, `--threshold-commune`, `--threshold-village` (default: 0.6).
