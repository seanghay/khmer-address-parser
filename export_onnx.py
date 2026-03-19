import argparse
import json
import torch
import torch.nn as nn
import numpy as np

from model import NERModel


class EmissionWrapper(nn.Module):
    """Wraps NERModel for ONNX export: bypasses pack/unpack and CRF decode."""

    def __init__(self, model: NERModel):
        super().__init__()
        self.cnn = model.cnn
        self.lstm = model.lstm
        self.linear = model.linear

    def forward(self, chars: torch.Tensor) -> torch.Tensor:
        # chars: (batch, seq_len) int64
        cnn_out = self.cnn(chars)          # (batch, seq_len, cnn_out_dim)
        lstm_out, _ = self.lstm(cnn_out)   # (batch, seq_len, hidden*2)
        return self.linear(lstm_out)       # (batch, seq_len, num_tags)


def export(checkpoint: str, output: str) -> None:
    print(f"Loading checkpoint: {checkpoint}")
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    vocab = ckpt["vocab"]
    model = NERModel(vocab_size=len(vocab), num_tags=14)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    wrapper = EmissionWrapper(model)
    wrapper.eval()

    dummy_chars = torch.zeros(1, 32, dtype=torch.long)

    print(f"Exporting ONNX model to: {output}")
    torch.onnx.export(
        wrapper,
        dummy_chars,
        output,
        input_names=["chars"],
        output_names=["emissions"],
        dynamic_axes={
            "chars": {0: "batch", 1: "seq_len"},
            "emissions": {0: "batch", 1: "seq_len"},
        },
        opset_version=14,
        do_constant_folding=True,
        dynamo=False,
    )
    print("ONNX export complete.")

    vocab_path = output.replace(".onnx", "_vocab.json")
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False)
    print(f"Vocab saved to: {vocab_path}")

    crf_path = output.replace(".onnx", "_crf_params.npz")
    np.savez(
        crf_path,
        transitions=model.crf.transitions.detach().numpy(),
        start_transitions=model.crf.start_transitions.detach().numpy(),
        end_transitions=model.crf.end_transitions.detach().numpy(),
    )
    print(f"CRF parameters saved to: {crf_path}")

    try:
        import onnxruntime as ort

        sess = ort.InferenceSession(output)
        out = sess.run(["emissions"], {"chars": dummy_chars.numpy()})
        print(f"ONNX validation passed. Output shape: {out[0].shape}")  # expect (1, 32, 14)

        # Cross-check against PyTorch
        with torch.no_grad():
            mask = torch.ones(1, 32, dtype=torch.bool)
            pt_out = model._emit(dummy_chars, mask).numpy()
        max_diff = abs(out[0] - pt_out).max()
        print(f"Max abs diff vs PyTorch _emit: {max_diff:.6f}")
    except ImportError:
        print("onnxruntime not installed — skipping validation.")


def main():
    parser = argparse.ArgumentParser(description="Export NERModel to ONNX")
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--output", default="checkpoints/best.onnx")
    args = parser.parse_args()
    export(args.checkpoint, args.output)


if __name__ == "__main__":
    main()
