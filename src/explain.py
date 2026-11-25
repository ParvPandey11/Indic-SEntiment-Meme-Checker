"""Perturbation based explanations for a trained Memotion checkpoint.

Reports leave-one-word-out text sensitivity and coarse image occlusion
sensitivity for the model's predicted class. These are diagnostic signals,
not causal explanations.

Example:
  python3 -m src.explain --run runs/fusion_frozen_e3 --split val --row 0
"""
import argparse
import json
import os
import re

import pandas as pd
import torch
from PIL import Image
from torchvision import transforms as T
from transformers import AutoTokenizer

import config
from src.gradcam import gradcam_overlay
from src.model import MemeModel


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def tokenize(tok, text, max_len, device):
    if tok is None:
        return None
    enc = tok(text, truncation=True, max_length=max_len, return_tensors="pt")
    return {k: v.to(device) for k, v in enc.items()}


@torch.no_grad()
def probabilities(model, image, tok, text, max_len, device):
    enc = tokenize(tok, text, max_len, device)
    return model(image, enc).softmax(dim=1)[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True, help="Training run directory containing best.pt and metrics.json")
    ap.add_argument("--split", choices=("train", "val", "test"), default="val")
    ap.add_argument("--row", type=int, default=0, help="Zero-based row in data/processed/<split>.csv")
    ap.add_argument("--grid", type=int, default=4, help="Image occlusion grid size per dimension")
    ap.add_argument("--max-words", type=int, default=40, help="Maximum text tokens to test individually")
    ap.add_argument("--output", default=None, help="Output JSON path (defaults to <run>/explanations_<split>_<row>.json)")
    a = ap.parse_args()

    run_dir = os.path.abspath(a.run)
    meta_path = os.path.join(run_dir, "metrics.json")
    ckpt_path = os.path.join(run_dir, "best.pt")
    if not os.path.isfile(meta_path) or not os.path.isfile(ckpt_path):
        raise FileNotFoundError("Run must contain metrics.json and best.pt")
    meta = json.load(open(meta_path, encoding="utf-8"))
    args = meta["args"]
    table_path = os.path.join(config.PROCESSED_DIR, f"{a.split}.csv")
    df = pd.read_csv(table_path)
    if not 0 <= a.row < len(df):
        raise IndexError(f"row must be between 0 and {len(df)-1} for {table_path}")
    row = df.iloc[a.row]
    text = "" if pd.isna(row.get("text")) else str(row.get("text"))
    device = get_device()
    tok = AutoTokenizer.from_pretrained(args["text_model"]) if args["mode"] in ("text", "fusion") else None
    transform = T.Compose([
        T.Resize((224, 224)), T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    image = transform(Image.open(row["image_path"]).convert("RGB")).unsqueeze(0).to(device)
    model = MemeModel(args["mode"], args["fusion"], args["text_model"]).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    base = probabilities(model, image, tok, text, args.get("max_len", 128), device)
    predicted = int(base.argmax().item())
    predicted_name = config.LABELS[predicted]
    result = {
        "run": os.path.basename(run_dir), "split": a.split, "row": a.row,
        "image_path": str(row["image_path"]), "gold_label": config.LABELS[int(row["label"])] if int(row["label"]) in range(3) else None,
        "predicted_label": predicted_name,
        "probabilities": {name: float(base[i].item()) for i, name in enumerate(config.LABELS)},
        "method": "leave-one-whitespace-token-out and normalized-image grid occlusion",
        "notice": "Sensitivity scores show how a prediction changes under perturbation; they are not causal explanations.",
    }

    if tok is not None and text.strip():
        words = re.findall(r"\S+", text)
        scored = []
        for i, word in enumerate(words[:a.max_words]):
            perturbed = " ".join(words[:i] + words[i + 1:])
            p = probabilities(model, image, tok, perturbed, args.get("max_len", 128), device)[predicted].item()
            scored.append({"token": word, "index": i, "predicted_probability_drop": float(base[predicted].item() - p)})
        result["text_token_sensitivity"] = sorted(scored, key=lambda x: x["predicted_probability_drop"], reverse=True)

    if args["mode"] in ("image", "fusion"):
        grid = max(1, a.grid)
        _, _, height, width = image.shape
        scores = []
        for gy in range(grid):
            y0, y1 = gy * height // grid, (gy + 1) * height // grid
            for gx in range(grid):
                x0, x1 = gx * width // grid, (gx + 1) * width // grid
                masked = image.clone()
                masked[:, :, y0:y1, x0:x1] = 0.0  # normalized mean-color baseline
                p = probabilities(model, masked, tok, text, args.get("max_len", 128), device)[predicted].item()
                scores.append({"grid_row": gy, "grid_col": gx, "predicted_probability_drop": float(base[predicted].item() - p)})
        result["image_grid_sensitivity"] = sorted(scores, key=lambda x: x["predicted_probability_drop"], reverse=True)
        gradcam_enc = tokenize(tok, text, args.get("max_len", 128), device)
        _original, overlay, _pred, _logits = gradcam_overlay(model, image, gradcam_enc, target_class=predicted)
        cam_path = os.path.join(run_dir, f"gradcam_{a.split}_{a.row}.png")
        overlay.save(cam_path)
        result["gradcam_path"] = cam_path
        result["method"] += " and Grad-CAM on the ResNet-50 layer4 block"

    out = a.output or os.path.join(run_dir, f"explanations_{a.split}_{a.row}.json")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"Predicted: {predicted_name} | saved explanation: {out}")


if __name__ == "__main__":
    main()
