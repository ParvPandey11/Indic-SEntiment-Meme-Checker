"""Train and evaluate text-only, image-only or fusion models.

Examples:
  python3 -m src.train --mode text   --name text_xlmr
  python3 -m src.train --mode image  --name image_resnet
  python3 -m src.train --mode fusion --fusion concat --name fusion_concat
  python3 -m src.train --mode fusion --fusion gated  --name fusion_gated
  python3 -m src.train --mode fusion --limit 64 --epochs 1 --name smoke   # quick test
"""
import argparse, json, os, random, sys
import numpy as np, pandas as pd, torch, torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.model import MemeModel


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


class MemeDS(Dataset):
    def __init__(self, df, tfm):
        self.df, self.tfm = df.reset_index(drop=True), tfm

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = Image.open(r.image_path).convert("RGB")
        return self.tfm(img), (str(r.text) if pd.notna(r.text) else ""), int(r.label)


def collate_fn(tok, max_len):
    def fn(batch):
        imgs = torch.stack([b[0] for b in batch])
        enc = tok([b[1] for b in batch], padding=True, truncation=True, max_length=max_len, return_tensors="pt") if tok else None
        return imgs, enc, torch.tensor([b[2] for b in batch])
    return fn


def load(split, limit=None):
    df = pd.read_csv(os.path.join(config.PROCESSED_DIR, f"{split}.csv"), encoding="utf-8")
    if limit:
        df = df.sample(min(limit, len(df)), random_state=config.SEED)
    return df


@torch.no_grad()
def run_eval(model, loader, dev, loss_fn=None):
    model.eval(); preds, labels, total = [], [], 0.0
    for imgs, enc, y in tqdm(loader, leave=False):
        imgs = imgs.to(dev)
        enc = {k: v.to(dev) for k, v in enc.items()} if enc is not None else None
        out = model(imgs, enc)
        if loss_fn is not None and (y >= 0).all():
            total += loss_fn(out, y.to(dev)).item() * len(y)
        preds += out.argmax(1).cpu().tolist(); labels += y.tolist()
    return total / max(len(labels), 1), np.array(preds), np.array(labels)


def metrics(y, p):
    return {"accuracy": accuracy_score(y, p), "macro_f1": f1_score(y, p, average="macro"),
            "weighted_f1": f1_score(y, p, average="weighted"),
            "report": classification_report(y, p, labels=[0, 1, 2], target_names=config.LABELS, output_dict=True, zero_division=0),
            "confusion_matrix": confusion_matrix(y, p, labels=[0, 1, 2]).tolist()}


def save_cm(cm, path):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4, 4)); ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(3)); ax.set_yticks(range(3))
    ax.set_xticklabels(config.LABELS, rotation=30); ax.set_yticklabels(config.LABELS)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, cm[i][j], ha="center", va="center")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["text", "image", "fusion"], default="fusion")
    ap.add_argument("--fusion", choices=["concat", "gated"], default="concat")
    ap.add_argument("--text_model", default="xlm-roberta-base")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--text_lr", type=float, default=2e-5)
    ap.add_argument("--img_lr", type=float, default=1e-4)
    ap.add_argument("--head_lr", type=float, default=1e-3)
    ap.add_argument("--freeze", action="store_true", help="freeze encoders, train head only")
    ap.add_argument("--limit", type=int, default=None, help="use N samples per split (smoke test)")
    # A nested collate function is not picklable under Python 3.14's default
    # macOS multiprocessing start method; zero workers is reliable on Mac.
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--name", default=None)
    a = ap.parse_args()

    seed_all(a.seed); dev = get_device()
    name = a.name or (f"fusion_{a.fusion}" if a.mode == "fusion" else a.mode)
    out_dir = os.path.join(config.RUNS_DIR, name); os.makedirs(out_dir, exist_ok=True)
    print("device:", dev, "| run:", name)

    tok = AutoTokenizer.from_pretrained(a.text_model) if a.mode in ("text", "fusion") else None
    norm = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    # No horizontal flip: it would mirror the text inside memes.
    tr_tfm = T.Compose([T.Resize((224, 224)), T.ColorJitter(0.2, 0.2, 0.2), T.ToTensor(), norm])
    ev_tfm = T.Compose([T.Resize((224, 224)), T.ToTensor(), norm])

    tr, va, te = load("train", a.limit), load("val", a.limit), load("test", a.limit)
    tr, va = tr[tr.label >= 0], va[va.label >= 0]
    if tr.empty or va.empty:
        raise ValueError("Run scripts/01_prepare_data.py first; it must create nonempty processed train.csv and val.csv")
    cf = collate_fn(tok, a.max_len)
    mk = lambda d, tfm, sh: DataLoader(MemeDS(d, tfm), batch_size=a.bs, shuffle=sh, num_workers=a.workers, collate_fn=cf)
    tr_l, va_l, te_l = mk(tr, tr_tfm, True), mk(va, ev_tfm, False), mk(te, ev_tfm, False)

    model = MemeModel(a.mode, a.fusion, a.text_model).to(dev)
    groups = [{"params": model.head_params(), "lr": a.head_lr}]
    for n, params in model.encoder_params():
        if a.freeze:
            for p in params:
                p.requires_grad = False
        else:
            groups.append({"params": params, "lr": a.text_lr if n == "text_enc" else a.img_lr})
    opt = torch.optim.AdamW(groups, weight_decay=0.01)
    steps = len(tr_l) * a.epochs
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)

    counts = np.bincount(tr.label.values, minlength=3).astype(float)
    w = torch.tensor(counts.sum() / (3 * np.maximum(counts, 1)), dtype=torch.float).to(dev)
    loss_fn = nn.CrossEntropyLoss(weight=w)

    best, hist = -1.0, []
    for ep in range(1, a.epochs + 1):
        model.train(); run = 0.0
        for imgs, enc, y in tqdm(tr_l, desc=f"epoch {ep}"):
            imgs = imgs.to(dev)
            enc = {k: v.to(dev) for k, v in enc.items()} if enc is not None else None
            y = y.to(dev)
            opt.zero_grad(); loss = loss_fn(model(imgs, enc), y); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step(); run += loss.item()
        train_loss = run / max(len(tr_l), 1)
        vl, vp, vy = run_eval(model, va_l, dev, loss_fn)
        vm = metrics(vy, vp)
        print(f"epoch {ep}: train_loss={train_loss:.4f} val_loss={vl:.4f} val_macroF1={vm['macro_f1']:.4f}")
        hist.append({"epoch": ep, "train_loss": train_loss, "val_loss": vl, "val_macro_f1": vm["macro_f1"]})
        torch.save(model.state_dict(), os.path.join(out_dir, "last.pt"))
        if vm["macro_f1"] > best:
            best = vm["macro_f1"]
            torch.save(model.state_dict(), os.path.join(out_dir, "best.pt"))

    model.load_state_dict(torch.load(os.path.join(out_dir, "best.pt"), map_location=dev))
    _, vp, vy = run_eval(model, va_l, dev)
    result = {"args": vars(a), "history": hist, "val": metrics(vy, vp)}
    _, tp, ty = run_eval(model, te_l, dev)
    if (ty >= 0).all():
        result["test"] = metrics(ty, tp)
        save_cm(result["test"]["confusion_matrix"], os.path.join(out_dir, "confusion_matrix_test.png"))
        print(f"TEST: acc={result['test']['accuracy']:.4f} macroF1={result['test']['macro_f1']:.4f} weightedF1={result['test']['weighted_f1']:.4f}")
    else:
        print("Test labels not available; saving predictions only. No test metrics are computed.")
        save_cm(result["val"]["confusion_matrix"], os.path.join(out_dir, "confusion_matrix_val.png"))
        pd.DataFrame({"image_path": te.image_path.values, "pred": [config.LABELS[i] for i in tp]}).to_csv(
            os.path.join(out_dir, "test_predictions.csv"), index=False)
    json.dump(result, open(os.path.join(out_dir, "metrics.json"), "w"), indent=2)
    print("saved to", out_dir)


if __name__ == "__main__":
    main()
