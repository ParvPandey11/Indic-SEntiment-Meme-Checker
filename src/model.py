import torch, torch.nn as nn
from torchvision import models
from transformers import AutoModel


class MemeModel(nn.Module):
    """mode: text | image | fusion.  fusion: concat | gated."""

    def __init__(self, mode="fusion", fusion="concat", text_model="xlm-roberta-base",
                 n_classes=3, hidden=512, dropout=0.3):
        super().__init__()
        self.mode, self.fusion = mode, fusion
        t_dim = i_dim = 0
        if mode in ("text", "fusion"):
            # MPS scaled-dot-product attention does not support the encoder's
            # training-time dropout on this runtime. Eager attention keeps the
            # fine-tuning path compatible across Mac and other devices.
            self.text_enc = AutoModel.from_pretrained(text_model, attn_implementation="eager")
            t_dim = self.text_enc.config.hidden_size
        if mode in ("image", "fusion"):
            m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
            i_dim = m.fc.in_features
            m.fc = nn.Identity()
            self.img_enc = m
        if mode == "text":
            in_dim = t_dim
        elif mode == "image":
            in_dim = i_dim
        elif fusion == "gated":
            self.t_proj, self.i_proj = nn.Linear(t_dim, hidden), nn.Linear(i_dim, hidden)
            self.gate = nn.Linear(2 * hidden, hidden)
            in_dim = hidden
        else:
            in_dim = t_dim + i_dim
        self.head = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, n_classes))

    def encoder_params(self):
        for name in ("text_enc", "img_enc"):
            if hasattr(self, name):
                yield name, list(getattr(self, name).parameters())

    def head_params(self):
        names = ("head", "t_proj", "i_proj", "gate")
        return [p for n in names if hasattr(self, n) for p in getattr(self, n).parameters()]

    def forward(self, imgs, enc):
        t = i = None
        if self.mode in ("text", "fusion"):
            t = self.text_enc(**enc).last_hidden_state[:, 0]       # first-token vector
        if self.mode in ("image", "fusion"):
            i = self.img_enc(imgs)
        if self.mode == "text":
            z = t
        elif self.mode == "image":
            z = i
        elif self.fusion == "gated":
            tp, ip = torch.tanh(self.t_proj(t)), torch.tanh(self.i_proj(i))
            g = torch.sigmoid(self.gate(torch.cat([tp, ip], dim=1)))
            z = g * tp + (1 - g) * ip
        else:
            z = torch.cat([t, i], dim=1)
        return self.head(z)
