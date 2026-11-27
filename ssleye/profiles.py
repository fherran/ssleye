"""What to expect from each SSL method."""

DINO_HEAD = ["dino/teacher_entropy", "dino/top_proto_share", "dino/protos_used", "dino/kl"]

PROFILES = {
    "auto": {"label": "Auto-detect", "family": "generic", "registers": None, "rope": None,
             "head_metrics": [], "dense_watch": False,
             "about": "Nothing assumed; everything is detected from the encoder."},
    "simclr": {"label": "SimCLR", "family": "contrastive", "registers": None, "rope": None,
               "head_metrics": [], "dense_watch": False,
               "about": "Contrastive: two views pulled together, other images pushed apart."},
    "byol": {"label": "BYOL", "family": "distillation", "registers": None, "rope": None,
             "head_metrics": [], "dense_watch": False,
             "about": "Online network predicts an EMA target network; no negatives."},
    "mae": {"label": "MAE", "family": "masked", "registers": None, "rope": None,
            "head_metrics": [], "dense_watch": True,
            "about": "Masked autoencoder: reconstructs hidden patches."},
    "dino": {"label": "DINO (v1)", "family": "dino", "registers": 0, "rope": False,
             "head_metrics": DINO_HEAD, "dense_watch": False,
             "about": "Self-distillation: student matches an EMA teacher over prototypes; centring and sharpening avoid collapse; multi-crop."},
    "dinov2": {"label": "DINOv2", "family": "dino", "registers": 4, "rope": False,
               "head_metrics": DINO_HEAD + ["ibot_loss", "koleo"], "dense_watch": True,
               "about": "DINO + iBOT masked patch loss + KoLeo spreading + Sinkhorn-Knopp centring; register tokens."},
    "dinov3": {"label": "DINOv3", "family": "dino", "registers": 4, "rope": True,
               "head_metrics": DINO_HEAD + ["ibot_loss", "koleo", "gram_loss"], "dense_watch": True,
               "about": "DINOv2 objective + Gram anchoring that keeps patch features from degrading; RoPE; 4 register tokens; patch 16."},
    "ijepa": {"label": "I-JEPA", "family": "jepa", "registers": None, "rope": None,
              "head_metrics": ["pred_cosine"], "dense_watch": True,
              "about": "Joint-embedding predictive: a predictor guesses the target encoder's features of hidden target blocks from a context block. No pixels, no augmentation-heavy views."},
    "vjepa": {"label": "V-JEPA", "family": "jepa", "registers": None, "rope": None,
              "head_metrics": ["pred_cosine"], "dense_watch": True,
              "about": "I-JEPA for clips and images: hidden spatio-temporal tubes (or blocks in still images) are predicted in feature space."},
}

ALIASES = {"i-jepa": "ijepa", "i_jepa": "ijepa", "v-jepa": "vjepa", "v_jepa": "vjepa", "vjepa2": "vjepa","dinov1": "dino", "dino_v1": "dino", "dino-v1": "dino", "dino_v2": "dinov2", "dino-v2": "dinov2",
           "dino_v3": "dinov3", "dino-v3": "dinov3", "none": "auto", "": "auto"}


def get(name):
    key = (name or "auto").strip().lower()
    key = ALIASES.get(key, key)
    if key not in PROFILES:
        raise ValueError(f"unknown model {name!r}; choose one of: {', '.join(PROFILES)}")
    return key, dict(PROFILES[key])


def check(profile, inspector_info):
    """Warnings where the model doesn't match its profile."""
    out = []
    regs = profile.get("registers")
    pre = inspector_info.get("prefix_tokens")
    if regs is not None and pre is not None and pre != 1 + regs:
        out.append(f"{profile['label']} usually has 1 [CLS] + {regs} register tokens, but this model has {pre} extra tokens.")
    if profile.get("rope") and inspector_info.get("attention_method") == "qkv":
        out.append("This profile uses RoPE, but attention could only be rebuilt from qkv, which ignores RoPE: attention maps may be wrong.")
    if profile.get("family") == "dino" and inspector_info.get("attention") is None:
        out.append("No attention layer was found, so there are no attention maps (expected for ConvNeXt-style distilled models).")
    return out
