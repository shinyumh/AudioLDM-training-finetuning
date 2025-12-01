import os, sys, torch
from omegaconf import OmegaConf
import torch.nn as nn

# ---------------- PATH SETUP ----------------
repo_root = "/Users/melodyhu/Desktop/genai/AudioLDM-training-finetuning"
sys.path.append(repo_root)

# ============================================================
#  FIX 1 — Fake taming package so imports don't break
# ============================================================
import types
fake_taming = types.ModuleType("taming")
fake_taming.modules = types.ModuleType("taming.modules")
fake_taming.modules.losses = types.ModuleType("taming.modules.losses")
fake_taming.modules.losses.vqperceptual = types.ModuleType(
    "taming.modules.losses.vqperceptual"
)

sys.modules["taming"] = fake_taming
sys.modules["taming.modules"] = fake_taming.modules
sys.modules["taming.modules.losses"] = fake_taming.modules.losses
sys.modules["taming.modules.losses.vqperceptual"] = fake_taming.modules.losses.vqperceptual

# ============================================================
#  FIX 2 — Dummy CLAP (replaces CLAPAudioEmbeddingClassifierFreev2)
# ============================================================
import audioldm_train.conditional_models as conditional_models

class DummyCLAP(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.embed_dim = 512
    def forward(self, *args, **kwargs):
        return torch.zeros(1, self.embed_dim)

conditional_models.CLAPAudioEmbeddingClassifierFreev2 = DummyCLAP

# ============================================================
#  FIX 3 — Dummy VAE Loss (replaces LPIPSWithDiscriminator)
# ============================================================
import audioldm_train.losses as losses

class DummyVAELoss(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
    def forward(self, *args, **kwargs):
        return torch.tensor(0.0), {}

losses.LPIPSWithDiscriminator = DummyVAELoss

# ============================================================
#  NOW import instantiate_from_config
# ============================================================
from audioldm_train.utilities.model_util import instantiate_from_config

# ---------------- LOAD CONFIG ----------------
config_path = os.path.join(
    repo_root,
    "audioldm_train/config/2023_08_23_reproduce_audioldm/audioldm_original.yaml",
)
print("Loading config:", config_path)
config = OmegaConf.load(config_path)

# ============================================================
#  FIX 4 — Disable REAL checkpoint loading (CPU-safe)
# ============================================================
try:
    cfg = config["model"]["params"]

    # Disable VAE checkpoint
    if (
        "first_stage_config" in cfg
        and "params" in cfg["first_stage_config"]
        and "reload_from_ckpt" in cfg["first_stage_config"]["params"]
    ):
        cfg["first_stage_config"]["params"]["reload_from_ckpt"] = ""
        print("[INFO] Disabled VAE checkpoint loading")

    # Disable DDPM checkpoint (main model)
    if "ckpt_path" in cfg:
        cfg["ckpt_path"] = ""
        print("[INFO] Disabled DDPM checkpoint loading")

except Exception as e:
    print("[WARN] Could not modify checkpoint paths:", e)

# ---------------- INSTANTIATE MODEL ----------------
print("Instantiating LatentDiffusion with dummy CLAP (CPU-safe)…")
model = instantiate_from_config(config["model"])

# ---------------- APPLY LORA FREEZING ----------------
for name, param in model.named_parameters():
    if "lora" not in name.lower():
        param.requires_grad = False

# ---------------- COUNT PARAMETERS ----------------
total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

print("\n===== PARAMETER SUMMARY =====")
print(f"Total parameters:      {total:,}")
print(f"Trainable parameters:  {trainable:,}")
print(f"Trainable %:           {100 * trainable / total:.6f}%")

print("\n===== TRAINABLE PARAMETER NAMES =====")
for name, p in model.named_parameters():
    if p.requires_grad:
        print(name, p.numel())
