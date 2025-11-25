import math
import torch
import torch.nn as nn

class LoRALinear(nn.Module):
    """
    LoRA for Attention/Linear layers inside AudioLDM.

    Hyperparam defaults follow audio diffusion recommendations:
    - rank (r) = 8
    - alpha = 2 * r  (balance stability vs capacity)
    - dropout = 0.05 to reduce overfitting in audio training
    """

    def __init__(self, base_layer, r=8, alpha=None, dropout=0.05):
        super().__init__()
        self.base = base_layer
        in_dim = base_layer.in_features
        out_dim = base_layer.out_features

        self.r = r
        # alpha defaults to 2*r if not provided
        self.alpha = 2 * r if alpha is None else alpha
        self.scaling = self.alpha / self.r

        self.lora_A = nn.Linear(in_dim, r, bias=False)
        self.lora_B = nn.Linear(r, out_dim, bias=False)

        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

        # optional dropout for better generalization in audio data
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # mark only LoRA params as trainable
        for p in self.base.parameters():
            p.requires_grad = False

    def forward(self, x):
        return self.base(x) + self.scaling * self.lora_B(self.dropout(self.lora_A(x)))
