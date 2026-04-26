import torch
import torch.nn as nn
import torch.nn.functional as F

class BitLinear158(nn.Linear):
    def forward(self, x):
        w = self.weight
        gamma = torch.mean(torch.abs(w))
        w_quant = torch.round(torch.clamp(w / (gamma + 1e-5), -1, 1))
        
        # Activation quantization (8-bit)
        x_quant = x - x.mean(dim=-1, keepdim=True)
        scale = 127 / torch.max(torch.abs(x_quant) + 1e-5)
        x_quant = torch.clamp(torch.round(x_quant * scale), -128, 127)
        
        out = F.linear(x_quant, w_quant)
        return out / (scale * gamma + 1e-5)