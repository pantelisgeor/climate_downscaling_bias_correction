"""FiLM layers."""
import torch
import torch.nn as nn


class LeadTimeEmbedding(nn.Module):
    def __init__(self, num_leads: int = 11, embed_dim: int = 128):
        super().__init__()
        self.embedding = nn.Embedding(num_leads, embed_dim)
        nn.init.normal_(self.embedding.weight, mean=0, std=0.02)
    
    def forward(self, lead_indices: torch.Tensor) -> torch.Tensor:
        return self.embedding(lead_indices)


class FiLMLayer(nn.Module):
    def __init__(self, num_features: int, cond_dim: int):
        super().__init__()
        
        self.gamma_fc = nn.Sequential(
            nn.Linear(cond_dim, cond_dim),
            nn.ReLU(),
            nn.Linear(cond_dim, num_features)
        )
        
        self.beta_fc = nn.Sequential(
            nn.Linear(cond_dim, cond_dim),
            nn.ReLU(),
            nn.Linear(cond_dim, num_features)
        )
        
        # Initialize gamma=1, beta=0 for identity transformation
        nn.init.zeros_(self.gamma_fc[0].weight)
        nn.init.zeros_(self.gamma_fc[0].bias)
        nn.init.zeros_(self.gamma_fc[2].weight)
        nn.init.ones_(self.gamma_fc[2].bias)
        
        nn.init.zeros_(self.beta_fc[0].weight)
        nn.init.zeros_(self.beta_fc[0].bias)
        nn.init.zeros_(self.beta_fc[2].weight)
        nn.init.zeros_(self.beta_fc[2].bias)
    
    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gamma = self.gamma_fc(cond).unsqueeze(-1).unsqueeze(-1)
        beta = self.beta_fc(cond).unsqueeze(-1).unsqueeze(-1)
        return gamma * x + beta
