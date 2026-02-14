"""Debug ViT encoder output with clean imports."""

import torch
import sys
from pathlib import Path

# Force reimport
for module in list(sys.modules.keys()):
    if 'film_layer' in module or 'climate_net' in module or 'encoder' in module:
        del sys.modules[module]

SRC_DIR = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from models.encoder import VisionTransformerEncoder, CNNEncoder
from models.climate_net import ClimateNet

# Test inputs
batch_size = 4
static = torch.randn(batch_size, 3, 35, 77) * 0.1
dynamic = torch.randn(batch_size, 16, 35, 77) * 0.1
lead_indices = torch.randint(0, 11, (batch_size,))

print("="*70)
print("VIT ENCODER TEST WITH CLEAN IMPORTS")
print("="*70)

vit_model = ClimateNet(
    encoder_type='vit',
    encoder_dim=512,
    encoder_blocks=6,
    vit_patch_size=7,
    vit_num_heads=8,
    vit_mlp_ratio=4.0,
    vit_dropout=0.1,
    vit_attention_dropout=0.1,
    use_film=True
)
vit_model.eval()

# Test FiLM initialization first
if hasattr(vit_model, 'film_layer'):
    test_cond = torch.randn(10, 128)
    test_gamma = vit_model.film_layer.gamma_fc(test_cond)
    test_beta = vit_model.film_layer.beta_fc(test_cond)
    
    print(f"\nFiLM Initialization Check:")
    print(f"  Gamma mean: {test_gamma.mean():.4f} (should be ~1.0)")
    print(f"  Beta mean: {test_beta.mean():.4f} (should be ~0.0)")
    
    if abs(test_gamma.mean() - 1.0) > 0.1:
        print("  ❌ WARNING: FiLM not properly initialized!")
    else:
        print("  ✅ FiLM correctly initialized")

# Now test full forward pass
with torch.no_grad():
    x = torch.cat([static, dynamic], dim=1)
    
    vit_encoded = vit_model.encoder(x)
    print(f"\nViT Encoded - min: {vit_encoded.min():.4f}, max: {vit_encoded.max():.4f}, mean: {vit_encoded.mean():.4f}, std: {vit_encoded.std():.4f}")
    
    if vit_model.use_film:
        lead_embed = vit_model.lead_embedding(lead_indices)
        vit_encoded_film = vit_model.film_layer(vit_encoded, lead_embed)
        
        print(f"After FiLM - min: {vit_encoded_film.min():.4f}, max: {vit_encoded_film.max():.4f}, mean: {vit_encoded_film.mean():.4f}, std: {vit_encoded_film.std():.4f}")
        
        vit_encoded = vit_encoded_film
    
    vit_outputs = vit_model.decoder(vit_encoded)
    
    print("\nViT Decoder outputs:")
    for var, pred in vit_outputs.items():
        print(f"  {var} - min: {pred.min():.4f}, max: {pred.max():.4f}, mean: {pred.mean():.4f}, std: {pred.std():.4f}")

print("\n" + "="*70)
if vit_encoded.std() > 0.1:
    print("✅ ViT output looks healthy! Ready to train.")
else:
    print("❌ ViT output still too small. FiLM initialization failed.")
print("="*70)
