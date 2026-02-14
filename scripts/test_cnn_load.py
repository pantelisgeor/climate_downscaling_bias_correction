# test_checkpoint.py
import torch
from src.models.climate_net import ClimateNet

print("Creating model with updated FiLM layer...")
model = ClimateNet(encoder_type='cnn')

print("\nModel FiLM structure:")
for name, param in model.encoder.lead_embedding.named_parameters():
    print(f"  {name}: {param.shape}")
for name, param in model.encoder.dynamic_blocks[0].film.named_parameters():
    print(f"  {name}: {param.shape}")

print("\nLoading checkpoint...")
checkpoint = torch.load(
    "/nvme/h/pgeorgiades/data_p185/AI_downscale/NEW_models_2/experiments/" + 
    "climate_net_baseline_20260213_214618/checkpoints/best_model.pt",
    map_location='cpu'
)

model.load_state_dict(checkpoint['model_state_dict'])

print("✅ SUCCESS! Checkpoint loaded.")
print(f"  Epoch: {checkpoint['epoch']}")
print(f"  Best val loss: {checkpoint['best_val_loss']:.6f}")

# Test forward pass
print("\nTesting forward pass...")
static = torch.randn(2, 3, 35, 77)
dynamic = torch.randn(2, 16, 35, 77)
lead = torch.tensor([3, 7])

with torch.no_grad():
    outputs = model(static, dynamic, lead)
    for var, pred in outputs.items():
        print(f"  {var}: {pred.shape}")

print("\n✅ Everything works!")
