import torch
import numpy as np
from torchvision import transforms as tfs


patch_size = 14

def init_dino(device):
    model = torch.hub.load(
        "facebookresearch/dinov2",
        "dinov2_vitb14",
    )
    
    model = model.to(device).eval()
    return model

@torch.no_grad
def get_dino_features(device, dino_model, img, grid):
    transform = tfs.Compose(
        [
            tfs.Resize((518, 518)),
            tfs.ToTensor(),
            tfs.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    img = transform(img)[:3].unsqueeze(0).to(device)
    features = dino_model.get_intermediate_layers(img, n=1)[0].half()
    h, w = int(img.shape[2] / patch_size), int(img.shape[3] / patch_size)
    dim = features.shape[-1]
    features = features.reshape(-1, h, w, dim).permute(0, 3, 1, 2)
    features = torch.nn.functional.grid_sample(
        features, grid, align_corners=False
    ).reshape(1, 768, -1)
    features = torch.nn.functional.normalize(features, dim=1)
    return features
