# import os
import torch
from torch.utils.data import DataLoader

from model import VAE
from ShapeNet import ShapeNet
from utils.save_volume import save_output

def extract_latent_code(model, train_dataloader, device):
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(train_dataloader):
            sample = data.to(device)
            _, _, z = model.encode(sample)
            print(f"Latent code for sample {i}: {z}")
            print(z.shape)
            if i != 0 and i % 100 == 0:
                break

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VAE().to(device)
    checkpoint = torch.load("./models/vae.pt")
    model.load_state_dict(checkpoint)
    model.eval()

    data_train = ShapeNet('datasets/dataset_voxels.tar')
    train_dataloader = DataLoader(data_train, batch_size=1, shuffle=False)

    extract_latent_code(model, train_dataloader, device) # 
