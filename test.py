import os
import torch
from torch.utils.data import DataLoader

from model import VAE
from ShapeNet import ShapeNet
from utils.save_volume import save_output

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VAE().to(device)
checkpoint = torch.load("./models/vae.pth", map_location=device)
model.load_state_dict(checkpoint)
model.eval()
# print(model)

data_train = ShapeNet('datasets/test_dataset_voxels')
train_dataloader = DataLoader(data_train, batch_size=1, shuffle=False)

if not os.path.exists('reconstructions'):
    os.makedirs('reconstructions')

for i, data in enumerate(train_dataloader):
    sample = data.to(device)

    reconstructions,_,_ = model(sample)
    reconstructions = torch.sigmoid(reconstructions)
    print(reconstructions.min(), reconstructions.max())
    # reconstructions = reconstructions.view(1, 32, 32, 32)
    reconstructions = (reconstructions > 0.5).float()
    reconstructions = reconstructions.detach().cpu()
    save_output(reconstructions[0][0], 32, 'reconstructions', i)

    print("Saved", i)

    if i != 0 and i % 100 == 0:
        break