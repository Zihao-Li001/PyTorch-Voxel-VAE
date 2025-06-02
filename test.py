import os
import torch
from torch.utils.data import DataLoader
from torchinfo import summary

from model import VAE
from utils.ShapeNet import ShapeNet
from utils.save_volume import save_output
from utils.visual_loss import calculate_metrics

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VAE().to(device)
checkpoint = torch.load("./models/vae.pth", map_location=device)
model.load_state_dict(checkpoint)
model.eval()
summary(model,(1,1,32,32,32))

datasetName = 'testset'
dataset = ShapeNet('datasets/'+datasetName+'_voxels.tar')
train_dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

if not os.path.exists('reconstructions'):
    os.makedirs('reconstructions')

total_metrics = {
    'solid_acc': 0.0,
    'empty_acc': 0.0,
    'iou': 0.0
}
num_samples_processed = 0

print("\n--- Starting reconstruction and evaluation ---")

for i, data in enumerate(train_dataloader):
    sample = data.to(device)
    with torch.no_grad():
        reconstructions,_,_ = model(sample)
    reconstructions = torch.sigmoid(reconstructions)

    # reconstructions = reconstructions.view(1, 32, 32, 32)
    reconstructions_save = (reconstructions > 0.5).float()
    reconstructions_save = reconstructions_save.detach().cpu()
    save_output(reconstructions_save[0][0], 32, 'reconstructions', i)
    print(f"Saved reconstruction {i} to reconstructions/reconstruction_{i}.stl")

    metics = calculate_metrics(reconstructions, sample)
    total_metrics['solid_acc'] += metics[0].item()
    total_metrics['empty_acc'] += metics[1].item()
    total_metrics['iou'] += metics[2]
    num_samples_processed += 1
    # if i == 25:
    #     break
    if i != 0 and i % 100 == 0:
        break

if num_samples_processed > 0:
    for key in total_metrics:
        total_metrics[key] /= num_samples_processed
    print("\n--- Reconstruction and Evaluation Summary ---")
    print(f"Solid Voxel Accuracy: {total_metrics['solid_acc']:.4f}")
    print(f"Empty Voxel Accuracy: {total_metrics['empty_acc']:.4f}")
    print(f"IoU: {total_metrics['iou']:.4f}")
else:
    print("No samples processed. Check your dataset and DataLoader configuration.")