import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from tqdm import tqdm
from model import VAE
from utils.ShapeNet import ShapeNet
from utils.visual_loss import plot_loss, plot_metrics, calculate_metrics

learning_rate = 0.001
batch_size = 16
epoch_num = 100
beta = 0.1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

data_train = ShapeNet('datasets/dataset_voxels.tar')
# data_train = ShapeNet('datasets/shapenet10_chairs_nr.tar')
train_dataloader = DataLoader(data_train, batch_size=batch_size, shuffle=True)

loss_history = {'total_loss': [], 
                'recon_loss': [], 
                'kl_loss': [],
                'solid_acc': [], 
                'empty_acc': [], 
                'iou': []}


for epoch in range(epoch_num):
    model.train()
    epoch_metrics = {'total': 0, 'recon': 0, 'kl': 0, 
                    'solid_acc': 0, 'empty_acc': 0, 'iou': 0}

    for i, data in enumerate(tqdm(train_dataloader)):
        try:
            inputs = data.to(device).float()

            optimizer.zero_grad()
            outputs, mu, sigma = model(inputs)

            loss, recon_loss, kl_loss = model.loss(inputs, outputs, mu, sigma, beta)
            loss.backward()
            optimizer.step()

            # record metrics
            epoch_metrics['total'] += loss.item()
            epoch_metrics['recon'] += recon_loss.item()
            epoch_metrics['kl'] += kl_loss.item()
            # if i % 10 == 0:
            solid_voxel_accuracy, empty_voxel_accuracy, iou = calculate_metrics(outputs, inputs)
            epoch_metrics['solid_acc'] += solid_voxel_accuracy.item()
            epoch_metrics['empty_acc'] += empty_voxel_accuracy.item()
            epoch_metrics['iou'] += iou

        except Exception as e:
            print(f"Error at batch {i} in epoch {epoch}: {e}")
            torch.cuda.empty_cache()  # Clear GPU memory
            continue

    num_batches = len(train_dataloader)
    for key in epoch_metrics:
        epoch_metrics[key] /= num_batches

    loss_history['total_loss'].append(epoch_metrics['total'])
    loss_history['recon_loss'].append(epoch_metrics['recon'])
    loss_history['kl_loss'].append(epoch_metrics['kl'])
    loss_history['solid_acc'].append(epoch_metrics['solid_acc'])
    loss_history['empty_acc'].append(epoch_metrics['empty_acc'])
    loss_history['iou'].append(epoch_metrics['iou'])

    print(f"\nEpoch {epoch} Summary:")
    print(f"  Loss: {epoch_metrics['total']:.4f}, "
          f"Recon Loss: {epoch_metrics['recon']:.4f}, "
          f"KL Loss: {epoch_metrics['kl']:.4f}")
    print(f"  Solid Voxel Accuracy: {epoch_metrics['solid_acc']:.4f}, "
          f"Empty Voxel Accuracy: {epoch_metrics['empty_acc']:.4f}, "
          f"IOU: {epoch_metrics['iou']:.4f}")

plot_loss(loss_history['total_loss'], loss_history['recon_loss'], loss_history['kl_loss'], beta)
plot_metrics(loss_history['solid_acc'], loss_history['empty_acc'], loss_history['iou'], beta)

torch.save(model.state_dict(), "./models/vae.pth")

print('Finished Training')