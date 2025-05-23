import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from tqdm import tqdm
from model import VAE
from ShapeNet import ShapeNet
from utils.visual_loss import plot_loss

learning_rate = 0.005
batch_size = 10
epoch_num = 100

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VAE().to(device)
# print(model)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

data_train = ShapeNet('datasets/dataset_voxels.tar')
train_dataloader = DataLoader(data_train, batch_size=batch_size, shuffle=True)


# Debug @ check loss_history @ May 23,11:04 Li
loss_history = []
recon_history = []
kl_history = []

for epoch in range(epoch_num):
    model.train(True)
    total_loss = 0.0
    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0
    for i, data in enumerate(tqdm(train_dataloader)):
        inputs = data.to(device)

        optimizer.zero_grad()

        outputs, mu, sigma = model(inputs)
        loss, recon_loss, kl_loss = model.loss(inputs, outputs, mu, sigma)

        loss.backward()
        optimizer.step()

        # Debug @ check loss_history @_May 23,11:04 Li
        total_loss += loss.item()
        total_recon += recon_loss.item()
        total_kl += kl_loss.item()
        

    avg_total_loss = total_loss / len(train_dataloader)  

    # Debug @ check loss_history @_May 23,11:04 Li   
    avg_recon = total_recon / len(train_dataloader)
    avg_kl = total_kl / len(train_dataloader)

    loss_history.append(avg_total_loss)
    recon_history.append(avg_recon)
    kl_history.append(avg_kl)

    print("Epoch", epoch, "Average Loss", avg_total_loss)

plot_loss(loss_history, recon_history, kl_history)
print('Finished Training')

torch.save(model.state_dict(), "./models/vae.pth")