import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from tqdm import tqdm
from model import VAE
from ShapeNetChairs import ShapeNetChairs

learning_rate = 0.005
momentum = 0.9
batch_size = 10
epoch_num = 150

# beta is a hyperparameter that controls the weight of the KL divergence term 
# @ Zihao_Li
beta = 0.0001

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VAE().to(device)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)

data_train = ShapeNetChairs('datasets/shapenet10_chairs_nr.tar')
train_dataloader = DataLoader(data_train, batch_size=batch_size, shuffle=True)

for epoch in range(epoch_num):
    model.train(True)
    epoch_loss = {'total_loss': 0, 'recon_loss': 0, 'kl_loss': 0}
    for i, data in enumerate(tqdm(train_dataloader)):
        inputs_for_model = data.to(device).float()
        inputs_for_loss = inputs_for_model.clamp(0, 1)
        optimizer.zero_grad()

        outputs, mu, logvar = model(inputs_for_model) # @ Zihao_Li
        
        loss, recon_loss, kl_loss = model.loss(inputs_for_loss,  # @ Zihao_Li
                          outputs,
                          mu,
                          logvar,
                          beta)

        loss.backward()
        optimizer.step()

        epoch_loss['total_loss'] += loss.item()
        epoch_loss['recon_loss'] += recon_loss.item()
        epoch_loss['kl_loss'] += kl_loss.item()
    loss = epoch_loss['total_loss'] / len(train_dataloader)
    recon_loss = epoch_loss['recon_loss'] / len(train_dataloader)
    kl_loss = epoch_loss['kl_loss'] / len(train_dataloader)
    
    print("Epoch", epoch, "Loss", epoch_loss['total_loss'], "recon_loss", epoch_loss['recon_loss'], "kl_loss", epoch_loss['kl_loss'])

print('Finished Training')

torch.save(model.state_dict(), "./models/vae.pt")