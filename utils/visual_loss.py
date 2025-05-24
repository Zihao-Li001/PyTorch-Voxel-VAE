import matplotlib.pyplot as plt
import torch
"""
This function plots the training loss, reconstruction loss, and KL divergence loss over epochs.
Parameters:
    loss_history (list): List of total loss values over epochs.
    recon_history (list): List of reconstruction loss values over epochs.
    kl_history (list): List of KL divergence loss values over epochs.
"""
def calculate_metrics(recon, target):
    with torch.no_grad():
        pred = recon > 0.5
        target = target > 0.5

        solid_voxel_accuracy = (pred[target]).float().mean()
        empty_voxel_accuracy = (pred[~target]).float().mean()

        intersection = (pred & target).float().sum().item()
        union = (pred | target).float().sum().item()
        iou = intersection / union if union > 0 else 0

        return solid_voxel_accuracy, empty_voxel_accuracy, iou

def plot_loss(loss_history, recon_history, kl_history, loss_beta):
    plt.plot(loss_history, label='Total Loss')
    plt.plot(recon_history, label='Reconstruction Loss')
    plt.plot(kl_history, label='KL Divergence')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.title('Training Losses')
    str_beta = 'beta_' + str(loss_beta)
    plt.savefig('vae_loss_'+ str_beta + '.png')
    plt.show()