import matplotlib.pyplot as plt
"""
This function plots the training loss, reconstruction loss, and KL divergence loss over epochs.
Parameters:
    loss_history (list): List of total loss values over epochs.
    recon_history (list): List of reconstruction loss values over epochs.
    kl_history (list): List of KL divergence loss values over epochs.
"""

def plot_loss(loss_history, recon_history, kl_history):
    plt.plot(loss_history, label='Total Loss')
    plt.plot(recon_history, label='Reconstruction Loss')
    plt.plot(kl_history, label='KL Divergence')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.title('Training Losses')
    plt.savefig('vae_loss_curve.png')
    plt.show()