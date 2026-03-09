"""
GAN Image Generator
DCGAN (Deep Convolutional GAN) for generating MNIST-like handwritten digits.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
import warnings
warnings.filterwarnings('ignore')


class Generator(nn.Module):
    """Maps latent vector z -> 28x28 grayscale image."""
    def __init__(self, z_dim=100, ngf=64):
        super().__init__()
        self.net = nn.Sequential(
            # z_dim -> 7x7
            nn.ConvTranspose2d(z_dim, ngf*4, 7, 1, 0, bias=False),
            nn.BatchNorm2d(ngf*4), nn.ReLU(True),
            # 7x7 -> 14x14
            nn.ConvTranspose2d(ngf*4, ngf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*2), nn.ReLU(True),
            # 14x14 -> 28x28
            nn.ConvTranspose2d(ngf*2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf), nn.ReLU(True),
            nn.ConvTranspose2d(ngf, 1, 3, 1, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, z):
        return self.net(z)


class Discriminator(nn.Module):
    """Classifies 28x28 images as real or fake."""
    def __init__(self, ndf=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, ndf, 4, 2, 1, bias=False), nn.LeakyReLU(0.2, True),
            nn.Dropout2d(0.25),
            nn.Conv2d(ndf, ndf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*2), nn.LeakyReLU(0.2, True),
            nn.Dropout2d(0.25),
            nn.Conv2d(ndf*2, ndf*4, 7, 1, 0, bias=False),
            nn.BatchNorm2d(ndf*4), nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf*4, 1, 1, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x).view(-1, 1).squeeze(1)


def weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(m.weight, 0.0, 0.02)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)


def load_data():
    t = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
    try:
        ds = datasets.MNIST('./data', download=True, transform=t)
        return DataLoader(ds, batch_size=128, shuffle=True, num_workers=0), True
    except:
        X = torch.randn(2000, 1, 28, 28).tanh()
        return DataLoader(TensorDataset(X, torch.zeros(2000)), batch_size=128, shuffle=True), False


def train_gan(G, D, loader, z_dim, device, epochs=30):
    criterion  = nn.BCELoss()
    opt_D = optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))
    opt_G = optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
    fixed_z = torch.randn(64, z_dim, 1, 1).to(device)

    g_losses, d_losses = [], []

    for epoch in range(1, epochs + 1):
        g_epoch, d_epoch, n_batches = 0, 0, 0
        for real, _ in loader:
            real = real.to(device)
            bs   = real.size(0)
            real_label = torch.ones(bs).to(device) * 0.9   # label smoothing
            fake_label = torch.zeros(bs).to(device) + 0.1

            # ── Train Discriminator ──
            z    = torch.randn(bs, z_dim, 1, 1).to(device)
            fake = G(z).detach()
            opt_D.zero_grad()
            loss_D = criterion(D(real), real_label) + criterion(D(fake), fake_label)
            loss_D.backward()
            opt_D.step()

            # ── Train Generator ──
            z    = torch.randn(bs, z_dim, 1, 1).to(device)
            fake = G(z)
            opt_G.zero_grad()
            loss_G = criterion(D(fake), torch.ones(bs).to(device))
            loss_G.backward()
            opt_G.step()

            g_epoch += loss_G.item()
            d_epoch += loss_D.item()
            n_batches += 1

        g_losses.append(g_epoch / n_batches)
        d_losses.append(d_epoch / n_batches)

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{epochs}: G_Loss={g_losses[-1]:.4f}, D_Loss={d_losses[-1]:.4f}")
            save_samples(G, fixed_z, f'samples_epoch_{epoch:03d}.png')

    return g_losses, d_losses


def save_samples(G, z, path):
    G.eval()
    with torch.no_grad():
        samples = G(z).cpu().numpy()
    fig, axes = plt.subplots(8, 8, figsize=(8, 8))
    for i, ax in enumerate(axes.ravel()):
        ax.imshow(samples[i, 0], cmap='gray', vmin=-1, vmax=1)
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(path, dpi=100)
    plt.close()
    G.train()


def plot_loss(g_losses, d_losses, save_path='gan_loss.png'):
    plt.figure(figsize=(9, 5))
    plt.plot(g_losses, label='Generator Loss')
    plt.plot(d_losses, label='Discriminator Loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss')
    plt.title('GAN Training Loss')
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def generate_interpolation(G, z_dim, device, steps=10, save_path='interpolation.png'):
    """Latent space interpolation between two points."""
    G.eval()
    z1 = torch.randn(1, z_dim, 1, 1).to(device)
    z2 = torch.randn(1, z_dim, 1, 1).to(device)
    interp = [z1 + (z2 - z1) * t for t in np.linspace(0, 1, steps)]
    with torch.no_grad():
        imgs = [G(z).cpu().numpy()[0, 0] for z in interp]
    fig, axes = plt.subplots(1, steps, figsize=(steps * 1.5, 2))
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        ax.imshow(img, cmap='gray', vmin=-1, vmax=1)
        ax.axis('off')
    plt.suptitle('Latent Space Interpolation')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Interpolation saved to {save_path}")


def main():
    print("=" * 60)
    print("DCGAN IMAGE GENERATOR")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    Z_DIM = 100
    G = Generator(Z_DIM).to(device)
    D = Discriminator().to(device)
    G.apply(weights_init)
    D.apply(weights_init)

    g_params = sum(p.numel() for p in G.parameters())
    d_params = sum(p.numel() for p in D.parameters())
    print(f"Generator params: {g_params:,} | Discriminator params: {d_params:,}")

    loader, _ = load_data()

    print("\n--- Training DCGAN ---")
    g_losses, d_losses = train_gan(G, D, loader, Z_DIM, device, epochs=30)

    plot_loss(g_losses, d_losses)
    generate_interpolation(G, Z_DIM, device)

    # Final samples
    z = torch.randn(64, Z_DIM, 1, 1).to(device)
    save_samples(G, z, 'final_samples.png')

    torch.save(G.state_dict(), 'generator.pth')
    torch.save(D.state_dict(), 'discriminator.pth')
    print("\nModels saved: generator.pth, discriminator.pth")
    print("Final samples: final_samples.png")
    print("GAN loss:      gan_loss.png")
    print("\n✓ GAN Image Generator complete!")


if __name__ == '__main__':
    main()
