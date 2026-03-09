# DCGAN Image Generator

Deep Convolutional GAN that generates handwritten digit images, trained on MNIST.

## Architecture
- **Generator**: Latent z (100) → 7×7 → 14×14 → 28×28 via transposed convolutions
- **Discriminator**: 28×28 → LeakyReLU conv blocks → sigmoid output

## Features
- Label smoothing (0.9 for real, 0.1 for fake) for training stability
- Weight initialization from N(0, 0.02)
- Latent space interpolation visualization
- Sample grid saved every 5 epochs

## Setup

```bash
pip install -r requirements.txt
python main.py
```

## Output
- `final_samples.png` — 64 generated digit images
- `interpolation.png` — smooth latent space walk
- `gan_loss.png` — generator and discriminator loss
- `generator.pth`, `discriminator.pth` — saved models
