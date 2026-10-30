"""Small U-Net and conditional PatchGAN for a four-band inpainting experiment."""
import torch
from torch import nn
from torch.nn import functional as F


def block(a, b):
    return nn.Sequential(nn.Conv2d(a, b, 3, padding=1), nn.GroupNorm(4, b),
                         nn.SiLU(), nn.Conv2d(b, b, 3, padding=1),
                         nn.GroupNorm(4, b), nn.SiLU())


class Generator(nn.Module):
    def __init__(self, width=16):
        super().__init__()
        self.e1 = block(5, width)
        self.e2 = block(width, width * 2)
        self.e3 = block(width * 2, width * 4)
        self.middle = block(width * 4, width * 8)
        self.d3 = block(width * 12, width * 4)
        self.d2 = block(width * 6, width * 2)
        self.d1 = block(width * 3, width)
        self.head = nn.Conv2d(width, 4, 1)

    def features(self, image, mask):
        a = self.e1(torch.cat([image, mask], 1))
        b = self.e2(F.avg_pool2d(a, 2))
        c = self.e3(F.avg_pool2d(b, 2))
        z = self.middle(F.avg_pool2d(c, 2))
        for skip, decoder in [(c, self.d3), (b, self.d2), (a, self.d1)]:
            z = decoder(torch.cat([F.interpolate(z, size=skip.shape[-2:],
                                                 mode='bilinear', align_corners=False), skip], 1))
        return z

    def forward(self, image, mask):
        return image * (1 - mask) + self.head(self.features(image, mask)) * mask


class ContextResidualGenerator(Generator):
    """Start at observed-pixel mean fill and learn bounded reflectance corrections.

    Cloudy values are removed before feature extraction. Absolute scene colour
    is carried by the mean, rather than reconstructed after normalization.
    The legacy generator/checkpoints retain their original behavior.
    """
    def __init__(self, width=16):
        super().__init__(width)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, image, mask):
        valid = 1 - mask
        mean = (image * valid).sum((2, 3), keepdim=True) / valid.sum((2, 3), keepdim=True).clamp_min(1)
        centered = (image - mean) * valid / .1
        correction = .05 * torch.tanh(self.head(self.features(centered, mask)))
        candidate = mean + correction
        return image * valid + candidate * mask


def make_generator(checkpoint):
    architecture = checkpoint.get('architecture', 'legacy')
    if architecture == 'legacy':
        return Generator(checkpoint['width'])
    if architecture == 'context_residual_v2':
        return ContextResidualGenerator(checkpoint['width'])
    raise ValueError(f'Unknown generator architecture: {architecture}')


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(9, 16, 4, 2, 1), nn.LeakyReLU(.2),
                                 nn.Conv2d(16, 32, 4, 2, 1), nn.LeakyReLU(.2),
                                 nn.Conv2d(32, 64, 4, 2, 1), nn.LeakyReLU(.2),
                                 nn.Conv2d(64, 1, 3, padding=1))

    def forward(self, observed, mask, candidate):
        return self.net(torch.cat([observed, mask, candidate], 1))
