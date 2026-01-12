import os
import math
import argparse
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils as vutils

# Optional metrics
try:
    import torchmetrics
    from torchmetrics.image.inception import InceptionScore
    from torchmetrics.image.fid import FrechetInceptionDistance
    _HAS_TORCHMETRICS = True
except Exception:
    _HAS_TORCHMETRICS = False

# ===============================================================
# Utilities
# ===============================================================

def default_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def normalize_to_neg_one_to_one(x):
    return x * 2.0 - 1.0

def unnormalize_to_zero_to_one(x):
    return (x + 1.0) * 0.5

def set_seed(seed: int = 42):
    import random, numpy as np
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def make_sigma_schedule(K: int = 10, sigma_min: float = 0.01, sigma_max: float = 1.0):
    return torch.linspace(sigma_min, sigma_max, K)

# ===============================================================
# Basic Blocks (your UNet style)
# ===============================================================

def getAutoPaddingSize(kernelSize, paddingSize=None):
    if paddingSize is None:
        if isinstance(kernelSize, int):
            paddingSize = kernelSize // 2
        else:
            paddingSize = [x // 2 for x in kernelSize]
    return paddingSize

class Conv(nn.Module):
    def __init__(self, inChannelNum, outChannelNum, kernelSize=1, strideSize=1, paddingSize=None, groupSize=1, hasActivation=True):
        super().__init__()
        self.conv = nn.Conv2d(inChannelNum, outChannelNum, kernelSize, strideSize,
                              getAutoPaddingSize(kernelSize, paddingSize), groups=groupSize, bias=False)
        self.batchNorm = nn.BatchNorm2d(outChannelNum)
        self.activation = nn.ReLU() if hasActivation is True else (hasActivation if isinstance(hasActivation, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.activation(self.batchNorm(self.conv(x)))

class ConvTranspose(nn.Module):
    def __init__(self, inChannelNum, outChannelNum, kernelSize=2, strideSize=2, paddingSize=0, hasBatchNorm=True, hasActivation=True):
        super().__init__()
        self.convTranspose = nn.ConvTranspose2d(inChannelNum, outChannelNum, kernelSize, strideSize,
                                                paddingSize, bias=not hasBatchNorm)
        self.batchNorm = nn.BatchNorm2d(outChannelNum) if hasBatchNorm else nn.Identity()
        self.activation = nn.ReLU() if hasActivation is True else (hasActivation if isinstance(hasActivation, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.activation(self.batchNorm(self.convTranspose(x)))

class Concat(nn.Module):
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension
    def forward(self, x):
        return torch.cat(x, self.d)

# ===============================================================
# σ Embedding + FiLM
# ===============================================================

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    def forward(self, x):
        device = x.device
        half = self.dim // 2
        freq = torch.exp(torch.arange(half, device=device) * (-math.log(10000.0) / (half - 1)))
        args = x[:, None] * freq[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0,1))
        return emb

class FiLM(nn.Module):
    def __init__(self, emb_dim: int, channels: int):
        super().__init__()
        self.net = nn.Sequential(nn.SiLU(), nn.Linear(emb_dim, channels * 2))
        self.act = nn.ReLU(inplace=True)
    def forward(self, x, emb):
        scale, shift = self.net(emb).chunk(2, dim=1)
        x = x * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        return self.act(x)

# ===============================================================
# Encoder & Decoder Blocks
# ===============================================================

class ContractPath(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, emb_dim: int):
        super().__init__()
        self.conv1 = Conv(in_ch, out_ch, 3, 1, hasActivation=False)
        self.film1 = FiLM(emb_dim, out_ch)
        self.conv2 = Conv(out_ch, out_ch, 3, 1, hasActivation=False)
        self.film2 = FiLM(emb_dim, out_ch)
        self.down = Conv(out_ch, out_ch, 4, 2, 1, hasActivation=True)

    def forward(self, x, emb):
        h = self.conv1(x); h = self.film1(h, emb)
        h = self.conv2(h); h = self.film2(h, emb)
        skip = h
        x = self.down(h)
        return x, skip

class ExtractPath(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, emb_dim: int):
        super().__init__()
        self.up = ConvTranspose(in_ch, out_ch, 2, 2, 0, hasBatchNorm=True, hasActivation=True)
        self.concat = Concat(1)
        self.conv1 = Conv(out_ch + skip_ch, out_ch, 3, 1, hasActivation=False)
        self.film1 = FiLM(emb_dim, out_ch)
        self.conv2 = Conv(out_ch, out_ch, 3, 1, hasActivation=False)
        self.film2 = FiLM(emb_dim, out_ch)

    def forward(self, skip, x, emb):
        x = self.up(x)
        # align sizes
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="nearest")
        x = self.concat([skip, x])
        h = self.conv1(x); h = self.film1(h, emb)
        h = self.conv2(h); h = self.film2(h, emb)
        return h

# ===============================================================
# UNetScore Network (σ-conditioned)
# ===============================================================

class UNetScore(nn.Module):
    def __init__(self, pathBlockNum: int = 3, base_ch: int = 64, emb_dim: int = 128, in_ch: int = 1, out_ch: int = 1):
        super().__init__()
        # σ embedding
        self.sigma_mlp = nn.Sequential(
            SinusoidalPosEmb(emb_dim),
            nn.Linear(emb_dim, emb_dim * 4),
            nn.SiLU(),
            nn.Linear(emb_dim * 4, emb_dim)
        )

        # Encoder
        self.contractPaths = nn.ModuleList()
        ch = base_ch
        enc_channels = [ch]
        self.contractPaths.append(ContractPath(in_ch, ch, emb_dim))
        for _ in range(pathBlockNum - 1):
            self.contractPaths.append(ContractPath(ch, ch * 2, emb_dim))
            ch *= 2
            enc_channels.append(ch)

        # Bottleneck
        self.b1 = Conv(ch, ch * 2, 3, 1, hasActivation=False)
        self.bf1 = FiLM(emb_dim, ch * 2)
        self.b2 = Conv(ch * 2, ch, 3, 1, hasActivation=False)
        self.bf2 = FiLM(emb_dim, ch)

        # Decoder
        self.extractPaths = nn.ModuleList()
        for skip_ch in reversed(enc_channels):
            self.extractPaths.append(ExtractPath(ch, skip_ch, ch // 2, emb_dim))
            ch //= 2

        self.fullyConv = Conv(ch, out_ch, 1, hasActivation=False)

    def forward(self, x, sigma):
        emb = self.sigma_mlp(torch.log(sigma.view(-1) + 1e-8))
        skips = []
        for path in self.contractPaths:
            x, skip = path(x, emb)
            skips.append(skip)
        x = self.b1(x); x = self.bf1(x, emb)
        x = self.b2(x); x = self.bf2(x, emb)
        for path, skip in zip(self.extractPaths, reversed(skips)):
            x = path(skip, x, emb)
        return self.fullyConv(x)

# ===============================================================
# DSM Loss
# ===============================================================

@dataclass
class DSMConfig:
    K: int = 10
    sigma_min: float = 0.01
    sigma_max: float = 1.0

class MultiscaleDSMLoss(nn.Module):
    def __init__(self, sigmas: torch.Tensor):
        super().__init__()
        self.register_buffer('sigmas', sigmas)

    def forward(self, model, x):
        B = x.size(0)
        device = x.device
        idx = torch.randint(low=0, high=self.sigmas.numel(), size=(B,), device=device)
        sigma = self.sigmas[idx]
        noise = torch.randn_like(x)
        sigma_view = sigma.view(B, *([1] * (x.dim()-1)))
        x_tilde = x + sigma_view * noise
        target = -(x_tilde - x) / (sigma_view ** 2)
        pred = model(x_tilde, sigma)
        lam = (sigma ** 2).view(B, *([1] * (x.dim()-1)))
        loss = 0.5 * lam * (pred - target) ** 2
        loss = loss.view(B, -1).sum(dim=1).mean()
        return loss, sigma

# ===============================================================
# Data
# ===============================================================

def get_mnist_loaders(batch_size=128, num_workers=4):
    tfm = transforms.Compose([transforms.ToTensor(), transforms.Lambda(normalize_to_neg_one_to_one)])
    train = datasets.MNIST('./data', train=True, transform=tfm, download=True)
    test  = datasets.MNIST('./data', train=False, transform=tfm, download=True)
    return (DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True),
            DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=num_workers))

# ===============================================================
# Training
# ===============================================================

def train(args):
    set_seed(args.seed)
    device = default_device()
    os.makedirs(args.outdir, exist_ok=True)
    log_path = os.path.join(args.outdir, "train_log.tsv")
    ckpt_path = os.path.join(args.outdir, "model.pt")

    sigmas = make_sigma_schedule(K=args.K, sigma_min=args.sigma_min, sigma_max=args.sigma_max).to(device)
    loss_fn = MultiscaleDSMLoss(sigmas)
    model = UNetScore(pathBlockNum=args.depth, base_ch=args.width).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_loader, _ = get_mnist_loaders(batch_size=args.batch_size)
    step = 0
    with open(log_path, "w") as f:
        f.write("step\tloss\tavg_sigma\n")

    model.train()
    print("Training started...")
    for epoch in range(args.epochs):
        for x, _ in train_loader:
            x = x.to(device)
            loss, sigma = loss_fn(model, x)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            step += 1
            if step % args.log_interval == 0:
                print(f"[{step}] loss={loss.item():.5f} sigma={sigma.mean().item():.4f}")
                with open(log_path, "a") as f:
                    f.write(f"{step}\t{loss.item():.6f}\t{sigma.mean().item():.5f}\n")
            if step % args.ckpt_interval == 0:
                torch.save({"model": model.state_dict(), "sigmas": sigmas}, ckpt_path)
        torch.save({"model": model.state_dict(), "sigmas": sigmas}, ckpt_path)

# ===============================================================
# Plot
# ===============================================================

def plot_loss(log_path, out_png):
    import pandas as pd, matplotlib.pyplot as plt
    df = pd.read_csv(log_path, sep="\t")
    plt.plot(df["step"], df["loss"])
    plt.xlabel("Step"); plt.ylabel("DSM Loss")
    plt.title("Training DSM Loss vs Steps")
    plt.tight_layout(); plt.savefig(out_png, dpi=150)
    print("Saved:", out_png)

# ===============================================================
# Sampling (Annealed Langevin Dynamics)
# ===============================================================


@torch.no_grad()
def ald_sample(model, sigmas, n, steps_per_level=75, c=0.1, device=None):
    """
    Exact implementation of Algorithm 1 from
    Song & Ermon (2019), Noise Conditional Score Networks.
    """
    device = default_device() if device is None else device
    model.eval()

    # Step-size scaling parameter ε ≡ c, following the paper
    sigma_L = sigmas[-1].item()
    x = torch.randn(n, 1, 28, 28, device=device)

    for s in reversed(sigmas):
        sigma_i = s.item()
        alpha_i = c * (sigma_i ** 2) / (sigma_L ** 2)      # line 3 of Alg 1
        for _ in range(steps_per_level):
            z = torch.randn_like(x)
            score = model(x, torch.full((n,), sigma_i, device=device))
            x = x + 0.5 * alpha_i * score + torch.sqrt(torch.tensor(alpha_i, device=device)) * z
        # line 8 of Alg 1: reuse final sample for next σ-level
    return x



def sample(args):
    device = default_device()
    ckpt = torch.load(args.ckpt, map_location=device)
    sigmas = ckpt["sigmas"].to(device)
    model = UNetScore(pathBlockNum=args.depth, base_ch=args.width).to(device)
    model.load_state_dict(ckpt["model"])
    imgs = ald_sample(model, sigmas, n=args.n_samples, steps_per_level=args.steps_per_level, c=args.c, device=device)
    imgs = unnormalize_to_zero_to_one(imgs.clamp(-1, 1))
    os.makedirs(args.outdir, exist_ok=True)
    grid = os.path.join(args.outdir, "samples_grid.png")
    vutils.save_image(imgs, grid, nrow=int(math.sqrt(args.n_samples)))
    print("Saved samples to:", grid)

# ===============================================================
# Evaluation (FID and IS)
# ===============================================================

@torch.no_grad()
def eval_metrics(args):
    device = default_device()
    os.makedirs(args.outdir, exist_ok=True)

    ckpt = torch.load(args.ckpt, map_location=device)
    sigmas = ckpt["sigmas"].to(device)
    model = UNetScore(pathBlockNum=args.depth, base_ch=args.width).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # 1) Generate samples
    n_gen = args.n_gen
    bs = args.batch_size
    ims = []
    for i in range(0, n_gen, bs):
        n = min(bs, n_gen - i)
        xg = ald_sample(model, sigmas, n=n, steps_per_level=args.steps_per_level, c=args.c, device=device)
        xg = unnormalize_to_zero_to_one(xg.clamp(-1, 1))
        ims.append(xg.cpu())
    samples = torch.cat(ims, dim=0)
    samples_rgb = samples.repeat(1, 3, 1, 1)
    samples_rgb = F.interpolate(samples_rgb, size=(299, 299), mode="bilinear", align_corners=False)

    # 2) Real MNIST
    _, test_loader = get_mnist_loaders(batch_size=bs)
    reals = []
    for x, _ in test_loader:
        reals.append(unnormalize_to_zero_to_one(x).cpu())
        if len(reals) * bs >= n_gen:
            break
    reals = torch.cat(reals, dim=0)[:n_gen]
    reals_rgb = reals.repeat(1, 3, 1, 1)
    reals_rgb = F.interpolate(reals_rgb, size=(299, 299), mode="bilinear", align_corners=False)

    # 3) Compute metrics
    if _HAS_TORCHMETRICS:
        print("Using torchmetrics for IS and FID...")
        is_metric = InceptionScore(normalize=True).to(device)
        fid_metric = FrechetInceptionDistance(normalize=True).to(device)

        # IS
        for i in range(0, samples_rgb.size(0), bs):
            is_metric.update(samples_rgb[i:i+bs].to(device))
        is_mean, is_std = is_metric.compute()
        print(f"Inception Score: {is_mean.item():.4f} ± {is_std.item():.4f}")

        # FID
        for i in range(0, reals_rgb.size(0), bs):
            fid_metric.update(reals_rgb[i:i+bs].to(device), real=True)
        for i in range(0, samples_rgb.size(0), bs):
            fid_metric.update(samples_rgb[i:i+bs].to(device), real=False)
        fid = fid_metric.compute().item()
        print(f"FID: {fid:.4f}")
    else:
        print("torchmetrics not available; please install torchmetrics for FID/IS.")


# Extend CLI
def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train")
    t.add_argument("--outdir", type=str, default="./runs/mnist_dsm_assignment")
    t.add_argument("--epochs", type=int, default=10)
    t.add_argument("--batch-size", type=int, default=128)
    t.add_argument("--lr", type=float, default=1e-4)
    t.add_argument("--width", type=int, default=64)
    t.add_argument("--depth", type=int, default=3)
    t.add_argument("--K", type=int, default=10)
    t.add_argument("--sigma-min", type=float, default=0.01)
    t.add_argument("--sigma-max", type=float, default=1.0)
    t.add_argument("--grad-clip", type=float, default=1.0)
    t.add_argument("--log-interval", type=int, default=50)
    t.add_argument("--ckpt-interval", type=int, default=1000)
    t.add_argument("--seed", type=int, default=42)

    s = sub.add_parser("sample")
    s.add_argument("--ckpt", required=True)
    s.add_argument("--outdir", default="./runs/mnist_dsm_assignment")
    s.add_argument("--width", type=int, default=64)
    s.add_argument("--depth", type=int, default=3)
    s.add_argument("--n-samples", type=int, default=64)
    s.add_argument("--steps-per-level", type=int, default=75)
    s.add_argument("--c", type=float, default=0.1)

    p = sub.add_parser("plot")
    p.add_argument("--log", required=True)
    p.add_argument("--out", default="./runs/mnist_dsm_assignment/loss.png")

    e = sub.add_parser("eval")
    e.add_argument("--ckpt", required=True)
    e.add_argument("--outdir", default="./runs/mnist_dsm_assignment")
    e.add_argument("--width", type=int, default=64)
    e.add_argument("--depth", type=int, default=3)
    e.add_argument("--n-gen", type=int, default=5000)
    e.add_argument("--steps-per-level", type=int, default=75)
    e.add_argument("--c", type=float, default=0.1)
    e.add_argument("--batch-size", type=int, default=64)

    args = parser.parse_args()
    if args.cmd == "train":
        train(args)
    elif args.cmd == "sample":
        sample(args)
    elif args.cmd == "plot":
        plot_loss(args.log, args.out)
    elif args.cmd == "eval":
        eval_metrics(args)


if __name__ == "__main__":
    main()
