"""DLG-style gradient inversion on a tiny Sigmoid LeNet (Zhu et al. 2019).

This is the **Full-model FL positive control** row in the Fig.7-style
figure. Setup follows the original Deep Leakage from Gradients paper
(NeurIPS 2019) as closely as possible:

  * 4-layer CNN: Conv -> Sigmoid -> Conv -> Sigmoid -> Conv -> Sigmoid
    -> Linear. ~5K parameters total, no BatchNorm, no ReLU, no residual
    connections (all bad for second-order gradient flow).
  * Label assumed known (batch size 1, white-box server).
  * L-BFGS optimizer with L2 grad-matching loss, no TV regularization.

On CIFAR-10 32x32 this reliably reconstructs the input within ~30-300
L-BFGS steps. In contrast, privacy_attack_coop.py (the PromptFL row)
shares only the CoOp prompt gradient and fails to reconstruct under
the same budget -- that is the contrast Fig. 7 is supposed to draw.
"""

import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.utils as vutils
from PIL import Image


CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)


class LeNetSigmoid(nn.Module):
    """The exact CNN used by Zhu et al. for CIFAR-10 in DLG.

    For 32x32 input: 32 -> 16 -> 8 -> 8 -> flatten(768) -> num_classes.
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()
        act = nn.Sigmoid
        self.body = nn.Sequential(
            nn.Conv2d(3, 12, kernel_size=5, padding=5 // 2, stride=2),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=2),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=1),
            act(),
        )
        self.fc = nn.Linear(768, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.body(x)
        return self.fc(h.flatten(1))


def _dlg_init(m: nn.Module) -> None:
    """Match the DLG reference implementation: uniform(-0.5, 0.5)."""
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.uniform_(m.weight, -0.5, 0.5)
        if m.bias is not None:
            nn.init.uniform_(m.bias, -0.5, 0.5)


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------


def _norm_consts(device: torch.device):
    mean = torch.tensor(CIFAR_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(CIFAR_STD, device=device).view(1, 3, 1, 1)
    return mean, std


def load_image(path: str, image_size: int, device: torch.device) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    tfm = T.Compose([
        T.Resize(image_size, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(image_size),
        T.ToTensor(),
        T.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    return tfm(img).unsqueeze(0).to(device)


def denormalize_to_unit(x: torch.Tensor) -> torch.Tensor:
    mean, std = _norm_consts(x.device)
    return (x * std + mean).clamp(0.0, 1.0)


# ---------------------------------------------------------------------------
# Attack
# ---------------------------------------------------------------------------


def grad_l2_loss(dummy_grads, target_grads) -> torch.Tensor:
    loss = torch.tensor(0.0, device=target_grads[0].device)
    for gd, gt in zip(dummy_grads, target_grads):
        if gd is None or gt is None:
            continue
        loss = loss + ((gd - gt) ** 2).sum()
    return loss


def run_attack(args):
    device = torch.device(args.device)
    torch.manual_seed(args.seed)

    save_iters = sorted({int(x) for x in args.save_iters.split(",") if x.strip()})
    os.makedirs(args.output, exist_ok=True)

    with open(args.classnames_file) as f:
        classnames = [ln.strip() for ln in f if ln.strip()]
    if not (0 <= args.label < len(classnames)):
        raise ValueError(
            f"--label {args.label} out of range for {len(classnames)} classnames")
    print(f"#classes = {len(classnames)} | target = {classnames[args.label]!r}")

    model = LeNetSigmoid(num_classes=len(classnames)).to(device)
    model.apply(_dlg_init)
    model.eval()
    shared_params = [p for p in model.parameters() if p.requires_grad]
    print(f"signal=full (DLG LeNet, Sigmoid) | #shared params = "
          f"{sum(p.numel() for p in shared_params):,}")

    # 1. Target gradient on the real image.
    x = load_image(args.image_path, args.image_size, device)
    y = torch.tensor([args.label], device=device, dtype=torch.long)
    vutils.save_image(denormalize_to_unit(x), os.path.join(args.output, "original.png"))

    logits = model(x)
    target_loss = F.cross_entropy(logits, y)
    target_grads = torch.autograd.grad(target_loss, shared_params)
    target_grads = [g.detach() for g in target_grads]
    print(f"target CE loss = {target_loss.item():.4f}")

    # 2. Dummy image init in normalized space.
    image_shape = (1, 3, args.image_size, args.image_size)
    x_dummy = torch.randn(image_shape, device=device)
    x_dummy.requires_grad_(True)

    if args.optimizer == "lbfgs":
        optimizer = torch.optim.LBFGS([x_dummy], lr=args.lr,
                                      max_iter=20, history_size=100)
    else:
        optimizer = torch.optim.Adam([x_dummy], lr=args.lr)

    def save_snapshot(it: int) -> None:
        with torch.no_grad():
            snap = denormalize_to_unit(x_dummy)
            vutils.save_image(snap, os.path.join(args.output, f"iter_{it:03d}.png"))

    # Iter 0 = initial random image, save before any optimization step.
    if 0 in save_iters:
        save_snapshot(0)

    # 3. Attack loop. For L-BFGS, each .step() runs an inner line search,
    # so the closure can be called several times per outer iteration.
    for it in range(1, args.attack_iters + 1):
        last_loss = {"value": None}

        def closure():
            optimizer.zero_grad(set_to_none=True)
            logits_hat = model(x_dummy)
            loss_hat = F.cross_entropy(logits_hat, y)
            dummy_grads = torch.autograd.grad(
                loss_hat, shared_params, create_graph=True)
            g_loss = grad_l2_loss(dummy_grads, target_grads)
            g_loss.backward()
            last_loss["value"] = g_loss.item()
            return g_loss

        optimizer.step(closure)

        if it in save_iters:
            save_snapshot(it)

        if it % args.log_every == 0:
            lv = last_loss["value"]
            print(f"[dlg-lenet] iter {it:4d} | grad-match L2 = "
                  f"{(lv if lv is not None else float('nan')):.6f}")

    print(f"Saved snapshots and original.png under {args.output}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--image-path", required=True)
    p.add_argument("--label", type=int, required=True)
    p.add_argument("--classnames-file", required=True)
    p.add_argument("--output", required=True)

    p.add_argument("--image-size", type=int, default=32)
    p.add_argument("--attack-iters", type=int, default=100)
    p.add_argument("--save-iters", default="0,20,40,60,80,100")
    p.add_argument("--optimizer", choices=["lbfgs", "adam"], default="lbfgs",
                   help="L-BFGS is the canonical DLG choice and converges fastest")
    p.add_argument("--lr", type=float, default=1.0,
                   help="default 1.0 for L-BFGS; try 0.1 if using --optimizer adam")

    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--log-every", type=int, default=10)
    return p.parse_args()


if __name__ == "__main__":
    run_attack(parse_args())
