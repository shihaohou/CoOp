"""Full-model gradient inversion attack with a small CIFAR-10 ResNet-18.

This is the **Full-model FL positive control** row in the Fig.7-style
figure: an ordinary federated client would share the full model
gradient of a small image classifier, and a white-box server can
reconstruct the input image from that gradient (Zhu et al. DLG,
Geiping et al. Inverting Gradients). We use:

  * ResNet-18 adapted for 32x32 CIFAR-10 (3x3 stem conv, no maxpool)
  * cosine gradient-matching loss + TV regularization
  * signed gradient updates (Geiping et al.)
  * cosine LR schedule over the attack budget

100 iters on 32x32 CIFAR-10 is usually enough to get a recognizable
reconstruction, matching the Iter 0/20/40/60/80/100 columns shown in
PromptFL's Fig. 7.

In contrast, privacy_attack_coop.py (the PromptFL row) shares only the
prompt gradient and fails to reconstruct in the same budget.
"""

import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.utils as vutils
from PIL import Image
from torchvision.models import resnet18


CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)


def make_resnet18(num_classes: int) -> nn.Module:
    """ResNet-18 adapted for 32x32 inputs (standard CIFAR conv stem)."""
    model = resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


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


def total_variation(x: torch.Tensor) -> torch.Tensor:
    dh = (x[..., 1:, :] - x[..., :-1, :]).abs().mean()
    dw = (x[..., :, 1:] - x[..., :, :-1]).abs().mean()
    return dh + dw


def grad_cosine_loss(dummy_grads, target_grads) -> torch.Tensor:
    loss = torch.tensor(0.0, device=target_grads[0].device)
    n = 0
    for gd, gt in zip(dummy_grads, target_grads):
        if gd is None or gt is None:
            continue
        gd = gd.reshape(-1)
        gt = gt.reshape(-1)
        loss = loss + (1.0 - F.cosine_similarity(gd, gt, dim=0))
        n += 1
    return loss / max(n, 1)


# ---------------------------------------------------------------------------
# Attack
# ---------------------------------------------------------------------------


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

    model = make_resnet18(len(classnames)).to(device)
    model.eval()
    shared_params = [p for p in model.parameters() if p.requires_grad]
    print(f"signal=full (ResNet-18) | #shared params = "
          f"{sum(p.numel() for p in shared_params):,}")

    # 1. Target gradient on the real image.
    x = load_image(args.image_path, args.image_size, device)
    y = torch.tensor([args.label], device=device, dtype=torch.long)
    vutils.save_image(denormalize_to_unit(x), os.path.join(args.output, "original.png"))

    logits = model(x)
    target_loss = F.cross_entropy(logits, y)
    target_grads = torch.autograd.grad(target_loss, shared_params,
                                       retain_graph=False, create_graph=False)
    target_grads = [g.detach() for g in target_grads]
    print(f"target CE loss = {target_loss.item():.4f}")

    # 2. Dummy image init in normalized space.
    image_shape = (1, 3, args.image_size, args.image_size)
    x_dummy = torch.randn(image_shape, device=device)
    x_dummy.requires_grad_(True)
    optimizer = torch.optim.Adam([x_dummy], lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.attack_iters, 1))

    # 3. Attack loop.
    for it in range(args.attack_iters + 1):
        if it in save_iters:
            with torch.no_grad():
                snap = denormalize_to_unit(x_dummy)
                vutils.save_image(snap, os.path.join(args.output, f"iter_{it:03d}.png"))

        if it == args.attack_iters:
            break

        optimizer.zero_grad(set_to_none=True)

        logits_hat = model(x_dummy)
        loss_hat = F.cross_entropy(logits_hat, y)
        dummy_grads = torch.autograd.grad(
            loss_hat, shared_params,
            create_graph=True, retain_graph=True,
        )

        g_loss = grad_cosine_loss(dummy_grads, target_grads)
        tv = total_variation(x_dummy) * args.tv_weight
        attack_loss = g_loss + tv

        attack_loss.backward()
        if args.signed:
            with torch.no_grad():
                x_dummy.grad.sign_()
        optimizer.step()
        scheduler.step()

        if it % args.log_every == 0:
            print(f"[full-resnet18] iter {it:4d} | "
                  f"g_loss {g_loss.item():.4f} | tv {tv.item():.4f}")

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
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--tv-weight", type=float, default=1e-2)
    p.add_argument("--signed", action="store_true", default=True,
                   help="signed gradient updates (Geiping et al.); on by default")
    p.add_argument("--no-signed", dest="signed", action="store_false")

    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--log-every", type=int, default=10)
    return p.parse_args()


if __name__ == "__main__":
    run_attack(parse_args())
