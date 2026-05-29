"""Prompt-only gradient inversion attack at native CIFAR-10 32x32.

This is the **PromptFL / CoOp / CAPT** row in the Fig.7-style figure,
designed to pair with ``privacy_attack_dlg.py`` (the full-model row).
Both attacks operate at 32x32 so the resulting cells can be stitched
together with no resampling.

Setup:

  * Frozen image encoder F: the same Sigmoid LeNet body used in DLG
    (3 conv + sigmoid), plus a small linear projection to ``embed_dim``.
    The server already knows F (analog to CLIP being public in CoOp).
  * Learnable per-class prompt embeddings P of shape
    ``[num_classes, embed_dim]`` -- the only thing the server sees a
    gradient of, mirroring CoOp's ``prompt_learner.ctx`` / CAPT's
    class-aware prompts.
  * ``logits = normalize(F(x)) @ normalize(P).T``.

Threat model:

  * White-box server (F, P, loss known).
  * Server observes ``target_grad = grad_P CE(logits, y)`` only.
  * Batch size 1, label known.

The prompt-only gradient is a low-dimensional summary of ``F(x)``
times softmax probabilities, so it carries far less information about
the input pixels than a full-model gradient -- the attack should fail
to reconstruct ``x`` even under the strongest reasonable settings.
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


class TinyPromptModel(nn.Module):
    """LeNet image encoder + learnable per-class prompt embeddings."""

    def __init__(self, num_classes: int = 10, embed_dim: int = 64):
        super().__init__()
        act = nn.Sigmoid
        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 12, kernel_size=5, padding=5 // 2, stride=2),  # 32 -> 16
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=2),  # 16 -> 8
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=1),
            act(),
            nn.Flatten(),
            nn.Linear(768, embed_dim),
        )
        # Per-class prompt embeddings; this is the *only* parameter the
        # server sees a gradient of.
        self.prompt = nn.Parameter(torch.randn(num_classes, embed_dim) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.image_encoder(x)
        feats = F.normalize(feats, dim=-1)
        prompt = F.normalize(self.prompt, dim=-1)
        return feats @ prompt.T


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


def total_variation(x: torch.Tensor) -> torch.Tensor:
    dh = (x[..., 1:, :] - x[..., :-1, :]).abs().mean()
    dw = (x[..., :, 1:] - x[..., :, :-1]).abs().mean()
    return dh + dw


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

    model = TinyPromptModel(num_classes=len(classnames),
                            embed_dim=args.embed_dim).to(device)
    model.apply(_dlg_init)
    model.eval()

    # Freeze everything except the prompt -- only the prompt's gradient
    # is "communicated" to the server in this threat model.
    for p in model.parameters():
        p.requires_grad_(False)
    model.prompt.requires_grad_(True)

    shared_params = [model.prompt]
    n_total = sum(p.numel() for p in model.parameters())
    n_shared = sum(p.numel() for p in shared_params)
    print(f"signal=prompt (TinyPromptModel) | "
          f"#total params = {n_total:,} | #shared params = {n_shared:,}")

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

    # Auto-pick a reasonable LR if the user did not override it. L-BFGS
    # wants ~1.0; Adam wants ~0.1 on this image scale.
    lr = args.lr
    if lr is None:
        lr = 1.0 if args.optimizer == "lbfgs" else 0.1

    if args.optimizer == "lbfgs":
        optimizer = torch.optim.LBFGS([x_dummy], lr=lr,
                                      max_iter=20, history_size=100)
    else:
        optimizer = torch.optim.Adam([x_dummy], lr=lr)

    def save_snapshot(it: int) -> None:
        with torch.no_grad():
            snap = denormalize_to_unit(x_dummy)
            vutils.save_image(snap, os.path.join(args.output, f"iter_{it:03d}.png"))

    if 0 in save_iters:
        save_snapshot(0)

    # 3. Attack loop. Default optimizer is Adam (not L-BFGS) for the
    #    CAPT row: the prompt-only gradient is a low-dimensional summary
    #    that L-BFGS can drive to zero in a handful of line-search steps,
    #    leaving every saved snapshot looking identical. Adam takes one
    #    small step per outer iter, so each iter is visibly distinct.
    #    A small TV regularizer keeps the optimizer moving even after
    #    grad-match L2 saturates, so the image keeps evolving but never
    #    actually reconstructs (the gradient bottleneck still blocks it).
    for it in range(1, args.attack_iters + 1):
        last_loss = {"g": None, "tv": None}

        def closure():
            optimizer.zero_grad(set_to_none=True)
            logits_hat = model(x_dummy)
            loss_hat = F.cross_entropy(logits_hat, y)
            dummy_grads = torch.autograd.grad(
                loss_hat, shared_params, create_graph=True)
            g_loss = grad_l2_loss(dummy_grads, target_grads)
            tv = total_variation(x_dummy)
            total = g_loss + args.tv_weight * tv
            total.backward()
            last_loss["g"] = g_loss.item()
            last_loss["tv"] = tv.item()
            return total

        optimizer.step(closure)

        if it in save_iters:
            save_snapshot(it)

        if it % args.log_every == 0:
            g = last_loss["g"]
            tv = last_loss["tv"]
            print(f"[capt-prompt] iter {it:4d} | grad-match L2 = "
                  f"{(g if g is not None else float('nan')):.6f} | "
                  f"tv = {(tv if tv is not None else float('nan')):.4f}")

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
    p.add_argument("--embed-dim", type=int, default=64,
                   help="embedding dim shared by image features and prompts")
    p.add_argument("--attack-iters", type=int, default=100)
    p.add_argument("--save-iters", default="0,20,40,60,80,100")
    p.add_argument("--optimizer", choices=["lbfgs", "adam"], default="adam",
                   help="adam: slow per-step, every iter visibly different "
                        "(recommended). lbfgs: converges in 1-2 line searches "
                        "and every saved iter looks the same.")
    p.add_argument("--lr", type=float, default=None,
                   help="auto-picked from --optimizer if omitted "
                        "(1.0 for L-BFGS, 0.1 for Adam)")
    p.add_argument("--tv-weight", type=float, default=1e-3,
                   help="total-variation regularizer on x_dummy; keeps the "
                        "optimizer moving once grad-match saturates")

    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--log-every", type=int, default=10)
    return p.parse_args()


if __name__ == "__main__":
    run_attack(parse_args())
