"""Standalone gradient inversion attack on top of CoOp / PromptFL.

Two threat models, both share the same CLIP backbone built by CoOp:

  --signal full     Attacker observes the gradient w.r.t. the full CLIP
                    visual encoder. This is the strong "Full-model FL"
                    positive control: under one-step FedAvg the model
                    update is proportional to this gradient, so it is a
                    worst-case exposure for image reconstruction.

  --signal prompt   Attacker observes the gradient w.r.t. the prompt
                    parameters only (prompt_learner.ctx). This is the
                    PromptFL / CoOp setting.

The script:
  1. Builds CoOp's CustomCLIP (frozen image + text encoders, learnable
     prompt context) in fp32.
  2. Loads one real image, computes the cross-entropy loss with the
     known label, takes the gradient w.r.t. shared_params -> target_grads.
  3. Initializes a dummy image with random pixels and Adam-optimizes it
     to minimize 1 - cos(dummy_grads, target_grads) + lambda * TV(image).
  4. Snapshots the dummy image at the requested attack iterations.

Threat model assumed in the figure / paper:
  - white-box server (model + loss known)
  - label known (worst case for the defender)
  - batch size 1

Designed to be edited on a Mac and run on an H800 GPU server.
"""

import argparse
import os
from typing import List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.utils as vutils
from PIL import Image
from yacs.config import CfgNode as CN

from clip import clip
from trainers.coop import PromptLearner, TextEncoder, load_clip_to_cpu


CLIP_PIXEL_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_PIXEL_STD = (0.26862954, 0.26130258, 0.27577711)


# ---------------------------------------------------------------------------
# Model / config
# ---------------------------------------------------------------------------


def build_cfg(backbone: str, image_size: int, n_ctx: int, csc: bool,
              ctx_init: str, class_token_position: str) -> CN:
    cfg = CN(new_allowed=True)

    cfg.MODEL = CN(new_allowed=True)
    cfg.MODEL.BACKBONE = CN(new_allowed=True)
    cfg.MODEL.BACKBONE.NAME = backbone

    cfg.INPUT = CN(new_allowed=True)
    cfg.INPUT.SIZE = (image_size, image_size)

    cfg.TRAINER = CN(new_allowed=True)
    cfg.TRAINER.COOP = CN(new_allowed=True)
    cfg.TRAINER.COOP.N_CTX = n_ctx
    cfg.TRAINER.COOP.CSC = csc
    cfg.TRAINER.COOP.CTX_INIT = ctx_init
    cfg.TRAINER.COOP.PREC = "fp32"
    cfg.TRAINER.COOP.CLASS_TOKEN_POSITION = class_token_position
    return cfg


class CustomCLIP(nn.Module):
    """Mirror of trainers.coop.CustomCLIP, kept here so we do not need to
    instantiate Dassl's trainer for an attack-only run."""

    def __init__(self, cfg: CN, classnames: Sequence[str], clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        image_features = self.image_encoder(image.type(self.dtype))
        prompts = self.prompt_learner()
        text_features = self.text_encoder(prompts, self.tokenized_prompts)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        logit_scale = self.logit_scale.exp()
        return logit_scale * image_features @ text_features.t()


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------


def _norm_consts(device: torch.device):
    mean = torch.tensor(CLIP_PIXEL_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(CLIP_PIXEL_STD, device=device).view(1, 3, 1, 1)
    return mean, std


def load_image(path: str, image_size: int, device: torch.device) -> torch.Tensor:
    """Return a 1x3xHxW CLIP-normalized tensor."""
    img = Image.open(path).convert("RGB")
    tfm = T.Compose([
        T.Resize(image_size, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(image_size),
        T.ToTensor(),
        T.Normalize(CLIP_PIXEL_MEAN, CLIP_PIXEL_STD),
    ])
    return tfm(img).unsqueeze(0).to(device)


def denormalize_to_unit(x: torch.Tensor) -> torch.Tensor:
    mean, std = _norm_consts(x.device)
    return (x * std + mean).clamp(0.0, 1.0)


def total_variation(x: torch.Tensor) -> torch.Tensor:
    dh = (x[..., 1:, :] - x[..., :-1, :]).abs().mean()
    dw = (x[..., :, 1:] - x[..., :, :-1]).abs().mean()
    return dh + dw


# ---------------------------------------------------------------------------
# Attack
# ---------------------------------------------------------------------------


def grad_cosine_loss(dummy_grads, target_grads) -> torch.Tensor:
    """Layerwise cosine distance, averaged. Matches Geiping et al."""
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


def grad_l2_loss(dummy_grads, target_grads) -> torch.Tensor:
    loss = torch.tensor(0.0, device=target_grads[0].device)
    n = 0
    for gd, gt in zip(dummy_grads, target_grads):
        if gd is None or gt is None:
            continue
        loss = loss + F.mse_loss(gd, gt.detach())
        n += 1
    return loss / max(n, 1)


def collect_shared_params(model: CustomCLIP, signal: str) -> List[nn.Parameter]:
    if signal == "prompt":
        return [model.prompt_learner.ctx]
    if signal == "full":
        return [p for p in model.image_encoder.parameters() if p.requires_grad]
    raise ValueError(f"unknown signal {signal!r}")


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
    print(f"#classes = {len(classnames)} | target class = {classnames[args.label]!r}")

    # 1. Build CoOp-style model in fp32 (gradient inversion is unstable in fp16).
    cfg = build_cfg(args.backbone, args.image_size, args.n_ctx, args.csc,
                    args.ctx_init, args.class_token_position)
    clip_model = load_clip_to_cpu(cfg)
    clip_model.float()
    model = CustomCLIP(cfg, classnames, clip_model).to(device)
    model.eval()  # freeze BN running stats; we are not training the model

    # Freeze everything, then unfreeze the signal-specific parameters.
    for p in model.parameters():
        p.requires_grad_(False)
    if args.signal == "prompt":
        model.prompt_learner.ctx.requires_grad_(True)
    elif args.signal == "full":
        for p in model.image_encoder.parameters():
            p.requires_grad_(True)

    shared_params = collect_shared_params(model, args.signal)
    n_shared = sum(p.numel() for p in shared_params)
    print(f"signal={args.signal} | #shared params = {n_shared:,}")

    # 2. Compute the target (server-observed) gradient on the real image.
    x = load_image(args.image_path, args.image_size, device)
    y = torch.tensor([args.label], device=device, dtype=torch.long)
    vutils.save_image(denormalize_to_unit(x), os.path.join(args.output, "original.png"))

    logits = model(x)
    target_loss = F.cross_entropy(logits, y)
    target_grads = torch.autograd.grad(target_loss, shared_params,
                                       retain_graph=False, create_graph=False)
    target_grads = [g.detach() for g in target_grads]
    print(f"target CE loss = {target_loss.item():.4f}")

    # 3. Initialize the dummy image in CLIP-normalized space.
    image_shape = (1, 3, args.image_size, args.image_size)
    x_dummy = (torch.rand(image_shape, device=device) * 2.0 - 1.0).clone()
    x_dummy.requires_grad_(True)
    optimizer = torch.optim.Adam([x_dummy], lr=args.lr)

    grad_loss_fn = grad_cosine_loss if args.match == "cosine" else grad_l2_loss

    # 4. Inversion loop.
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
            create_graph=True, retain_graph=True, allow_unused=True,
        )

        g_loss = grad_loss_fn(dummy_grads, target_grads)
        # TV regularization is computed in normalized-pixel space; this is
        # the same as Geiping et al. up to a constant scaling.
        tv = total_variation(x_dummy) * args.tv_weight
        attack_loss = g_loss + tv

        attack_loss.backward()
        optimizer.step()

        if it % args.log_every == 0:
            print(f"[{args.signal}] iter {it:4d} | "
                  f"g_loss {g_loss.item():.4f} | tv {tv.item():.4f}")

    print(f"Saved snapshots and original.png under {args.output}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--image-path", required=True)
    p.add_argument("--label", type=int, required=True,
                   help="ground-truth class index into --classnames-file")
    p.add_argument("--classnames-file", required=True,
                   help="one classname per line; line k = class index k")
    p.add_argument("--signal", choices=["full", "prompt"], required=True)
    p.add_argument("--output", required=True)

    p.add_argument("--backbone", default="RN50",
                   choices=list(clip._MODELS.keys()))
    p.add_argument("--image-size", type=int, default=224,
                   help="must match the CLIP backbone input resolution")
    p.add_argument("--n-ctx", type=int, default=16)
    p.add_argument("--csc", action="store_true",
                   help="use class-specific context (matches Lambda_c^k proxy setup)")
    p.add_argument("--ctx-init", default="")
    p.add_argument("--class-token-position", default="end",
                   choices=["end", "middle", "front"])

    p.add_argument("--attack-iters", type=int, default=100)
    p.add_argument("--save-iters", default="0,20,40,60,80,100")
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--tv-weight", type=float, default=1e-4)
    p.add_argument("--match", choices=["cosine", "l2"], default="cosine",
                   help="gradient-matching loss; cosine = Geiping et al.")

    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--log-every", type=int, default=10)
    return p.parse_args()


if __name__ == "__main__":
    run_attack(parse_args())
