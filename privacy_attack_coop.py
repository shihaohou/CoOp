"""Standalone gradient inversion attack on top of CoOp / PromptFL.

This file is intentionally self-contained: it does **not** import Dassl or
yacs and does **not** import anything from ``trainers/``. PromptLearner,
TextEncoder, and ``load_clip_to_cpu`` are inlined below so the attack can
run in an environment that only has torch / torchvision / Pillow / ftfy /
the local ``clip`` submodule.

Two threat models:

  --signal full     Attacker observes the gradient w.r.t. the full CLIP
                    visual encoder. Strong "Full-model FL" positive
                    control: under one-step FedAvg the model update is
                    proportional to this gradient.

  --signal prompt   Attacker observes the gradient w.r.t. the prompt
                    parameters only (prompt_learner.ctx). PromptFL /
                    CoOp setting.

Threat model assumed in the figure: white-box server (model + loss
known), label known, batch size 1.
"""

import argparse
import os
from types import SimpleNamespace
from typing import List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.utils as vutils
from PIL import Image

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()


# Gradient inversion requires double-backward through CLIP's MultiheadAttention
# (text transformer) and AttentionPool2d (visual). The flash and
# memory-efficient SDPA kernels in PyTorch 2.x do not implement the second
# derivative ("derivative for aten::_scaled_dot_product_efficient_attention_
# backward is not implemented"), so force the math backend, which does.
if torch.cuda.is_available():
    try:
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    except AttributeError:
        pass


CLIP_PIXEL_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_PIXEL_STD = (0.26862954, 0.26130258, 0.27577711)


# ---------------------------------------------------------------------------
# CoOp model components (inlined from trainers/coop.py to avoid the Dassl
# dependency for an attack-only run).
# ---------------------------------------------------------------------------


def load_clip_to_cpu(backbone_name: str):
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)
    try:
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None
    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    return clip.build_model(state_dict or model.state_dict())


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x


class PromptLearner(nn.Module):
    """Pared-down copy of ``trainers.coop.PromptLearner``."""

    def __init__(self, cfg, classnames: Sequence[str], clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.n_ctx
        ctx_init = cfg.ctx_init
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        assert cfg.image_size == clip_imsize, (
            f"cfg.image_size ({cfg.image_size}) must equal clip_imsize ({clip_imsize})"
        )

        if ctx_init:
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            if cfg.csc:
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        self.register_buffer("token_prefix", embedding[:, :1, :])
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts
        self.name_lens = name_lens
        self.class_token_position = cfg.class_token_position

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            return torch.cat([prefix, ctx, suffix], dim=1)

        if self.class_token_position == "middle":
            half = self.n_ctx // 2
            out = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1]
                class_i = suffix[i : i + 1, :name_len]
                suffix_i = suffix[i : i + 1, name_len:]
                out.append(torch.cat([
                    prefix_i,
                    ctx[i : i + 1, :half],
                    class_i,
                    ctx[i : i + 1, half:],
                    suffix_i,
                ], dim=1))
            return torch.cat(out, dim=0)

        if self.class_token_position == "front":
            out = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1]
                class_i = suffix[i : i + 1, :name_len]
                suffix_i = suffix[i : i + 1, name_len:]
                out.append(torch.cat([
                    prefix_i,
                    class_i,
                    ctx[i : i + 1],
                    suffix_i,
                ], dim=1))
            return torch.cat(out, dim=0)

        raise ValueError(self.class_token_position)


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames: Sequence[str], clip_model):
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

    cfg = SimpleNamespace(
        n_ctx=args.n_ctx,
        csc=args.csc,
        ctx_init=args.ctx_init,
        class_token_position=args.class_token_position,
        image_size=args.image_size,
    )
    clip_model = load_clip_to_cpu(args.backbone)
    clip_model.float()  # gradient inversion is unstable in fp16
    model = CustomCLIP(cfg, classnames, clip_model).to(device)
    model.eval()

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

    x = load_image(args.image_path, args.image_size, device)
    y = torch.tensor([args.label], device=device, dtype=torch.long)
    vutils.save_image(denormalize_to_unit(x), os.path.join(args.output, "original.png"))

    logits = model(x)
    target_loss = F.cross_entropy(logits, y)
    target_grads = torch.autograd.grad(target_loss, shared_params,
                                       retain_graph=False, create_graph=False)
    target_grads = [g.detach() for g in target_grads]
    print(f"target CE loss = {target_loss.item():.4f}")

    image_shape = (1, 3, args.image_size, args.image_size)
    x_dummy = (torch.rand(image_shape, device=device) * 2.0 - 1.0).clone()
    x_dummy.requires_grad_(True)
    optimizer = torch.optim.Adam([x_dummy], lr=args.lr)

    grad_loss_fn = grad_cosine_loss if args.match == "cosine" else grad_l2_loss

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
    p.add_argument("--label", type=int, required=True)
    p.add_argument("--classnames-file", required=True)
    p.add_argument("--signal", choices=["full", "prompt"], required=True)
    p.add_argument("--output", required=True)

    p.add_argument("--backbone", default="RN50",
                   choices=list(clip._MODELS.keys()))
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--n-ctx", type=int, default=16)
    p.add_argument("--csc", action="store_true")
    p.add_argument("--ctx-init", default="")
    p.add_argument("--class-token-position", default="end",
                   choices=["end", "middle", "front"])

    p.add_argument("--attack-iters", type=int, default=100)
    p.add_argument("--save-iters", default="0,20,40,60,80,100")
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--tv-weight", type=float, default=1e-4)
    p.add_argument("--match", choices=["cosine", "l2"], default="cosine")

    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--log-every", type=int, default=10)
    return p.parse_args()


if __name__ == "__main__":
    run_attack(parse_args())
