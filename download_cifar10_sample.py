"""Download CIFAR-10 once and save one sample as a 32x32 PNG.

The privacy attack script will upscale this image to 224x224 via bicubic
during CLIP normalization, so the saved file does not need to be large.

Example:
    python download_cifar10_sample.py \\
        --class-idx 8 --sample-idx 0 \\
        --output data/cifar10_ship.png
"""

import argparse
import os

import torchvision


CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                description=__doc__)
    p.add_argument("--root", default="data/cifar10",
                   help="where to download CIFAR-10 (~170MB)")
    p.add_argument("--split", default="test", choices=["train", "test"])
    p.add_argument("--class-idx", type=int, default=8,
                   help="0..9; 0=airplane 1=automobile 2=bird 3=cat 4=deer "
                        "5=dog 6=frog 7=horse 8=ship 9=truck")
    p.add_argument("--sample-idx", type=int, default=0,
                   help="which sample within the chosen class (0-indexed)")
    p.add_argument("--output", default="data/cifar10_sample.png",
                   help="path to save the chosen sample as a 32x32 PNG")
    return p.parse_args()


def main():
    args = parse_args()
    if not (0 <= args.class_idx < 10):
        raise SystemExit("--class-idx must be in [0, 9]")

    os.makedirs(args.root, exist_ok=True)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    ds = torchvision.datasets.CIFAR10(
        root=args.root, train=(args.split == "train"), download=True,
    )

    # ds.targets is a list of ints; iterate over it directly to avoid
    # materializing every PIL Image.
    matches = [i for i, y in enumerate(ds.targets) if y == args.class_idx]
    if args.sample_idx >= len(matches):
        raise SystemExit(
            f"--sample-idx {args.sample_idx} out of range "
            f"(only {len(matches)} samples in class "
            f"{args.class_idx} of split={args.split})"
        )

    idx = matches[args.sample_idx]
    img, label = ds[idx]
    img.save(args.output)

    print(f"class index = {label}  ({CIFAR10_CLASSES[label]})")
    print(f"saved to {args.output}")
    print()
    print("Run the attack with:")
    print(f"  bash scripts/coop/privacy_attack.sh {args.output} {label} "
          "configs/privacy/classnames_cifar10.txt output/privacy_gia")


if __name__ == "__main__":
    main()
