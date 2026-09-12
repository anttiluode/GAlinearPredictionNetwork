from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import os
from pathlib import Path
import platform
import random
import sys

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from galinear.reporting import write_receipts
from galinear.training import RunConfig, train_arm


ARMS = ("plain", "scalar_linear", "joint_operator", "joint_guarded")


class VanillaCNN(nn.Module):
    """PyTorch translation of the public WNN CIFAR-10 SimpleNet."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 2 * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )
        self.reset_parameters_keras_like()

    def reset_parameters_keras_like(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Matched CIFAR-10 test: plain vs scalar curve-fit vs joint operator nowcasting"
    )
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=50, help="maximum real gradient epochs")
    p.add_argument("--history", type=int, default=5)
    p.add_argument("--skip", type=int, default=5)
    p.add_argument("--max-rank", type=int, default=4)
    p.add_argument("--ridge", type=float, default=1e-6)
    p.add_argument("--guard-rel-tol", type=float, default=0.0)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--target-accuracy", type=float, default=0.59)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--output-dir", default="results")
    p.add_argument("--arms", default=",".join(ARMS), help="comma-separated arm names")
    p.add_argument("--no-download", action="store_true")
    p.add_argument("--nondeterministic", action="store_true")
    return p


def set_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def _mean_image_from_cifar(raw_data: np.ndarray) -> torch.Tensor:
    mean_hwc = raw_data.mean(axis=0, dtype=np.float64) / 255.0
    return torch.from_numpy(mean_hwc.astype(np.float32)).permute(2, 0, 1).contiguous()


class CifarTrainTransform:
    def __init__(self, mean_image: torch.Tensor) -> None:
        from torchvision.transforms import RandomAffine
        from torchvision.transforms import functional as TF
        from torchvision.transforms.functional import InterpolationMode

        self.mean_image = mean_image
        self._to_tensor = TF.to_tensor
        # Public WNN code: rotation=10, width/height shift=.15, zoom=.3,
        # fill_mode='nearest'. RandomAffine is the closest torchvision analogue.
        self.affine = RandomAffine(
            degrees=10,
            translate=(0.15, 0.15),
            scale=(0.7, 1.3),
            interpolation=InterpolationMode.NEAREST,
            fill=0.0,
        )

    def __call__(self, image):
        x = self._to_tensor(image) - self.mean_image
        return self.affine(x)


class CifarValTransform:
    def __init__(self, mean_image: torch.Tensor) -> None:
        from torchvision.transforms import functional as TF

        self.mean_image = mean_image
        self._to_tensor = TF.to_tensor

    def __call__(self, image):
        return self._to_tensor(image) - self.mean_image


def make_loaders(args, *, arm_seed: int):
    from torchvision.datasets import CIFAR10

    base = CIFAR10(root=args.data_dir, train=True, download=not args.no_download, transform=None)
    mean_image = _mean_image_from_cifar(base.data)
    train_ds = CIFAR10(
        root=args.data_dir,
        train=True,
        download=False,
        transform=CifarTrainTransform(mean_image),
    )
    val_ds = CIFAR10(
        root=args.data_dir,
        train=False,
        download=not args.no_download,
        transform=CifarValTransform(mean_image),
    )
    generator = torch.Generator().manual_seed(arm_seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=str(args.device).startswith("cuda"),
        generator=generator,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=max(args.batch_size, 1024),
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=str(args.device).startswith("cuda"),
    )
    return train_loader, val_loader


def make_callbacks(train_loader, val_loader, device: torch.device):
    criterion = nn.CrossEntropyLoss()

    def real_epoch(model: nn.Module, optimizer: torch.optim.Optimizer, epoch_index: int):
        del epoch_index
        model.train()
        loss_sum = torch.zeros((), device=device)
        count = 0
        backwards = 0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            batch_n = x.shape[0]
            loss_sum = loss_sum + loss.detach() * batch_n
            count += batch_n
            backwards += 1
        return float(loss_sum / max(count, 1)), backwards

    def validate(model: nn.Module):
        model.eval()
        loss_sum = torch.zeros((), device=device)
        correct = torch.zeros((), device=device, dtype=torch.long)
        count = 0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                logits = model(x)
                loss = criterion(logits, y)
                batch_n = x.shape[0]
                loss_sum = loss_sum + loss.detach() * batch_n
                correct = correct + (logits.argmax(dim=1) == y).sum()
                count += batch_n
        return float(loss_sum / max(count, 1)), float(correct.float() / max(count, 1))

    return real_epoch, validate


def run(args) -> tuple[dict, Path, Path]:
    arms = tuple(a.strip() for a in args.arms.split(",") if a.strip())
    unknown = sorted(set(arms) - set(ARMS))
    if unknown:
        raise ValueError(f"unknown arms: {unknown}")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")

    set_seed(args.seed, deterministic=not args.nondeterministic)
    base = VanillaCNN().to(device)
    initial_state = copy.deepcopy(base.state_dict())
    config = RunConfig(
        max_real_epochs=args.epochs,
        history=args.history,
        skip=args.skip,
        max_rank=args.max_rank,
        ridge=args.ridge,
        guard_rel_tol=args.guard_rel_tol,
        target_accuracy=args.target_accuracy,
    )

    receipts = {}
    for arm in arms:
        # Reset every random stream before each arm so kth real epoch sees the
        # same shuffle/augmentation stream across matched arms.
        set_seed(args.seed, deterministic=not args.nondeterministic)
        train_loader, val_loader = make_loaders(args, arm_seed=args.seed)
        model = VanillaCNN().to(device)
        model.load_state_dict(initial_state)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8
        )
        real_epoch, validate = make_callbacks(train_loader, val_loader, device)
        print(f"\n=== {arm} ===", flush=True)
        receipt = train_arm(
            arm=arm,
            model=model,
            optimizer=optimizer,
            real_epoch_fn=real_epoch,
            validation_fn=validate,
            config=config,
        )
        receipts[arm] = receipt
        print(
            f"status={receipt.status} target={receipt.reached_target} "
            f"real_epochs={receipt.trained_epochs} backprops={receipt.backward_passes} "
            f"jumps={receipt.accepted_jumps}/{receipt.rejected_jumps} "
            f"acc={receipt.final_accuracy:.4f} sec={receipt.total_seconds:.3f}",
            flush=True,
        )

    metadata = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "device": str(device),
        "cuda_device": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "reference": "Jang et al. ICML 2023 public CIFAR10 WNN recipe translated to PyTorch",
    }
    config_dict = vars(args).copy()
    run_name = f"cifar10_seed{args.seed}_h{args.history}_s{args.skip}"
    json_path, csv_path = write_receipts(
        output_dir=args.output_dir,
        run_name=run_name,
        config=config_dict,
        metadata=metadata,
        receipts=receipts,
    )
    return receipts, json_path, csv_path


def main() -> None:
    args = build_parser().parse_args()
    receipts, json_path, csv_path = run(args)
    print("\n=== summary ===")
    plain = receipts.get("plain")
    for arm, r in receipts.items():
        speed = (plain.total_seconds / r.total_seconds) if plain and r.total_seconds > 0 else float("nan")
        print(
            f"{arm:17s} sec={r.total_seconds:8.3f} speedup_vs_plain={speed:6.3f} "
            f"epochs={r.trained_epochs:3d} backprops={r.backward_passes:5d} "
            f"acc={r.final_accuracy:.4f} target={r.reached_target}"
        )
    print(f"JSON: {json_path}")
    print(f"CSV:  {csv_path}")


if __name__ == "__main__":
    main()
