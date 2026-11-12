import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from mmfi_dataset import MMFiWindowDataset
from mmfi_finetune import (
    NUM_CLASSES,
    make_components,
    seed_all,
    to_network_inputs,
)


TARGET_ENVIRONMENTS = ("E02", "E03", "E04")
ARCHITECTURE_DEFAULTS = {
    "token_dim": 512,
    "projection_dim": 512,
    "z_hidden_dim": 512,
    "num_tokens": 32,
    "sharing_num": 4,
    "dropout": 0.1,
    "generator_depth": 2,
    "generator_heads": 8,
    "generator_dropout": 0.1,
    "cls_gamma_init": 0.1,
    "add_modality_embed": False,
}


def load_source_checkpoint(path, device):
    checkpoint = torch.load(Path(path).expanduser(), map_location=device)
    required = (
        "wifi_encoder",
        "mmwave_encoder",
        "projector",
        "token_projector",
        "token_encoder",
        "classifier",
    )
    missing = [key for key in required if key not in checkpoint]
    if missing:
        raise KeyError(
            f"source checkpoint is missing {missing}; pass a checkpoint saved by "
            "mmfi_finetune.py"
        )
    return checkpoint


def resolve_architecture_args(args, checkpoint):
    checkpoint_args = checkpoint.get("args", {})
    for name, default in ARCHITECTURE_DEFAULTS.items():
        if getattr(args, name) is None:
            value = checkpoint_args.get(name, checkpoint.get(name, default))
            setattr(args, name, value)


def make_loaded_components(args, checkpoint, device):
    modules = make_components(args, device)
    keys = (
        "wifi_encoder",
        "mmwave_encoder",
        "projector",
        "token_projector",
        "token_encoder",
        "classifier",
    )
    for module, key in zip(modules, keys):
        module.load_state_dict(checkpoint[key], strict=True)
    return modules


def make_target_split(args, environment, split_seed):
    dataset = MMFiWindowDataset(
        args.data_root,
        args.segments_csv,
        environments=[environment],
        split=None,
        window_size=args.window_size,
        max_points=args.max_points,
        seed=split_seed,
    )

    groups = {}
    for index, sample in enumerate(dataset.samples):
        key = (sample["student"], sample["action"])
        groups.setdefault(key, []).append(index)

    rng = random.Random(split_seed)
    support_indices = []
    query_indices = []
    for key in sorted(groups):
        indices = list(groups[key])
        rng.shuffle(indices)
        if len(indices) <= args.shots:
            raise RuntimeError(
                f"{environment} {key} has {len(indices)} segments; "
                f"at least {args.shots + 1} are required for support/query splitting"
            )
        support_indices.extend(indices[: args.shots])
        query_indices.extend(indices[args.shots :])

    if not support_indices or not query_indices:
        raise RuntimeError(f"empty support/query split for {environment}")
    return dataset, Subset(dataset, support_indices), Subset(dataset, query_indices), len(groups)


def forward_logits(modules, wifi, mmwave):
    wifi_encoder, mmwave_encoder, projector, token_projector, token_encoder, classifier = modules
    wifi_input, mmwave_input = to_network_inputs(wifi, mmwave)
    wifi_map = wifi_encoder.forward_map(wifi_input)
    mmwave_map = mmwave_encoder.forward_map(mmwave_input)
    wifi_z, mmwave_z = projector(
        wifi_encoder.forward_global(wifi_map),
        mmwave_encoder.forward_global(mmwave_map),
    )
    tokens = token_encoder(token_projector(wifi_map, mmwave_map))
    return classifier((wifi_z, mmwave_z), tokens)


def evaluate(loader, device, modules):
    for module in modules:
        module.eval()
    confusion = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long)
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total = 0
    with torch.no_grad():
        for wifi, mmwave, labels, _ in loader:
            wifi = wifi.to(device, non_blocking=True)
            mmwave = mmwave.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = forward_logits(modules, wifi, mmwave)
            total_loss += criterion(logits, labels).item() * labels.numel()
            predictions = logits.argmax(1)
            total += labels.numel()
            for target, prediction in zip(labels.cpu(), predictions.cpu()):
                confusion[int(target), int(prediction)] += 1

    accuracy = 100.0 * confusion.diag().sum().item() / max(total, 1)
    support = confusion.sum(1).float()
    predicted = confusion.sum(0).float()
    precision = confusion.diag().float() / predicted.clamp_min(1.0)
    recall = confusion.diag().float() / support.clamp_min(1.0)
    valid = support > 0
    f1 = (2.0 * precision * recall / (precision + recall).clamp_min(1e-8))[valid].mean().item()
    return total_loss / max(total, 1), accuracy, 100.0 * f1


def build_optimizer(args, modules):
    wifi_encoder, mmwave_encoder, projector, token_projector, token_encoder, classifier = modules
    pretrained_components = (
        list(wifi_encoder.parameters())
        + list(mmwave_encoder.parameters())
        + list(projector.parameters())
        + list(token_projector.parameters())
        + list(token_encoder.parameters())
    )
    return torch.optim.AdamW(
        [
            {"params": pretrained_components, "lr": args.lr},
            {"params": classifier.parameters(), "lr": args.head_lr},
        ],
        weight_decay=args.weight_decay,
    )


def build_scheduler(args, optimizer):
    if args.scheduler == "constant":
        return None
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.adapt_epochs,
        eta_min=args.min_lr,
    )


def adapt_one_environment(args, checkpoint, environment, device, env_index):
    split_seed = args.seed + env_index
    _, support_set, query_set, group_count = make_target_split(args, environment, split_seed)
    support_loader = DataLoader(
        support_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        generator=torch.Generator().manual_seed(split_seed),
    )
    query_loader = DataLoader(
        query_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )

    modules = make_loaded_components(args, checkpoint, device)
    optimizer = build_optimizer(args, modules)
    scheduler = build_scheduler(args, optimizer)
    before_loss, before_acc, before_f1 = evaluate(query_loader, device, modules)

    print(
        f"[{environment}] groups={group_count} support={len(support_set)} "
        f"query={len(query_set)} before_acc={before_acc:.2f} before_f1={before_f1:.2f}",
        flush=True,
    )

    criterion = nn.CrossEntropyLoss()
    for epoch in range(1, args.adapt_epochs + 1):
        for module in modules:
            module.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for wifi, mmwave, labels, _ in support_loader:
            wifi = wifi.to(device, non_blocking=True)
            mmwave = mmwave.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = forward_logits(modules, wifi, mmwave)
            loss = criterion(logits, labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [parameter for group in optimizer.param_groups for parameter in group["params"]],
                1.0,
            )
            optimizer.step()
            running_loss += loss.item() * labels.numel()
            correct += logits.argmax(1).eq(labels).sum().item()
            total += labels.numel()

        if scheduler is not None:
            scheduler.step()
        if args.eval_interval > 0 and (
            epoch % args.eval_interval == 0 or epoch == args.adapt_epochs
        ):
            query_loss, query_acc, query_f1 = evaluate(query_loader, device, modules)
            print(
                f"[{environment}] epoch {epoch:03d}/{args.adapt_epochs} "
                f"support_loss={running_loss / max(total, 1):.4f} "
                f"support_acc={100.0 * correct / max(total, 1):.2f} "
                f"query_acc={query_acc:.2f} query_f1={query_f1:.2f} "
                f"lr_pretrained={optimizer.param_groups[0]['lr']:.6g} "
                f"lr_head={optimizer.param_groups[1]['lr']:.6g}",
                flush=True,
            )

    final_loss, final_acc, final_f1 = evaluate(query_loader, device, modules)
    print(
        f"[{environment}] final_loss={final_loss:.4f} "
        f"final_acc={final_acc:.2f} final_f1={final_f1:.2f}",
        flush=True,
    )

    if args.save_dir:
        save_dir = Path(args.save_dir).expanduser()
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / f"{environment}_{args.shots}shot_seed{args.seed}.pth"
        torch.save(
            {
                "adapt_type": "mmfi_cross_domain_full",
                "source_checkpoint": str(Path(args.source_ckpt).expanduser()),
                "environment": environment,
                "shots": args.shots,
                "seed": args.seed,
                "support_samples": len(support_set),
                "query_samples": len(query_set),
                "args": vars(args),
                "wifi_encoder": modules[0].state_dict(),
                "mmwave_encoder": modules[1].state_dict(),
                "projector": modules[2].state_dict(),
                "token_projector": modules[3].state_dict(),
                "token_encoder": modules[4].state_dict(),
                "classifier": modules[5].state_dict(),
            },
            save_path,
        )
        print(f"[{environment}] saved: {save_path}", flush=True)

    del modules, optimizer, scheduler, support_loader, query_loader
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return final_acc, final_f1


def main():
    parser = argparse.ArgumentParser(
        description="MM-Fi E01-to-target full few-shot adaptation for sepdual"
    )
    parser.add_argument("--data_root", default="~/MMFi/MMFi_Dataset")
    parser.add_argument("--segments_csv", default="~/MMFi/MMFi_action_segments.csv")
    parser.add_argument(
        "--source_ckpt",
        required=True,
        help="E01 source-domain fine-tuning checkpoint from mmfi_finetune.py",
    )
    parser.add_argument(
        "--save_dir",
        default="",
        help="Optional directory for saving one adapted checkpoint per target environment",
    )
    parser.add_argument("--target_envs", nargs="+", default=list(TARGET_ENVIRONMENTS))
    parser.add_argument("--shots", type=int, choices=(1, 2), required=True)
    parser.add_argument("--adapt_epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1.5e-4)
    parser.add_argument("--head_lr", type=float, default=1.5e-4)
    parser.add_argument("--min_lr", type=float, default=0.0)
    parser.add_argument("--scheduler", choices=("cosine", "constant"), default="cosine")
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--window_size", type=int, default=32)
    parser.add_argument("--max_points", type=int, default=128)
    parser.add_argument("--eval_interval", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")

    for name, default in ARCHITECTURE_DEFAULTS.items():
        if isinstance(default, bool):
            parser.add_argument(f"--{name}", action="store_true", default=None)
        elif isinstance(default, int):
            parser.add_argument(f"--{name}", type=int, default=None)
        else:
            parser.add_argument(f"--{name}", type=float, default=None)

    args = parser.parse_args()
    if args.adapt_epochs < 1:
        parser.error("--adapt_epochs must be positive")
    if args.lr <= 0 or args.head_lr <= 0:
        parser.error("--lr and --head_lr must be positive")
    invalid_envs = [env for env in args.target_envs if env not in TARGET_ENVIRONMENTS]
    if invalid_envs:
        parser.error(
            f"target_envs must be drawn from {TARGET_ENVIRONMENTS}; got {invalid_envs}"
        )

    seed_all(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    checkpoint = load_source_checkpoint(args.source_ckpt, device)
    resolve_architecture_args(args, checkpoint)

    print(f"device: {device}", flush=True)
    print(f"source checkpoint: {Path(args.source_ckpt).expanduser()}", flush=True)
    print(f"shots: {args.shots} scheduler: {args.scheduler}", flush=True)
    print(f"target environments: {', '.join(args.target_envs)}", flush=True)

    results = []
    for env_index, environment in enumerate(args.target_envs):
        if environment == "E01":
            raise ValueError("E01 is the source domain; target_envs must use E02/E03/E04")
        acc, f1 = adapt_one_environment(args, checkpoint, environment, device, env_index)
        results.append((environment, acc, f1))

    mean_acc = sum(acc for _, acc, _ in results) / len(results)
    mean_f1 = sum(f1 for _, _, f1 in results) / len(results)
    print("results:", flush=True)
    for environment, acc, f1 in results:
        print(f"  {environment}: accuracy={acc:.2f} f1={f1:.2f}", flush=True)
    print(f"  average: accuracy={mean_acc:.2f} f1={mean_f1:.2f}", flush=True)


if __name__ == "__main__":
    main()


