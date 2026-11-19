import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from model.resnet2d import resnet18_mutual

from mmfi_dataset import MMFiWindowDataset


NUM_CLASSES = 27
NUM_MODALITIES = 2
BACKBONE_FEATURE_DIM = 1024


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ResNetBranch(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        net = resnet18_mutual(num_classes=NUM_CLASSES, include_top=False)
        net.conv1 = nn.Conv2d(in_channels, 128, kernel_size=7, stride=2, padding=3, bias=False)
        nn.init.kaiming_normal_(net.conv1.weight, mode="fan_out", nonlinearity="relu")
        self.net = net

    def forward_map(self, x):
        net = self.net
        if x.dim() == 5 and x.size(1) == 1:
            x = x.squeeze(1)
        x = net.conv1(x)
        x = net.bn1(x)
        x = net.relu(x)
        x = net.maxpool(x)
        x = net.layer1(x)
        x = net.layer2(x)
        x = net.layer3(x)
        return net.layer4(x)

    def forward_global(self, feature_map):
        return F.adaptive_avg_pool2d(feature_map, 1).flatten(1)


class ModalityAdapter(nn.Module):
    def __init__(self, feature_dim=BACKBONE_FEATURE_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, feature_dim),
        )

    def forward(self, x):
        return self.net(x)


class GlobalProjector(nn.Module):
    def __init__(self, feature_dim=BACKBONE_FEATURE_DIM, projection_dim=256):
        super().__init__()
        self.wifi_adapter = ModalityAdapter(feature_dim)
        self.mmwave_adapter = ModalityAdapter(feature_dim)
        self.shared_projector = nn.Sequential(
            nn.Linear(feature_dim, projection_dim),
            nn.GELU(),
            nn.Linear(projection_dim, projection_dim),
        )

    def forward(self, wifi_feature, mmwave_feature):
        return (
            self.shared_projector(self.wifi_adapter(wifi_feature)),
            self.shared_projector(self.mmwave_adapter(mmwave_feature)),
        )


class TokenProjector(nn.Module):
    def __init__(self, feature_dim=BACKBONE_FEATURE_DIM, token_dim=256, num_tokens=32):
        super().__init__()
        self.num_tokens = int(num_tokens)
        self.wifi_proj = nn.Sequential(
            nn.Conv1d(feature_dim, token_dim, kernel_size=1),
            nn.BatchNorm1d(token_dim),
            nn.ReLU(),
        )
        self.mmwave_proj = nn.Sequential(
            nn.Conv1d(feature_dim, token_dim, kernel_size=1),
            nn.BatchNorm1d(token_dim),
            nn.ReLU(),
        )

    def _to_tokens(self, x, projector):
        x = projector(x.flatten(2))
        if x.size(-1) != self.num_tokens:
            x = F.interpolate(x, size=self.num_tokens, mode="linear", align_corners=False)
        return x.transpose(1, 2)

    def forward(self, wifi_map, mmwave_map):
        return {
            "wifi": self._to_tokens(wifi_map, self.wifi_proj),
            "mmwave": self._to_tokens(mmwave_map, self.mmwave_proj),
        }


class TokenContextEncoder(nn.Module):
    """Reuse the pretraining generator's source encoder for downstream tokens."""

    def __init__(self, token_dim=256, num_tokens=32, depth=2, heads=8, dropout=0.1):
        super().__init__()
        self.type_embed = nn.Parameter(torch.zeros(NUM_MODALITIES, 1, token_dim))
        self.pos_embed = nn.Parameter(torch.zeros(NUM_MODALITIES, num_tokens, token_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=heads,
            dim_feedforward=4 * token_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.source_encoder = nn.TransformerEncoder(layer, num_layers=depth)

    def forward(self, tokens):
        output = {}
        for index, name in enumerate(("wifi", "mmwave")):
            x = tokens[name]
            x = x + self.type_embed[index].unsqueeze(0)
            x = x + self.pos_embed[index, : x.size(1)].unsqueeze(0)
            output[name] = self.source_encoder(x)
        return output


def build_dual_masks(len_wifi, len_mmwave, device):
    total = 1 + len_wifi + len_mmwave
    wifi_begin, wifi_end = 1, 1 + len_wifi
    mm_begin, mm_end = wifi_end, total

    inter_mask = torch.zeros(total, total, device=device)
    inter_mask[0, 0] = float("-inf")
    inter_mask[wifi_begin:wifi_end, wifi_begin:wifi_end] = float("-inf")
    inter_mask[mm_begin:mm_end, mm_begin:mm_end] = float("-inf")

    intra_mask = torch.full((total, total), float("-inf"), device=device)
    intra_mask[0, :] = 0
    intra_mask[:, 0] = 0
    intra_mask[0, 0] = float("-inf")
    intra_mask[wifi_begin:wifi_end, wifi_begin:wifi_end] = 0
    intra_mask[mm_begin:mm_end, mm_begin:mm_end] = 0
    return inter_mask, intra_mask


class Gate(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.inter_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.intra_proj = nn.Linear(embed_dim, embed_dim, bias=True)

    def forward(self, intra_feature, inter_feature):
        gate = torch.sigmoid(self.inter_proj(inter_feature) + self.intra_proj(intra_feature))
        return gate * inter_feature + (1.0 - gate) * intra_feature, gate


class DualIntraInterClassifier(nn.Module):
    """Two-modal counterpart of LargeCleanGenEncoderZGuidedIIN."""

    def __init__(
        self,
        token_dim=256,
        projection_dim=256,
        z_hidden_dim=256,
        num_tokens=32,
        sharing_num=4,
        dropout=0.1,
        cls_gamma_init=0.1,
        add_modality_embed=False,
    ):
        super().__init__()
        self.num_tokens = int(num_tokens)
        self.sharing_num = int(sharing_num)
        self.z_to_cls = nn.Sequential(
            nn.LayerNorm(2 * projection_dim),
            nn.Linear(2 * projection_dim, z_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(z_hidden_dim, token_dim),
            nn.LayerNorm(token_dim),
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, token_dim))
        self.cls_gamma = nn.Parameter(torch.tensor(float(cls_gamma_init)))
        self.cls_norm = nn.LayerNorm(token_dim)
        self.mte = nn.Embedding(NUM_MODALITIES, token_dim) if add_modality_embed else None
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + 2 * num_tokens, token_dim))

        self.inter_unit = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=8,
            dim_feedforward=4 * token_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.intra_unit = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=8,
            dim_feedforward=4 * token_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.gate = Gate(token_dim)
        self.mlp_head = nn.Sequential(
            nn.Linear(token_dim, token_dim // 2, bias=False),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(token_dim // 2, NUM_CLASSES),
        )

    def forward(self, model_zs, encoded_tokens, return_gate=False):
        wifi_tokens = encoded_tokens["wifi"]
        mmwave_tokens = encoded_tokens["mmwave"]
        if self.mte is not None:
            wifi_tokens = wifi_tokens + self.mte.weight[0].view(1, 1, -1)
            mmwave_tokens = mmwave_tokens + self.mte.weight[1].view(1, 1, -1)

        z = torch.cat(model_zs, dim=1)
        cls_guidance = self.z_to_cls(z).unsqueeze(1)
        cls = self.cls_token.expand(z.size(0), -1, -1)
        cls = self.cls_norm(cls + self.cls_gamma * cls_guidance)
        x = torch.cat([cls, wifi_tokens, mmwave_tokens], dim=1)
        x = x + self.pos_embed[:, : x.size(1)]
        inter_mask, intra_mask = build_dual_masks(
            wifi_tokens.size(1), mmwave_tokens.size(1), wifi_tokens.device
        )

        x_inter = x
        x_intra = x
        for _ in range(self.sharing_num):
            x_inter = self.inter_unit(x_inter, src_mask=inter_mask)
            x_intra = self.intra_unit(x_intra, src_mask=intra_mask)

        cls_inter = x_inter[:, 0]
        cls_intra = x_intra[:, 0]
        fused, gate = self.gate(cls_intra, cls_inter)
        logits = self.mlp_head(fused)
        return (logits, gate) if return_gate else logits


def to_network_inputs(wifi, mmwave):
    wifi_input = wifi.permute(0, 2, 1, 3, 4).reshape(
        wifi.size(0), wifi.size(2), wifi.size(1) * wifi.size(3), wifi.size(4)
    )
    mmwave_input = mmwave.permute(0, 3, 1, 2)
    return wifi_input, mmwave_input


def make_components(args, device):
    wifi_encoder = ResNetBranch(3).to(device)
    mmwave_encoder = ResNetBranch(5).to(device)
    projector = GlobalProjector(BACKBONE_FEATURE_DIM, args.projection_dim).to(device)
    token_projector = TokenProjector(BACKBONE_FEATURE_DIM, args.token_dim, args.num_tokens).to(device)
    token_encoder = TokenContextEncoder(
        args.token_dim,
        args.num_tokens,
        depth=args.generator_depth,
        heads=args.generator_heads,
        dropout=args.generator_dropout,
    ).to(device)
    classifier = DualIntraInterClassifier(
        token_dim=args.token_dim,
        projection_dim=args.projection_dim,
        z_hidden_dim=args.z_hidden_dim,
        num_tokens=args.num_tokens,
        sharing_num=args.sharing_num,
        dropout=args.dropout,
        cls_gamma_init=args.cls_gamma_init,
        add_modality_embed=args.add_modality_embed,
    ).to(device)
    return wifi_encoder, mmwave_encoder, projector, token_projector, token_encoder, classifier


def load_pretrain(path, device, modules):
    checkpoint = torch.load(Path(path).expanduser(), map_location=device)
    required = ("wifi_encoder", "mmwave_encoder", "projector", "token_projector", "cross_modal_generator")
    missing = [key for key in required if key not in checkpoint]
    if missing:
        raise KeyError(
            f"checkpoint is not a full MM-Fi sepdual checkpoint; missing {missing}. "
            "Re-run pretrain_mmfi_sepdual_full.py."
        )
    for module, key in zip(modules[:4], required[:4]):
        module.load_state_dict(checkpoint[key])
    generator_state = checkpoint["cross_modal_generator"]
    source_encoder_state = {
        key: value
        for key, value in generator_state.items()
        if key == "type_embed" or key == "pos_embed" or key.startswith("source_encoder.")
    }
    modules[4].load_state_dict(source_encoder_state, strict=True)
    return checkpoint


def build_optimizer(args, modules):
    wifi_encoder, mmwave_encoder, projector, token_projector, token_encoder, classifier = modules
    pretrained = list(wifi_encoder.parameters()) + list(mmwave_encoder.parameters())
    pretrained += list(projector.parameters()) + list(token_projector.parameters())
    pretrained += list(token_encoder.parameters())
    groups = [
        {"params": pretrained, "lr": args.lr},
        {"params": classifier.parameters(), "lr": args.head_lr},
    ]
    return torch.optim.AdamW(groups, weight_decay=args.weight_decay)


def build_scheduler(args, optimizer):
    if args.scheduler == "constant":
        return None
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.min_lr
    )


def evaluate(loader, device, modules):
    wifi_encoder, mmwave_encoder, projector, token_projector, token_encoder, classifier = modules
    for module in modules:
        module.eval()
    confusion = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long)
    total_loss = 0.0
    total = 0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for wifi, mmwave, labels, _ in loader:
            wifi, mmwave, labels = wifi.to(device), mmwave.to(device), labels.to(device)
            wifi_input, mmwave_input = to_network_inputs(wifi, mmwave)
            wifi_map = wifi_encoder.forward_map(wifi_input)
            mmwave_map = mmwave_encoder.forward_map(mmwave_input)
            wifi_z, mmwave_z = projector(
                wifi_encoder.forward_global(wifi_map), mmwave_encoder.forward_global(mmwave_map)
            )
            tokens = token_encoder(token_projector(wifi_map, mmwave_map))
            logits = classifier((wifi_z, mmwave_z), tokens)
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
    f1 = (2 * precision * recall / (precision + recall).clamp_min(1e-8))[support > 0].mean().item()
    return total_loss / max(total, 1), accuracy, 100.0 * f1


def main():
    parser = argparse.ArgumentParser(description="Full two-modal sepdual fine-tuning on MM-Fi")
    parser.add_argument("--data_root", default="~/MMFi/MMFi_Dataset")
    parser.add_argument("--segments_csv", default="~/MMFi/MMFi_action_segments.csv")
    parser.add_argument("--pretrain_ckpt", default="")
    parser.add_argument("--no_pretrain", action="store_true")
    parser.add_argument("--save", default="./result/mmfi_sepdual_full_finetune.pth")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--head_lr", type=float, default=1e-4)
    parser.add_argument("--min_lr", type=float, default=0.0)
    parser.add_argument("--scheduler", choices=("cosine", "constant"), default="cosine")
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--eval_interval", type=int, default=5)
    parser.add_argument("--window_size", type=int, default=32)
    parser.add_argument("--max_points", type=int, default=128)
    parser.add_argument("--token_dim", type=int, default=512)
    parser.add_argument("--projection_dim", type=int, default=512)
    parser.add_argument("--z_hidden_dim", type=int, default=512)
    parser.add_argument("--num_tokens", type=int, default=32)
    parser.add_argument("--sharing_num", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--generator_depth", type=int, default=2)
    parser.add_argument("--generator_heads", type=int, default=8)
    parser.add_argument("--generator_dropout", type=float, default=0.1)
    parser.add_argument("--cls_gamma_init", type=float, default=0.1)
    parser.add_argument("--add_modality_embed", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    if args.no_pretrain and args.pretrain_ckpt:
        parser.error("--no_pretrain and --pretrain_ckpt cannot be used together")
    if not args.no_pretrain and not args.pretrain_ckpt:
        parser.error("provide --pretrain_ckpt or use --no_pretrain")

    seed_all(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    train_set = MMFiWindowDataset(
        args.data_root,
        args.segments_csv,
        environments=["E01"],
        split="train",
        window_size=args.window_size,
        max_points=args.max_points,
        seed=args.seed,
    )
    test_set = MMFiWindowDataset(
        args.data_root,
        args.segments_csv,
        environments=["E01"],
        split="test",
        window_size=args.window_size,
        max_points=args.max_points,
        seed=args.seed,
    )
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )

    modules = make_components(args, device)
    if not args.no_pretrain:
        load_pretrain(args.pretrain_ckpt, device, modules)

    optimizer = build_optimizer(args, modules)
    scheduler = build_scheduler(args, optimizer)
    best_accuracy = -1.0
    print(f"train samples: {len(train_set)} test samples: {len(test_set)}", flush=True)
    print(f"device: {device} scheduler: {args.scheduler}", flush=True)

    wifi_encoder, mmwave_encoder, projector, token_projector, token_encoder, classifier = modules
    for epoch in range(args.epochs):
        for module in modules:
            module.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for wifi, mmwave, labels, _ in train_loader:
            wifi, mmwave, labels = wifi.to(device), mmwave.to(device), labels.to(device)
            wifi_input, mmwave_input = to_network_inputs(wifi, mmwave)
            wifi_map = wifi_encoder.forward_map(wifi_input)
            mmwave_map = mmwave_encoder.forward_map(mmwave_input)
            wifi_z, mmwave_z = projector(
                wifi_encoder.forward_global(wifi_map), mmwave_encoder.forward_global(mmwave_map)
            )
            tokens = token_encoder(token_projector(wifi_map, mmwave_map))
            logits = classifier((wifi_z, mmwave_z), tokens)
            loss = F.cross_entropy(logits, labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [parameter for group in optimizer.param_groups for parameter in group["params"]], 1.0
            )
            optimizer.step()
            running_loss += loss.item() * labels.numel()
            correct += logits.argmax(1).eq(labels).sum().item()
            total += labels.numel()

        if scheduler is not None:
            scheduler.step()
        if (epoch + 1) % args.eval_interval == 0 or epoch + 1 == args.epochs:
            test_loss, test_accuracy, test_f1 = evaluate(test_loader, device, modules)
            print(
                f"epoch {epoch + 1:03d}/{args.epochs} "
                f"train_loss={running_loss / max(total, 1):.4f} "
                f"train_acc={100.0 * correct / max(total, 1):.2f} "
                f"source_test_loss={test_loss:.4f} "
                f"source_test_acc={test_accuracy:.2f} "
                f"source_test_f1={test_f1:.2f} "
                f"lr_downstream={optimizer.param_groups[0]['lr']:.6g} "
                f"lr_head={optimizer.param_groups[1]['lr']:.6g}",
                flush=True,
            )
            if test_accuracy > best_accuracy:
                best_accuracy = test_accuracy
                output = Path(args.save).expanduser()
                output.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "finetune_type": "mmfi_sepdual_full",
                        "args": vars(args),
                        "feature_dim": BACKBONE_FEATURE_DIM,
                        "projection_dim": args.projection_dim,
                        "token_dim": args.token_dim,
                        "best_acc": best_accuracy,
                        "wifi_encoder": wifi_encoder.state_dict(),
                        "mmwave_encoder": mmwave_encoder.state_dict(),
                        "projector": projector.state_dict(),
                        "token_projector": token_projector.state_dict(),
                        "token_encoder": token_encoder.state_dict(),
                        "classifier": classifier.state_dict(),
                    },
                    output,
                )
                print(f"saved best checkpoint: {output}", flush=True)


if __name__ == "__main__":
    main()


