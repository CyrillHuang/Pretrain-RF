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


NUM_MODALITIES = 2
NUM_CLASSES = 27
BACKBONE_FEATURE_DIM = 1024
MODALITY_NAMES = ("wifi", "mmwave")
REC_DIRECTIONS = ("wifi", "mmwave")


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
        wifi_z = self.shared_projector(self.wifi_adapter(wifi_feature))
        mmwave_z = self.shared_projector(self.mmwave_adapter(mmwave_feature))
        return wifi_z, mmwave_z


class TokenProjector(nn.Module):
    """Convert each ResNet feature map into a fixed 32-token sequence."""

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
        x = projector(x.flatten(2))  # [B, token_dim, spatial]
        if x.size(-1) != self.num_tokens:
            x = F.interpolate(x, size=self.num_tokens, mode="linear", align_corners=False)
        return x.transpose(1, 2)  # [B, tokens, D]

    def forward(self, wifi_map, mmwave_map):
        return {
            "wifi": self._to_tokens(wifi_map, self.wifi_proj),
            "mmwave": self._to_tokens(mmwave_map, self.mmwave_proj),
        }


def span_mask_tokens(x, min_span=4, max_span=8):
    """Mask one contiguous token span per sample, as in the XRF55 generator."""
    batch_size, num_tokens, _ = x.shape
    min_span = max(1, min(int(min_span), num_tokens))
    max_span = max(min_span, min(int(max_span), num_tokens))
    keep_mask = torch.ones(batch_size, num_tokens, 1, device=x.device, dtype=x.dtype)
    span_lengths = torch.randint(min_span, max_span + 1, (batch_size,), device=x.device)
    for index in range(batch_size):
        span_len = int(span_lengths[index].item())
        start = int(torch.randint(0, num_tokens - span_len + 1, (1,), device=x.device).item())
        keep_mask[index, start : start + span_len] = 0
    return x * keep_mask


def augment_tokens(
    x,
    drop_prob=0.0,
    scale_low=1.0,
    scale_high=1.0,
    noise_std=0.0,
    span_mask=True,
    span_min=4,
    span_max=8,
):
    """Apply the same source-token augmentation used by XRF55 sepdual."""
    if scale_low != 1.0 or scale_high != 1.0:
        scale = torch.empty(x.size(0), 1, 1, device=x.device, dtype=x.dtype)
        scale.uniform_(scale_low, scale_high)
        x = x * scale
    if noise_std > 0:
        sample_std = x.std(dim=(1, 2), keepdim=True).clamp_min(1e-6)
        x = x + torch.randn_like(x) * (noise_std * sample_std)
    if drop_prob > 0:
        keep_mask = (torch.rand(x.size(0), x.size(1), 1, device=x.device) >= drop_prob).to(x.dtype)
        x = x * keep_mask
    if span_mask:
        x = span_mask_tokens(x, min_span=span_min, max_span=span_max)
    return x


class LocalCrossModalGenerator(nn.Module):
    """Two-modal version of XRF55's separate-source cross-modal generator.

    With only WiFi and mmWave, each target is reconstructed from the other
    modality. The source encoder, type/position embeddings, target queries,
    cross-attention, and prediction head follow the original generator.
    """

    def __init__(
        self,
        token_dim=256,
        num_tokens=32,
        heads=8,
        encoder_depth=2,
        dropout=0.1,
        drop_prob=0.0,
        scale_low=1.0,
        scale_high=1.0,
        noise_std=0.0,
        span_mask=True,
        span_min=4,
        span_max=8,
    ):
        super().__init__()
        self.token_dim = int(token_dim)
        self.num_tokens = int(num_tokens)
        self.drop_prob = float(drop_prob)
        self.scale_low = float(scale_low)
        self.scale_high = float(scale_high)
        self.noise_std = float(noise_std)
        self.span_mask = bool(span_mask)
        self.span_min = int(span_min)
        self.span_max = int(span_max)
        self.mod_to_id = {name: index for index, name in enumerate(MODALITY_NAMES)}

        self.type_embed = nn.Parameter(torch.zeros(NUM_MODALITIES, 1, token_dim))
        self.pos_embed = nn.Parameter(torch.zeros(NUM_MODALITIES, num_tokens, token_dim))
        self.target_query = nn.Parameter(torch.zeros(NUM_MODALITIES, 1, token_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=heads,
            dim_feedforward=4 * token_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.source_encoder = nn.TransformerEncoder(encoder_layer, num_layers=encoder_depth)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=token_dim,
            num_heads=heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_dropout = nn.Dropout(dropout)
        self.query_norm = nn.LayerNorm(token_dim)
        self.query_mlp = nn.Sequential(
            nn.Linear(token_dim, 4 * token_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * token_dim, token_dim),
            nn.Dropout(dropout),
        )
        self.pred_head = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, 4 * token_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * token_dim, token_dim),
        )
        self.apply(self._init_weights)
        nn.init.trunc_normal_(self.type_embed, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.target_query, std=0.02)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.weight, 1.0)
            nn.init.constant_(module.bias, 0)

    def add_source_embedding(self, tokens, source_mod):
        mod_id = self.mod_to_id[source_mod]
        return tokens + self.type_embed[mod_id].unsqueeze(0) + self.pos_embed[mod_id].unsqueeze(0)

    def make_target_queries(self, batch_size, target_mod, device, dtype):
        target_id = self.mod_to_id[target_mod]
        query = self.target_query[target_id].unsqueeze(0).expand(batch_size, self.num_tokens, self.token_dim)
        query = query.to(device=device, dtype=dtype)
        return (
            query
            + self.type_embed[target_id].unsqueeze(0).to(device=device, dtype=dtype)
            + self.pos_embed[target_id].unsqueeze(0).to(device=device, dtype=dtype)
        )

    def encode_source(self, tokens, source_mod):
        source_tokens = augment_tokens(
            tokens[source_mod],
            drop_prob=self.drop_prob,
            scale_low=self.scale_low,
            scale_high=self.scale_high,
            noise_std=self.noise_std,
            span_mask=self.span_mask,
            span_min=self.span_min,
            span_max=self.span_max,
        )
        source_tokens = self.add_source_embedding(source_tokens, source_mod)
        return self.source_encoder(source_tokens)

    def forward_one_target(self, tokens, target_mod):
        source_mod = "mmwave" if target_mod == "wifi" else "wifi"
        source_context = self.encode_source(tokens, source_mod)
        target_queries = self.make_target_queries(
            batch_size=source_context.size(0),
            target_mod=target_mod,
            device=source_context.device,
            dtype=source_context.dtype,
        )
        attn_out, _ = self.cross_attn(
            query=target_queries,
            key=source_context,
            value=source_context,
            need_weights=False,
        )
        hidden = target_queries + self.attn_dropout(attn_out)
        hidden = hidden + self.query_mlp(self.query_norm(hidden))
        return self.pred_head(hidden)

    def forward(self, tokens):
        return {
            "wifi_to_mmwave": self.forward_one_target(tokens, "mmwave"),
            "mmwave_to_wifi": self.forward_one_target(tokens, "wifi"),
        }


def to_network_inputs(wifi, mmwave):
    # WiFi: [B,T,3,114,10] -> [B,3,T*114,10]
    wifi_input = wifi.permute(0, 2, 1, 3, 4).reshape(
        wifi.size(0), wifi.size(2), wifi.size(1) * wifi.size(3), wifi.size(4)
    )
    # mmWave: [B,T,P,5] -> [B,5,T,P]
    mmwave_input = mmwave.permute(0, 3, 1, 2)
    return wifi_input, mmwave_input


class MultiModalContrastiveLoss(nn.Module):
    """Two-modal version of the XRF55 symmetric InfoNCE objective."""

    def __init__(self, temperature=0.07, label_smoothing=0.1):
        super().__init__()
        self.temp = nn.Parameter(torch.tensor(float(temperature)))
        self.label_smoothing = float(label_smoothing)

    def forward(self, wifi_z, mmwave_z):
        wifi_feats = F.normalize(wifi_z, dim=-1)
        mmwave_feats = F.normalize(mmwave_z, dim=-1)
        temperature = self.temp.clamp(min=1e-6)
        logits = wifi_feats @ mmwave_feats.t() / temperature
        labels = torch.arange(wifi_z.size(0), device=wifi_z.device)
        return 0.5 * (
            F.cross_entropy(logits, labels, label_smoothing=self.label_smoothing)
            + F.cross_entropy(logits.t(), labels, label_smoothing=self.label_smoothing)
        )


class MatchingHead(nn.Module):
    """Hard-negative matching head aligned with XRF55's matching loss."""

    def __init__(self, projection_dim=256):
        super().__init__()
        self.matching_head = nn.Sequential(
            nn.LayerNorm(projection_dim * 4),
            nn.Linear(projection_dim * 4, projection_dim),
            nn.GELU(),
            nn.Linear(projection_dim, 2),
        )

    @staticmethod
    def _sample_hard_negatives(source_z, target_z, temperature):
        if source_z.size(0) < 2:
            raise ValueError("matching loss needs batch_size >= 2 for negative pairs")
        with torch.no_grad():
            source_feats = F.normalize(source_z, dim=-1)
            target_feats = F.normalize(target_z, dim=-1)
            sim_source2target = source_feats @ target_feats.t() / temperature.clamp(min=1e-6)
            sim_target2source = target_feats @ source_feats.t() / temperature.clamp(min=1e-6)
            sim_source2target.fill_diagonal_(-10000)
            sim_target2source.fill_diagonal_(-10000)
            target_neg = torch.multinomial(F.softmax(sim_source2target, dim=1), 1).squeeze(1)
            source_neg = torch.multinomial(F.softmax(sim_target2source, dim=1), 1).squeeze(1)
        return source_neg, target_neg

    def forward(self, wifi_z, mmwave_z, temperature):
        source_neg, target_neg = self._sample_hard_negatives(wifi_z, mmwave_z, temperature)
        source_neg_z = wifi_z[source_neg]
        target_neg_z = mmwave_z[target_neg]
        source_all = torch.cat([wifi_z, wifi_z, source_neg_z], dim=0)
        target_all = torch.cat([mmwave_z, target_neg_z, mmwave_z], dim=0)
        features = torch.cat(
            [source_all, target_all, torch.abs(source_all - target_all), source_all * target_all], dim=-1
        )
        labels = torch.cat(
            [
                torch.ones(wifi_z.size(0), device=wifi_z.device, dtype=torch.long),
                torch.zeros(2 * wifi_z.size(0), device=wifi_z.device, dtype=torch.long),
            ]
        )
        return F.cross_entropy(self.matching_head(features), labels)


def similarity_kl_loss(wifi_z, mmwave_z, temperature):
    """Align within-modality sample-relations, avoiding the two-modal transpose degeneracy."""
    if wifi_z.size(0) < 2:
        return wifi_z.new_zeros(())
    wifi_z = F.normalize(wifi_z, dim=-1)
    mmwave_z = F.normalize(mmwave_z, dim=-1)
    wifi_logits = wifi_z @ wifi_z.t() / temperature
    mmwave_logits = mmwave_z @ mmwave_z.t() / temperature
    diagonal = torch.eye(wifi_z.size(0), device=wifi_z.device, dtype=torch.bool)
    wifi_logits = wifi_logits.masked_fill(diagonal, float("-inf"))
    mmwave_logits = mmwave_logits.masked_fill(diagonal, float("-inf"))
    wifi_prob = F.softmax(wifi_logits, dim=-1)
    mmwave_prob = F.softmax(mmwave_logits, dim=-1)
    return 0.5 * (
        F.kl_div(torch.log(wifi_prob.clamp_min(1e-8)), mmwave_prob.detach(), reduction="batchmean")
        + F.kl_div(torch.log(mmwave_prob.clamp_min(1e-8)), wifi_prob.detach(), reduction="batchmean")
    )


def reconstruction_loss(pred, target, loss_type="mse"):
    """Compute token-level cross-modal reconstruction loss.

    The XRF55 sepdual implementation uses raw token MSE by default.  Do not
    normalize both 512-dimensional token vectors before MSE: that changes the
    scale to roughly O(1/token_dim), making reconstruction almost irrelevant
    compared with the contrastive and matching losses.
    """
    target = target.detach()
    if loss_type == "mse":
        return F.mse_loss(pred, target)
    if loss_type == "cos":
        pred_n = F.normalize(pred, dim=-1, eps=1e-6)
        target_n = F.normalize(target, dim=-1, eps=1e-6)
        return (1.0 - F.cosine_similarity(pred_n, target_n, dim=-1)).mean()
    raise ValueError(f"unknown reconstruction loss: {loss_type}")


def build_scheduler(args, optimizer):
    if args.scheduler == "constant":
        return None
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.min_lr
    )


def main():
    parser = argparse.ArgumentParser(description="Full two-modal sepdual pre-training on MM-Fi")
    parser.add_argument("--data_root", default="~/MMFi/MMFi_Dataset")
    parser.add_argument("--segments_csv", default="~/MMFi/MMFi_action_segments.csv")
    parser.add_argument("--save", default="./result/mmfi_sepdual_full_pretrain.pth")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--min_lr", type=float, default=0.0)
    parser.add_argument("--scheduler", choices=("cosine", "constant"), default="cosine")
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--window_size", type=int, default=32)
    parser.add_argument("--max_points", type=int, default=128)
    parser.add_argument("--token_dim", type=int, default=512)
    parser.add_argument("--projection_dim", type=int, default=512)
    parser.add_argument("--num_tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--learn_temperature", dest="learn_temperature", action="store_true", default=True)
    parser.add_argument("--fix_temperature", dest="learn_temperature", action="store_false")
    parser.add_argument("--min_temperature", type=float, default=0.03)
    parser.add_argument("--max_temperature", type=float, default=1.0)
    parser.add_argument("--con_label_smoothing", type=float, default=0.1)
    parser.add_argument("--con_weight", type=float, default=1.0)
    parser.add_argument("--match_weight", type=float, default=1.0)
    parser.add_argument("--rec_weight", type=float, default=1.0)
    parser.add_argument("--rec_loss", choices=("mse", "cos"), default="mse")
    parser.add_argument("--kl_weight", type=float, default=1.0)
    parser.add_argument("--use_kl", action="store_true")
    parser.add_argument("--token_drop_prob", type=float, default=0.0)
    parser.add_argument("--span_min", type=int, default=4)
    parser.add_argument("--span_max", type=int, default=8)
    parser.add_argument("--scale_low", type=float, default=0.9)
    parser.add_argument("--scale_high", type=float, default=1.1)
    parser.add_argument("--noise_std", type=float, default=0.02)
    parser.add_argument("--generator_depth", type=int, default=2)
    parser.add_argument("--generator_heads", type=int, default=8)
    parser.add_argument("--generator_dropout", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    seed_all(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dataset = MMFiWindowDataset(
        args.data_root,
        args.segments_csv,
        environments=["E01"],
        split="train",
        window_size=args.window_size,
        max_points=args.max_points,
        seed=args.seed,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )

    wifi_encoder = ResNetBranch(3).to(device)
    mmwave_encoder = ResNetBranch(5).to(device)
    projector = GlobalProjector(BACKBONE_FEATURE_DIM, args.projection_dim).to(device)
    token_projector = TokenProjector(BACKBONE_FEATURE_DIM, args.token_dim, args.num_tokens).to(device)
    contrastive_loss_fn = MultiModalContrastiveLoss(
        temperature=args.temperature,
        label_smoothing=args.con_label_smoothing,
    ).to(device)
    contrastive_loss_fn.temp.requires_grad_(args.learn_temperature)
    generator = LocalCrossModalGenerator(
        args.token_dim,
        args.num_tokens,
        heads=args.generator_heads,
        encoder_depth=args.generator_depth,
        dropout=args.generator_dropout,
        drop_prob=args.token_drop_prob,
        scale_low=args.scale_low,
        scale_high=args.scale_high,
        noise_std=args.noise_std,
        span_mask=True,
        span_min=args.span_min,
        span_max=args.span_max,
    ).to(device)
    matching_head = MatchingHead(args.projection_dim).to(device)

    parameters = list(wifi_encoder.parameters()) + list(mmwave_encoder.parameters())
    parameters += list(projector.parameters()) + list(token_projector.parameters())
    parameters += list(contrastive_loss_fn.parameters()) + list(generator.parameters())
    parameters += list(matching_head.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = build_scheduler(args, optimizer)

    print(f"pretrain samples: {len(dataset)}", flush=True)
    print(f"device: {device}", flush=True)
    print(f"scheduler: {args.scheduler}", flush=True)
    for epoch in range(args.epochs):
        wifi_encoder.train()
        mmwave_encoder.train()
        projector.train()
        token_projector.train()
        contrastive_loss_fn.train()
        generator.train()
        matching_head.train()
        meters = {"loss": 0.0, "con": 0.0, "match": 0.0, "rec": 0.0, "kl": 0.0}
        seen = 0

        for wifi, mmwave, _, _ in loader:
            wifi = wifi.to(device, non_blocking=True)
            mmwave = mmwave.to(device, non_blocking=True)
            if wifi.size(0) < 2:
                continue
            wifi_input, mmwave_input = to_network_inputs(wifi, mmwave)
            wifi_map = wifi_encoder.forward_map(wifi_input)
            mmwave_map = mmwave_encoder.forward_map(mmwave_input)
            wifi_global = wifi_encoder.forward_global(wifi_map)
            mmwave_global = mmwave_encoder.forward_global(mmwave_map)
            wifi_z, mmwave_z = projector(wifi_global, mmwave_global)
            raw_tokens = token_projector(wifi_map, mmwave_map)
            predictions = generator(raw_tokens)

            loss_con = contrastive_loss_fn(wifi_z, mmwave_z)
            temperature = contrastive_loss_fn.temp.detach()
            loss_match = matching_head(wifi_z, mmwave_z, temperature)
            loss_rec = reconstruction_loss(
                predictions["wifi_to_mmwave"], raw_tokens["mmwave"], args.rec_loss
            )
            loss_rec = loss_rec + reconstruction_loss(
                predictions["mmwave_to_wifi"], raw_tokens["wifi"], args.rec_loss
            )
            loss_kl = similarity_kl_loss(wifi_z, mmwave_z, temperature)
            loss = (
                args.con_weight * loss_con
                + args.match_weight * loss_match
                + args.rec_weight * loss_rec
                + (args.kl_weight * loss_kl if args.use_kl else 0.0)
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(parameters, args.grad_clip)
            optimizer.step()
            if args.learn_temperature:
                with torch.no_grad():
                    contrastive_loss_fn.temp.clamp_(
                        min=args.min_temperature, max=args.max_temperature
                    )

            batch = wifi.size(0)
            seen += batch
            meters["loss"] += loss.item() * batch
            meters["con"] += loss_con.item() * batch
            meters["match"] += loss_match.item() * batch
            meters["rec"] += loss_rec.item() * batch
            meters["kl"] += loss_kl.item() * batch

        if scheduler is not None:
            scheduler.step()
        denom = max(seen, 1)
        print(
            f"epoch {epoch + 1:03d}/{args.epochs} "
            f"loss={meters['loss'] / denom:.4f} "
            f"con={meters['con'] / denom:.4f} "
            f"match={meters['match'] / denom:.4f} "
            f"rec={meters['rec'] / denom:.4f} "
            f"kl={meters['kl'] / denom:.4f} "
            f"use_kl={args.use_kl} "
            f"rec_loss={args.rec_loss} "
            f"temp={float(contrastive_loss_fn.temp.detach()):.4f} "
            f"lr={optimizer.param_groups[0]['lr']:.6g}",
            flush=True,
        )

    output = Path(args.save).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "pretrain_type": "mmfi_sepdual_full",
            "epoch": args.epochs,
            "args": vars(args),
            "feature_dim": BACKBONE_FEATURE_DIM,
            "projection_dim": args.projection_dim,
            "token_dim": args.token_dim,
            "num_tokens": args.num_tokens,
            "wifi_encoder": wifi_encoder.state_dict(),
            "mmwave_encoder": mmwave_encoder.state_dict(),
            "projector": projector.state_dict(),
            "token_projector": token_projector.state_dict(),
            "cross_modal_generator": generator.state_dict(),
            "contrastive_loss": contrastive_loss_fn.state_dict(),
            "matching_head": matching_head.state_dict(),
        },
        output,
    )
    print(f"saved pretrain checkpoint: {output}", flush=True)


if __name__ == "__main__":
    main()


