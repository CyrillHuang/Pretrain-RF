import argparse
from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn as nn
import torch.utils.data as Data
from timm.models.layers import trunc_normal_

from xrf55_dataset import XRFMultiModalDataset
from model.fusion.gated_iin import Gate, build_mask
from xrf55_pretrain import MODALITY_NAMES, MultiModalTokenProjector

import argparse
import random
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
from model import resnet1d, resnet1d_rfid, resnet2d
from timm.models.layers import trunc_normal_
from model.fusion.gated_iin import Gate, build_mask
from xrf55_pretrain import MODALITY_NAMES, MultiModalTokenProjector, linear_projector

class ModalityAdapter(nn.Module):
    def __init__(self, feature_dim=512):
        super(ModalityAdapter, self).__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, feature_dim),
        )

    def forward(self, feature):
        return self.net(feature)
class MultiModalProjector(nn.Module):
    def __init__(self, feature_dim=512, projection_dim=256):
        super(MultiModalProjector, self).__init__()
        self.wifi_adapter = ModalityAdapter(feature_dim)
        self.rfid_adapter = ModalityAdapter(feature_dim)
        self.mmwave_adapter = ModalityAdapter(feature_dim)
        self.shared_projector = nn.Sequential(
            nn.Linear(feature_dim, projection_dim),
            nn.GELU(),
            nn.Linear(projection_dim, projection_dim),
        )

    def forward(self, wifi_feature, rfid_feature, mmwave_feature):
        wifi_z = self.shared_projector(self.wifi_adapter(wifi_feature))
        rfid_z = self.shared_projector(self.rfid_adapter(rfid_feature))
        mmwave_z = self.shared_projector(self.mmwave_adapter(mmwave_feature))
        return [wifi_z, rfid_z, mmwave_z]
class LargeMutualFeatureBranch(nn.Module):
    """Use XRF55 mutual-learning ResNet, but keep only its 1024-d feature."""

    def __init__(self, backbone, feature_dim=1024):
        super().__init__()
        self.backbone = backbone
        self.feature_dim = feature_dim

    def forward(self, x):
        output = self.backbone(x)
        if not isinstance(output, tuple) or len(output) < 2:
            raise RuntimeError("mutual backbone should return (logits, feature)")
        features = [item for item in output if item.dim() == 2 and item.size(1) == self.feature_dim]
        if len(features) != 1:
            shapes = [tuple(item.shape) for item in output if torch.is_tensor(item)]
            raise RuntimeError(
                f"expected exactly one {self.feature_dim}-d feature tensor from mutual backbone, got {shapes}"
            )
        return features[0]
def build_models(device):
    models = [
        LargeMutualFeatureBranch(resnet1d.resnet18_mutual(), feature_dim=1024),
        LargeMutualFeatureBranch(resnet1d_rfid.resnet18_mutual(), feature_dim=1024),
        LargeMutualFeatureBranch(resnet2d.resnet18_mutual(), feature_dim=1024),
    ]
    return [model.to(device) for model in models]
def forward_mmwave_map(backbone, x):
    if x.dim() == 5 and x.size(1) == 1:
        x = x.squeeze(1)
    x = backbone.conv1(x)
    x = backbone.bn1(x)
    x = backbone.relu(x)
    x = backbone.maxpool(x)
    x = backbone.layer1(x)
    x = backbone.layer2(x)
    x = backbone.layer3(x)
    x = backbone.layer4(x)
    return x
def forward_rfid_map(backbone, x):
    x = backbone.conv1(x)
    x = backbone.bn1(x)
    x = backbone.relu(x)
    x = backbone.maxpool(x)
    x = backbone.layer1(x)
    x = backbone.layer2(x)
    x = backbone.layer3(x)
    x = backbone.layer4(x)
    return x
def forward_wifi_map(backbone, x):
    x = backbone.conv1(x)
    x = backbone.bn1(x)
    x = backbone.relu(x)
    x = backbone.maxpool(x)
    x = backbone.layer1(x)
    x = backbone.layer2(x)
    x = backbone.layer3(x)
    x = backbone.layer4(x)
    return x
def pool_feature_maps(models, wifi_map, rfid_map, mmwave_map):
    wifi_feature = models[0].backbone.avg_pool(wifi_map).flatten(1)
    rfid_feature = models[1].backbone.avg_pool(rfid_map).flatten(1)
    mmwave_feature = models[2].backbone.avgpool(mmwave_map).flatten(1)
    return wifi_feature, rfid_feature, mmwave_feature
def forward_backbone_outputs(models, projector, samples_wifi, samples_rfid, samples_mmwave):
    wifi_map = forward_wifi_map(models[0].backbone, samples_wifi)
    rfid_map = forward_rfid_map(models[1].backbone, samples_rfid)
    mmwave_map = forward_mmwave_map(models[2].backbone, samples_mmwave)
    pooled_features = pool_feature_maps(models, wifi_map, rfid_map, mmwave_map)
    model_zs = projector(*pooled_features)
    return model_zs, wifi_map, rfid_map, mmwave_map
def gate_stats(meter):
    if meter["count"] == 0:
        return None
    mean = meter["sum"] / meter["count"]
    var = max(meter["sq_sum"] / meter["count"] - mean * mean, 0.0)
    return mean, var ** 0.5
def gate_text(stats):
    if stats is None:
        return ""
    return f" gate_mean {stats[0]:.4f} gate_std {stats[1]:.4f}"
def new_gate_meter():
    return {"sum": 0.0, "sq_sum": 0.0, "count": 0}
def set_mode(modules, train=True):
    for module in modules:
        module.train(train)
def set_requires_grad(modules, requires_grad):
    for module in modules:
        for param in module.parameters():
            param.requires_grad = requires_grad
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
def split_logits_gate(output):
    if isinstance(output, tuple):
        return output
    return output, None
def update_gate_meter(meter, gate):
    if gate is None:
        return
    gate = gate.detach().float()
    meter["sum"] += gate.sum().item()
    meter["sq_sum"] += gate.square().sum().item()
    meter["count"] += gate.numel()



def checkpoint_config(checkpoint, args=None):
    checkpoint_args = checkpoint.get("args", {})

    def first_not_none(*values, default):
        for value in values:
            if value is not None:
                return value
        return default

    feature_dim = first_not_none(
        checkpoint.get("feature_dim"),
        checkpoint_args.get("feature_dim"),
        default=1024,
    )
    projection_dim = first_not_none(
        checkpoint.get("projection_dim"),
        checkpoint_args.get("projection_dim"),
        default=512,
    )
    token_dim = first_not_none(
        checkpoint.get("token_dim"),
        checkpoint_args.get("token_dim"),
        default=512,
    )
    num_tokens = first_not_none(
        checkpoint.get("num_tokens"),
        checkpoint_args.get("num_tokens"),
        checkpoint_args.get("num_tokens_per_modality"),
        default=32,
    )
    encoder_depth = first_not_none(
        checkpoint.get("encoder_depth"),
        checkpoint_args.get("generator_depth"),
        checkpoint_args.get("encoder_depth"),
        default=2,
    )
    encoder_heads = first_not_none(
        checkpoint.get("encoder_heads"),
        checkpoint_args.get("generator_heads"),
        checkpoint_args.get("encoder_heads"),
        default=8,
    )
    encoder_mlp_ratio = first_not_none(
        checkpoint.get("encoder_mlp_ratio"),
        checkpoint_args.get("encoder_mlp_ratio"),
        default=4.0,
    )
    drop_rate = first_not_none(
        checkpoint.get("drop_rate"),
        checkpoint_args.get("generator_dropout"),
        checkpoint_args.get("dropout"),
        default=0.1,
    )
    config = {
        "feature_dim": int(feature_dim),
        "projection_dim": int(projection_dim),
        "token_dim": int(token_dim),
        "num_tokens_per_modality": int(num_tokens),
        "encoder_depth": int(encoder_depth),
        "encoder_heads": int(encoder_heads),
        "encoder_mlp_ratio": float(encoder_mlp_ratio),
        "drop_rate": float(drop_rate),
    }
    return config


class CleanGenTokenEncoder(nn.Module):
    """Reuse only sepdual type/position embeddings and source encoder."""

    def __init__(
        self,
        token_dim=512,
        num_tokens_per_modality=32,
        depth=2,
        num_heads=8,
        mlp_ratio=4.0,
        drop_rate=0.1,
    ):
        super().__init__()
        self.token_dim = token_dim
        self.num_tokens_per_modality = num_tokens_per_modality
        self.mod_to_id = {name: index for index, name in enumerate(MODALITY_NAMES)}
        self.type_embed = nn.Parameter(torch.zeros(3, 1, token_dim))
        self.pos_embed = nn.Parameter(torch.zeros(3, num_tokens_per_modality, token_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=num_heads,
            dim_feedforward=int(mlp_ratio * token_dim),
            dropout=drop_rate,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.source_encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)

    def _embed_tokens(self, domain, tokens):
        if tokens.size(1) > self.pos_embed.size(1):
            raise ValueError(
                f"{domain} has {tokens.size(1)} tokens, pos_embed supports {self.pos_embed.size(1)}"
            )
        mod_id = self.mod_to_id[domain]
        return tokens + self.type_embed[mod_id].unsqueeze(0) + self.pos_embed[mod_id, : tokens.size(1)].unsqueeze(0)

    def forward(self, input_task_tokens):
        output = OrderedDict()
        for domain in MODALITY_NAMES:
            embedded_tokens = self._embed_tokens(domain, input_task_tokens[domain])  # [B, 32, d]
            output[domain] = self.source_encoder(embedded_tokens)  # [B, 32, d]
        return output


def build_clean_gen_encoder(config):
    return CleanGenTokenEncoder(
        token_dim=config["token_dim"],
        num_tokens_per_modality=config["num_tokens_per_modality"],
        depth=config["encoder_depth"],
        num_heads=config["encoder_heads"],
        mlp_ratio=config["encoder_mlp_ratio"],
        drop_rate=config["drop_rate"],
    )


def load_clean_gen_encoder_state(clean_gen_encoder, generator_state):
    keep = {}
    for key, value in generator_state.items():
        if key in ("type_embed", "pos_embed") or key.startswith("source_encoder."):
            keep[key] = value
    missing, unexpected = clean_gen_encoder.load_state_dict(keep, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            "failed to load clean generation encoder state; "
            f"missing={missing}, unexpected={unexpected}"
        )


def load_pretrained_checkpoint(checkpoint_path, device, models, use_shared_encoder=True):
    checkpoint = torch.load(Path(checkpoint_path).expanduser(), map_location=device)
    required_keys = ("models", "projector", "token_projector")
    missing_keys = [key for key in required_keys if key not in checkpoint]
    if missing_keys:
        raise KeyError(
            f"sepdual pretrain checkpoint must contain {required_keys}; missing {missing_keys}"
        )
    if use_shared_encoder and "cross_modal_generator" not in checkpoint:
        raise KeyError(
            "shared-encoder finetune requires sepdual pretrain key 'cross_modal_generator'"
        )
    allowed_pretrain_types = ("large_sepdual", "large_sepdual_selfmae")
    if use_shared_encoder and checkpoint.get("pretrain_type") not in allowed_pretrain_types:
        raise ValueError(
            "finetune_sepdual.py expects a sepdual-compatible checkpoint with "
            f"pretrain_type in {allowed_pretrain_types}, "
            f"got {checkpoint.get('pretrain_type', 'unknown')!r}. "
            "Use --no_shared_encoder only for raw-token ablations."
        )

    config = checkpoint_config(checkpoint)
    projector = MultiModalProjector(
        feature_dim=config["feature_dim"],
        projection_dim=config["projection_dim"],
    ).to(device)
    token_projector = MultiModalTokenProjector(
        feature_dim=config["feature_dim"],
        token_dim=config["token_dim"],
    ).to(device)
    clean_gen_encoder = build_clean_gen_encoder(config).to(device) if use_shared_encoder else None

    for model, state_dict in zip(models, checkpoint["models"]):
        model.load_state_dict(state_dict)
    projector.load_state_dict(checkpoint["projector"])
    token_projector.linear_proj.load_state_dict(checkpoint["token_projector"])
    if clean_gen_encoder is not None:
        load_clean_gen_encoder_state(clean_gen_encoder, checkpoint["cross_modal_generator"])
    return checkpoint, projector, token_projector, clean_gen_encoder, config


def build_random_components(device, use_shared_encoder=True):
    """Build the same downstream inputs without loading any pretraining weights."""
    config = checkpoint_config({})
    projector = MultiModalProjector(
        feature_dim=config["feature_dim"],
        projection_dim=config["projection_dim"],
    ).to(device)
    token_projector = MultiModalTokenProjector(
        feature_dim=config["feature_dim"],
        token_dim=config["token_dim"],
    ).to(device)
    clean_gen_encoder = build_clean_gen_encoder(config).to(device) if use_shared_encoder else None
    return projector, token_projector, clean_gen_encoder, config


class LargeCleanGenEncoderZGuidedIIN(nn.Module):
    def __init__(
        self,
        token_dim=512,
        projection_dim=512,
        z_hidden_dim=512,
        sharing_num=4,
        dropout=0.1,
        cls_gamma_init=0.1,
        return_g=False,
        add_iin_modality_embed=False,
    ):
        super().__init__()
        self.token_dim = token_dim
        self.sharing_num = sharing_num
        self.return_g = return_g

        self.z_to_cls = nn.Sequential(
            nn.LayerNorm(projection_dim * 3),
            nn.Linear(projection_dim * 3, z_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(z_hidden_dim, token_dim),
            nn.LayerNorm(token_dim),
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, token_dim))
        self.cls_gamma = nn.Parameter(torch.tensor(float(cls_gamma_init)))
        self.cls_norm = nn.LayerNorm(token_dim)
        trunc_normal_(self.cls_token, std=0.02)

        self.mte = nn.Embedding(3, token_dim) if add_iin_modality_embed else None
        self.pos_embed = nn.Parameter(torch.zeros(1, 97, token_dim))
        trunc_normal_(self.pos_embed, std=0.02)

        self.inter_unit = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=8,
            dim_feedforward=token_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.intra_unit = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=8,
            dim_feedforward=token_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.gate = Gate(token_dim, return_g=return_g)
        self.mlp_head = nn.Sequential(
            nn.Linear(token_dim, token_dim // 2, bias=False),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(token_dim // 2, 55, bias=True),
        )

    def make_cls_token(self, model_zs):
        z_cm = self.z_to_cls(torch.cat(model_zs, dim=1))
        cls_token = self.cls_token.expand(z_cm.size(0), -1, -1)
        return self.cls_norm(cls_token + self.cls_gamma * z_cm.unsqueeze(1))

    def add_downstream_modality_embed(self, encoded_tokens):
        if self.mte is None:
            return encoded_tokens
        output = OrderedDict()
        for index, (domain, tokens) in enumerate(encoded_tokens.items()):
            output[domain] = tokens + self.mte(torch.tensor(index, device=tokens.device)).view(1, 1, -1)
        return output

    def forward(self, model_zs, encoded_tokens, return_g=None):
        if return_g is None:
            return_g = self.return_g

        encoded_tokens = self.add_downstream_modality_embed(encoded_tokens)
        wifi_tokens = encoded_tokens["wifi"]
        rfid_tokens = encoded_tokens["rfid"]
        mmwave_tokens = encoded_tokens["mmwave"]

        cls_token = self.make_cls_token(model_zs)
        x = torch.cat([cls_token, wifi_tokens, rfid_tokens, mmwave_tokens], dim=1)
        x = x + self.pos_embed[:, : x.size(1), :]

        inter_mask, intra_mask = build_mask(
            len_wifi=wifi_tokens.size(1),
            len_rfid=rfid_tokens.size(1),
            len_mmwave=mmwave_tokens.size(1),
            device=wifi_tokens.device,
        )

        x_inter = x
        x_intra = x
        for _ in range(self.sharing_num):
            x_inter = self.inter_unit(x_inter, src_mask=inter_mask)
            x_intra = self.intra_unit(x_intra, src_mask=intra_mask)

        cls_inter = x_inter[:, 0, :]
        cls_intra = x_intra[:, 0, :]
        if return_g:
            fused_cls, gate = self.gate(cls_intra, cls_inter, return_g=True)
            return self.mlp_head(fused_cls), gate
        fused_cls = self.gate(cls_intra, cls_inter, return_g=False)
        return self.mlp_head(fused_cls)


def build_optimizer(args, models, projector, token_projector, clean_gen_encoder, classifier):
    if args.freeze_encoder:
        downstream_params = list(projector.parameters()) + list(token_projector.parameters())
        if clean_gen_encoder is not None:
            downstream_params.extend(clean_gen_encoder.parameters())
        return torch.optim.AdamW(
            [
                # These modules are loaded from the pre-training checkpoint.
                {"params": downstream_params, "lr": args.lr},
                # The classifier/fusion head is initialized for downstream HAR.
                {"params": classifier.parameters(), "lr": args.head_lr},
            ],
            weight_decay=args.weight_decay,
        )

    pretrained_params = []
    for model in models:
        pretrained_params.extend(model.parameters())
    pretrained_params.extend(projector.parameters())
    pretrained_params.extend(token_projector.parameters())
    if clean_gen_encoder is not None:
        pretrained_params.extend(clean_gen_encoder.parameters())

    return torch.optim.AdamW(
        [
            {"params": pretrained_params, "lr": args.lr},
            {"params": classifier.parameters(), "lr": args.head_lr},
        ],
        weight_decay=args.weight_decay,
    )


def audit_freeze_configuration(
    args,
    models,
    projector,
    token_projector,
    clean_gen_encoder,
    classifier,
    optimizer,
):
    """Fail loudly if a frozen backbone can still be updated by the optimizer."""
    backbone_params = [param for model in models for param in model.parameters()]
    optimizer_param_ids = {
        id(param)
        for group in optimizer.param_groups
        for param in group["params"]
    }
    trainable_backbone = sum(param.requires_grad for param in backbone_params)
    optimizer_backbone = sum(id(param) in optimizer_param_ids for param in backbone_params)
    downstream_modules = [projector, token_projector, classifier]
    if clean_gen_encoder is not None:
        downstream_modules.append(clean_gen_encoder)
    downstream_params = [param for module in downstream_modules for param in module.parameters()]
    trainable_downstream = sum(param.requires_grad for param in downstream_params)

    print(
        f"freeze audit: backbone_params={len(backbone_params)} "
        f"trainable_backbone={trainable_backbone} "
        f"optimizer_backbone={optimizer_backbone} "
        f"trainable_downstream={trainable_downstream}"
    )

    if args.freeze_encoder:
        if trainable_backbone != 0:
            raise RuntimeError("freeze audit failed: some backbone parameters still require gradients")
        if optimizer_backbone != 0:
            raise RuntimeError("freeze audit failed: frozen backbone parameters are present in the optimizer")


def forward_logits(
    models,
    projector,
    token_projector,
    clean_gen_encoder,
    classifier,
    samples_wifi,
    samples_rfid,
    samples_mmwave,
    return_g=False,
):
    model_zs, wifi_map, rfid_map, mmwave_map = forward_backbone_outputs(
        models,
        projector,
        samples_wifi,
        samples_rfid,
        samples_mmwave,
    )
    raw_tokens = token_projector(wifi_map, rfid_map, mmwave_map)
    encoded_tokens = clean_gen_encoder(raw_tokens) if clean_gen_encoder is not None else raw_tokens
    return classifier(model_zs, encoded_tokens, return_g=return_g)


def train_one_epoch(
    epoch,
    args,
    device,
    train_loader,
    models,
    projector,
    token_projector,
    clean_gen_encoder,
    classifier,
    optimizer,
    criterion,
):
    encoder_modules = [*models, projector, token_projector]
    if clean_gen_encoder is not None:
        encoder_modules.append(clean_gen_encoder)
    if args.freeze_encoder:
        # Freeze only the three modality ResNet backbones. Projection, token,
        # shared-generation, and IIN classifier layers remain trainable.
        set_mode(models, train=False)
        set_mode([projector, token_projector, *([] if clean_gen_encoder is None else [clean_gen_encoder])], train=True)
        set_mode([classifier], train=True)
    else:
        set_mode([*encoder_modules, classifier], train=True)

    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    gate_meter = new_gate_meter()

    for step, (samples_wifi, samples_rfid, samples_mmwave, labels) in enumerate(train_loader, start=1):
        samples_wifi = samples_wifi.to(device, non_blocking=True)
        samples_rfid = samples_rfid.to(device, non_blocking=True)
        samples_mmwave = samples_mmwave.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if args.freeze_encoder:
            # Backbone parameters have requires_grad=False; projector and all
            # downstream modules still receive gradients.
            model_zs, wifi_map, rfid_map, mmwave_map = forward_backbone_outputs(
                models, projector, samples_wifi, samples_rfid, samples_mmwave
            )
            raw_tokens = token_projector(wifi_map, rfid_map, mmwave_map)
            encoded_tokens = clean_gen_encoder(raw_tokens) if clean_gen_encoder is not None else raw_tokens
            logits, gate = split_logits_gate(classifier(model_zs, encoded_tokens, return_g=args.return_g))
        else:
            logits, gate = split_logits_gate(
                forward_logits(
                    models,
                    projector,
                    token_projector,
                    clean_gen_encoder,
                    classifier,
                    samples_wifi,
                    samples_rfid,
                    samples_mmwave,
                    return_g=args.return_g,
                )
            )

        loss = criterion(logits, labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip > 0:
            params = [param for group in optimizer.param_groups for param in group["params"]]
            torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
        optimizer.step()

        batch_size = labels.size(0)
        total_samples += batch_size
        total_loss += loss.item() * batch_size
        total_correct += logits.argmax(dim=1).eq(labels).sum().item()
        update_gate_meter(gate_meter, gate)

        if args.log_interval > 0 and step % args.log_interval == 0:
            print(
                f"epoch {epoch:03d} step {step:05d}/{len(train_loader):05d} "
                f"loss {total_loss / total_samples:.4f} "
                f"acc {100.0 * total_correct / total_samples:.2f}"
                f"{gate_text(gate_stats(gate_meter))} "
                f"cls_gamma {classifier.cls_gamma.item():.4f}"
            )

    return total_loss / total_samples, 100.0 * total_correct / total_samples, gate_stats(gate_meter)


@torch.no_grad()
def evaluate(
    device,
    val_loader,
    models,
    projector,
    token_projector,
    clean_gen_encoder,
    classifier,
    criterion,
    return_g=False,
):
    modules = [*models, projector, token_projector, classifier]
    if clean_gen_encoder is not None:
        modules.append(clean_gen_encoder)
    set_mode(modules, train=False)
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    gate_meter = new_gate_meter()

    for samples_wifi, samples_rfid, samples_mmwave, labels in val_loader:
        samples_wifi = samples_wifi.to(device, non_blocking=True)
        samples_rfid = samples_rfid.to(device, non_blocking=True)
        samples_mmwave = samples_mmwave.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits, gate = split_logits_gate(
            forward_logits(
                models,
                projector,
                token_projector,
                clean_gen_encoder,
                classifier,
                samples_wifi,
                samples_rfid,
                samples_mmwave,
                return_g=return_g,
            )
        )
        loss = criterion(logits, labels)

        batch_size = labels.size(0)
        total_samples += batch_size
        total_loss += loss.item() * batch_size
        total_correct += logits.argmax(dim=1).eq(labels).sum().item()
        update_gate_meter(gate_meter, gate)

    return total_loss / total_samples, 100.0 * total_correct / total_samples, gate_stats(gate_meter)


def save_checkpoint(
    save_path,
    epoch,
    args,
    models,
    projector,
    token_projector,
    clean_gen_encoder,
    classifier,
    optimizer,
    scheduler,
    best_acc,
    config,
):
    save_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "epoch": epoch,
        "args": vars(args),
        "classifier_type": (
            "large_sepdual_z_iin"
            if args.use_clean_gen_encoder
            else "large_sepdual_raw_z_iin"
        ),
        "backbone_type": "resnet18_mutual",
        "feature_dim": config["feature_dim"],
        "projection_dim": config["projection_dim"],
        "token_dim": config["token_dim"],
        "num_tokens_per_modality": config["num_tokens_per_modality"],
        "encoder_depth": config["encoder_depth"],
        "encoder_heads": config["encoder_heads"],
        "encoder_mlp_ratio": config["encoder_mlp_ratio"],
        "drop_rate": config["drop_rate"],
        "use_clean_gen_encoder": bool(args.use_clean_gen_encoder),
        "models": [model.state_dict() for model in models],
        "projector": projector.state_dict(),
        "token_projector": token_projector.linear_proj.state_dict(),
        "classifier": classifier.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "best_acc": best_acc,
    }
    if clean_gen_encoder is not None:
        checkpoint["clean_gen_encoder"] = clean_gen_encoder.state_dict()
    torch.save(checkpoint, save_path)


def load_finetune_checkpoint(
    checkpoint_path,
    device,
    models,
    projector,
    token_projector,
    clean_gen_encoder,
    classifier,
    optimizer,
    scheduler,
):
    checkpoint = torch.load(Path(checkpoint_path).expanduser(), map_location=device)
    required_keys = ("models", "projector", "token_projector", "classifier", "optimizer")
    missing_keys = [key for key in required_keys if key not in checkpoint]
    if missing_keys:
        raise KeyError(f"finetune checkpoint missing keys: {missing_keys}")

    for model, state_dict in zip(models, checkpoint["models"]):
        model.load_state_dict(state_dict)
    projector.load_state_dict(checkpoint["projector"])
    token_projector.linear_proj.load_state_dict(checkpoint["token_projector"])
    classifier.load_state_dict(checkpoint["classifier"])

    if clean_gen_encoder is not None:
        if "clean_gen_encoder" not in checkpoint:
            raise KeyError("resume checkpoint does not contain clean_gen_encoder state")
        clean_gen_encoder.load_state_dict(checkpoint["clean_gen_encoder"])

    optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and checkpoint.get("scheduler") is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])

    start_epoch = int(checkpoint.get("epoch", -1)) + 1
    best_acc = float(checkpoint.get("best_acc", 0.0))
    return start_epoch, best_acc


def parse_args():
    parser = argparse.ArgumentParser(
        description="XRF55 fine-tuning with optional large sepdual pretraining and GCM-CLS IIN"
    )
    parser.add_argument("--pretrain_ckpt", "--ckpt", dest="pretrain_ckpt", type=str, default="")
    parser.add_argument(
        "--no_pretrain",
        action="store_true",
        help="Randomly initialize all encoders and fine-tune only with Scene 1 labels.",
    )
    parser.add_argument("--data_dir", type=str, default="~/XRF55-repo/dataset/XRF_dataset")
    parser.add_argument("--scene", type=str, default="dml")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--head_lr", type=float, default=1e-4)
    parser.add_argument("--min_lr", type=float, default=0.0)
    parser.add_argument(
        "--scheduler",
        choices=("cosine", "constant"),
        default="cosine",
        help="Learning-rate schedule; constant keeps each optimizer-group LR fixed.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--grad_clip", type=float, default=0.0)
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--freeze_encoder", action="store_true")
    parser.add_argument("--z_hidden_dim", type=int, default=None)
    parser.add_argument("--sharing_num", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--cls_gamma_init", type=float, default=0.1)
    parser.add_argument(
        "--no_shared_encoder",
        action="store_true",
        help="Use raw token_projector tokens in IIN instead of loading the sepdual shared encoder.",
    )
    parser.add_argument("--no_clean_gen_encoder", dest="no_shared_encoder", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--add_iin_modality_embed", action="store_true")
    parser.add_argument("--return_g", action="store_true")
    parser.add_argument("--save_dir", type=str, default="./result/finetune_sepdual")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--log_interval", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()
    if args.no_pretrain and args.pretrain_ckpt:
        parser.error("--no_pretrain cannot be used with --pretrain_ckpt")
    if not args.no_pretrain and not args.pretrain_ckpt:
        parser.error("provide --pretrain_ckpt, or use --no_pretrain for a random-initialization baseline")
    if args.no_pretrain and args.freeze_encoder:
        parser.error("--freeze_encoder is invalid with --no_pretrain")
    return args


def main():
    args = parse_args()
    args.use_clean_gen_encoder = not args.no_shared_encoder
    set_seed(args.seed)

    requested_device = torch.device(args.device)
    if requested_device.type == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = requested_device

    train_dataset = XRFMultiModalDataset(file_path=args.data_dir, is_train=True, scene=args.scene)
    val_dataset = XRFMultiModalDataset(file_path=args.data_dir, is_train=False, scene=args.scene)
    train_loader = Data.DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=device.type == "cuda",
        num_workers=args.num_workers,
        drop_last=True,
    )
    val_loader = Data.DataLoader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=device.type == "cuda",
        num_workers=args.num_workers,
        drop_last=False,
    )

    models = build_models(device)
    if args.no_pretrain:
        checkpoint = None
        projector, token_projector, clean_gen_encoder, config = build_random_components(
            device,
            use_shared_encoder=args.use_clean_gen_encoder,
        )
    else:
        checkpoint, projector, token_projector, clean_gen_encoder, config = load_pretrained_checkpoint(
            args.pretrain_ckpt,
            device,
            models,
            use_shared_encoder=args.use_clean_gen_encoder,
        )
    z_hidden_dim = args.z_hidden_dim or config["projection_dim"]
    classifier = LargeCleanGenEncoderZGuidedIIN(
        token_dim=config["token_dim"],
        projection_dim=config["projection_dim"],
        z_hidden_dim=z_hidden_dim,
        sharing_num=args.sharing_num,
        dropout=args.dropout,
        cls_gamma_init=args.cls_gamma_init,
        return_g=args.return_g,
        add_iin_modality_embed=args.add_iin_modality_embed,
    ).to(device)

    if args.freeze_encoder:
        # Only freeze the three modality backbones. The projector, token
        # projector, shared encoder, and IIN classifier remain trainable.
        set_requires_grad(models, False)

    optimizer = build_optimizer(args, models, projector, token_projector, clean_gen_encoder, classifier)
    audit_freeze_configuration(
        args,
        models,
        projector,
        token_projector,
        clean_gen_encoder,
        classifier,
        optimizer,
    )
    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
            eta_min=args.min_lr,
        )
    else:
        # Keep both optimizer groups at their configured learning rates.
        scheduler = None
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing).to(device)

    save_dir = Path(args.save_dir).expanduser()
    if checkpoint is None:
        print("pretraining: disabled; backbone and downstream encoders are randomly initialized")
    else:
        print(f"loaded pretrain checkpoint: {args.pretrain_ckpt}")
        print(f"pretrain epoch: {checkpoint.get('epoch', 'unknown')}")
        print(f"pretrain type: {checkpoint.get('pretrain_type', 'unknown')}")
    print(
        "classifier: "
        f"{'large_sepdual_z_iin' if args.use_clean_gen_encoder else 'large_sepdual_raw_z_iin'}"
    )
    if args.use_clean_gen_encoder:
        print("token path: featuremap -> token_projector -> clean type/pos -> shared source_encoder(separate modalities) -> z_guided_iin")
    else:
        print("token path: featuremap -> token_projector(raw tokens) -> z_guided_iin")
    print(f"feature dim: {config['feature_dim']}")
    print(f"projection dim: {config['projection_dim']}")
    print(f"token dim: {config['token_dim']}")
    print(f"encoder depth: {config['encoder_depth']}")
    print(f"encoder heads: {config['encoder_heads']}")
    print(f"use shared encoder: {args.use_clean_gen_encoder}")
    print(f"z hidden dim: {z_hidden_dim}")
    print(f"freeze encoder: {args.freeze_encoder}")
    print(f"label smoothing: {args.label_smoothing}")
    print(f"sharing num: {args.sharing_num}")
    print(f"cls gamma init: {args.cls_gamma_init}")
    print(f"add iin modality embed: {args.add_iin_modality_embed}")
    print(f"dropout: {args.dropout}")
    print(f"train samples: {len(train_dataset)}")
    print(f"val samples: {len(val_dataset)}")
    print(f"device: {device}")
    print(f"save dir: {save_dir}")

    best_acc = 0.0
    start_epoch = 0
    if args.resume:
        start_epoch, best_acc = load_finetune_checkpoint(
            args.resume,
            device,
            models,
            projector,
            token_projector,
            clean_gen_encoder,
            classifier,
            optimizer,
            scheduler,
        )
        print(f"resumed finetune checkpoint: {args.resume}")
        print(f"resume from epoch: {start_epoch}")
        print(f"resume best acc: {best_acc:.2f}")

    for epoch in range(start_epoch, args.epochs):
        train_loss, train_acc, train_gate = train_one_epoch(
            epoch,
            args,
            device,
            train_loader,
            models,
            projector,
            token_projector,
            clean_gen_encoder,
            classifier,
            optimizer,
            criterion,
        )
        val_loss, val_acc, val_gate = evaluate(
            device,
            val_loader,
            models,
            projector,
            token_projector,
            clean_gen_encoder,
            classifier,
            criterion,
            return_g=args.return_g,
        )
        if scheduler is not None:
            scheduler.step()

        lr_downstream = optimizer.param_groups[0]["lr"]
        lr_head = optimizer.param_groups[1]["lr"]
        print(
            f"epoch {epoch:03d} done "
            f"train_loss {train_loss:.4f} train_acc {train_acc:.2f} "
            f"val_loss {val_loss:.4f} val_acc {val_acc:.2f} "
            f"train{gate_text(train_gate)} "
            f"val{gate_text(val_gate)} "
            f"cls_gamma {classifier.cls_gamma.item():.4f} "
            f"lr_downstream {lr_downstream:.6g} "
            f"lr_head {lr_head:.6g}"
        )

        save_checkpoint(
            save_dir / "checkpoint_latest.pth",
            epoch,
            args,
            models,
            projector,
            token_projector,
            clean_gen_encoder,
            classifier,
            optimizer,
            scheduler,
            best_acc,
            config,
        )
        if val_acc > best_acc:
            best_acc = val_acc
            save_checkpoint(
                save_dir / "checkpoint_best.pth",
                epoch,
                args,
                models,
                projector,
                token_projector,
                clean_gen_encoder,
                classifier,
                optimizer,
                scheduler,
                best_acc,
                config,
            )


if __name__ == "__main__":
    main()

