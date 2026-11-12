import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data

from xrf55_dataset import XRFMultiModalDataset
from model import resnet1d, resnet1d_rfid, resnet2d
from model.fusion.gated_iin import linear_projector

import argparse
import random
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
from model import resnet1d, resnet1d_rfid, resnet2d

class MultiModalContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07, label_smoothing=0.1):
        super(MultiModalContrastiveLoss, self).__init__()
        self.temp = nn.Parameter(temperature * torch.ones([]))
        self.label_smoothing = label_smoothing

    def forward_pair(self, source_z, target_z):
        source_feats = F.normalize(source_z, dim=-1)
        target_feats = F.normalize(target_z, dim=-1)
        temp = self.temp.clamp(min=1e-6)
        sim_source2target = torch.matmul(source_feats, target_feats.T) / temp
        sim_target2source = torch.matmul(target_feats, source_feats.T) / temp
        targets = torch.arange(source_z.size(0), dtype=torch.long, device=source_z.device)
        loss = (
            F.cross_entropy(sim_source2target, targets, label_smoothing=self.label_smoothing)
            + F.cross_entropy(sim_target2source, targets, label_smoothing=self.label_smoothing)
        ) / 2
        return loss

    def forward(self, model_zs):
        pair_indices = ((0, 1), (0, 2), (1, 2))
        pair_losses = [self.forward_pair(model_zs[i], model_zs[j]) for i, j in pair_indices]
        return torch.stack(pair_losses).mean(), pair_losses
class MultiModalMatchingLoss(nn.Module):
    def __init__(self, projection_dim=256):
        super(MultiModalMatchingLoss, self).__init__()
        self.matching_head = nn.Sequential(
            nn.LayerNorm(projection_dim * 4),
            nn.Linear(projection_dim * 4, projection_dim),
            # nn.LayerNorm(projection_dim * 2),
            # nn.Linear(projection_dim * 2, projection_dim),
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
            temp = temperature.clamp(min=1e-6)
            sim_source2target = torch.matmul(source_feats, target_feats.T) / temp
            sim_target2source = torch.matmul(target_feats, source_feats.T) / temp
            sim_source2target.fill_diagonal_(-10000)
            sim_target2source.fill_diagonal_(-10000)
            weights_source2target = F.softmax(sim_source2target, dim=1)
            weights_target2source = F.softmax(sim_target2source, dim=1)
            # # Sample one non-diagonal sample as a hard negative based on similarity weights
            target_neg_indices = torch.multinomial(weights_source2target, num_samples=1).squeeze(1)
            source_neg_indices = torch.multinomial(weights_target2source, num_samples=1).squeeze(1)
            # 
        return source_neg_indices, target_neg_indices

    def forward_pair(self, source_z, target_z, temperature):
        source_neg_indices, target_neg_indices = self._sample_hard_negatives(
            source_z,
            target_z,
            temperature,
        )
        source_neg = source_z[source_neg_indices]
        target_neg = target_z[target_neg_indices]

        source_all = torch.cat([source_z, source_z, source_neg], dim=0)
        target_all = torch.cat([target_z, target_neg, target_z], dim=0)
        pair_features = torch.cat(
            [
                source_all,
                target_all,
                torch.abs(source_all - target_all),
                source_all * target_all,
            ],
            dim=-1,
        )
        logits = self.matching_head(pair_features)
        labels = torch.cat(
            [
                torch.ones(source_z.size(0), dtype=torch.long, device=source_z.device),
                torch.zeros(source_z.size(0) * 2, dtype=torch.long, device=source_z.device),
            ],
            dim=0,
        )
        return F.cross_entropy(logits, labels), logits

    def forward(self, model_zs, temperature):
        pair_indices = ((0, 1), (0, 2), (1, 2))
        pair_outputs = [self.forward_pair(model_zs[i], model_zs[j], temperature) for i, j in pair_indices]
        pair_losses = [output[0] for output in pair_outputs]
        return torch.stack(pair_losses).mean(), pair_outputs
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
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
def set_train_mode(modules):
    for module in modules:
        module.train()
class MultiModalMatrixKLLoss(nn.Module):
    def __init__(self, eps=1e-8):
        super(MultiModalMatrixKLLoss, self).__init__()
        self.eps = eps

    def _row_distribution(self, anchor_z, candidate_z, temperature):
        anchor_feats = F.normalize(anchor_z, dim=-1)
        candidate_feats = F.normalize(candidate_z, dim=-1)
        logits = torch.matmul(anchor_feats, candidate_feats.T) / temperature.clamp(min=1e-6)
        return F.softmax(logits, dim=1)

    def _anchor_group_loss(self, anchor_z, candidate_a_z, candidate_b_z, temperature):
        dist_a = self._row_distribution(anchor_z, candidate_a_z, temperature)
        dist_b = self._row_distribution(anchor_z, candidate_b_z, temperature)
        teacher = ((dist_a + dist_b) / 2).detach()
        loss_a = F.kl_div(torch.log(dist_a + self.eps), teacher, reduction="batchmean")
        loss_b = F.kl_div(torch.log(dist_b + self.eps), teacher, reduction="batchmean")
        return (loss_a + loss_b) / 2

    def forward(self, model_zs, temperature):
        wifi_z, rfid_z, mmwave_z = model_zs
        group_losses = [
            self._anchor_group_loss(wifi_z, rfid_z, mmwave_z, temperature),
            self._anchor_group_loss(rfid_z, wifi_z, mmwave_z, temperature),
            self._anchor_group_loss(mmwave_z, wifi_z, rfid_z, temperature),
        ]
        return torch.stack(group_losses).mean(), group_losses



MODALITY_NAMES = ("wifi", "rfid", "mmwave")
MODALITY_ABBR = {"wifi": "w", "rfid": "r", "mmwave": "m"}
REC_DIRECTIONS = MODALITY_NAMES


def rec_detail_key(target_mod):
    return f"rec_{target_mod}"


def rec_log_items(meters, denom):
    items = []
    for target_mod in REC_DIRECTIONS:
        key = rec_detail_key(target_mod)
        name = f"r{MODALITY_ABBR[target_mod]}"
        items.append(f"{name} {meters[key] / denom:.4f}")
    return " ".join(items)


def trunc_normal_(tensor, std=0.02):
    if hasattr(nn.init, "trunc_normal_"):
        nn.init.trunc_normal_(tensor, std=std)
    else:
        with torch.no_grad():
            tensor.normal_(mean=0.0, std=std)


def span_mask_tokens(x, min_span=3, max_span=5):
    """Mask one contiguous token span for each sample.

    Args:
        x: [B, N, D]
    Returns:
        Tokens with one zeroed span per sample, shape [B, N, D].
    """

    if max_span <= 0:
        return x

    batch_size, num_tokens, _ = x.shape
    min_span = max(1, min(min_span, num_tokens))
    max_span = max(min_span, min(max_span, num_tokens))
    keep_mask = torch.ones(batch_size, num_tokens, 1, device=x.device, dtype=x.dtype)
    span_lengths = torch.randint(min_span, max_span + 1, (batch_size,), device=x.device)

    for index in range(batch_size):
        span_len = int(span_lengths[index].item())
        start = int(torch.randint(0, num_tokens - span_len + 1, (1,), device=x.device).item())
        keep_mask[index, start:start + span_len] = 0

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
    """Augment source tokens.

    Args:
        x: [B, 32, d]
    Returns:
        Augmented source tokens with shape [B, 32, d].
    """

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


def global_reconstruction_loss(pred, target, global_loss="cos"):
    pred_g = pred.mean(dim=1)        # [B, d]
    target_g = target.mean(dim=1)    # [B, d]
    if global_loss == "cos":
        pred_g = F.normalize(pred_g, dim=-1, eps=1e-6)
        target_g = F.normalize(target_g, dim=-1, eps=1e-6)
        return 1 - F.cosine_similarity(pred_g, target_g, dim=-1).mean()
    if global_loss == "mse":
        return F.mse_loss(pred_g, target_g)
    raise ValueError(f"unknown global reconstruction loss: {global_loss}")


def cross_modal_generation_loss(preds, targets, rec_loss="cos", global_weight=0.0, global_loss="cos"):
    losses = []
    details = {}

    for target_mod in REC_DIRECTIONS:
        pred = preds[target_mod]                  # [B, 32, d]
        target = targets[target_mod].detach()     # [B, 32, d]
        if rec_loss == "mse":
            loss = F.mse_loss(pred, target)
        elif rec_loss == "cos":
            pred_n = F.normalize(pred, dim=-1, eps=1e-6)
            target_n = F.normalize(target, dim=-1, eps=1e-6)
            loss_token = 1 - F.cosine_similarity(pred_n, target_n, dim=-1)
            loss = loss_token.mean()
            if global_weight > 0:
                loss_global = global_reconstruction_loss(pred, target, global_loss)
                loss = loss + global_weight * loss_global
        else:
            raise ValueError(f"unknown reconstruction loss: {rec_loss}")

        losses.append(loss)
        details[rec_detail_key(target_mod)] = loss.detach()

    return torch.stack(losses).mean(), details


class LargeMutualFeatureBranch(nn.Module):
    """Use XRF55 mutual-learning ResNet while keeping its 1024-d feature."""

    def __init__(self, backbone, feature_dim=1024):
        super().__init__()
        self.backbone = backbone
        self.feature_dim = feature_dim

    def forward(self, x):
        output = self.backbone(x)
        if not isinstance(output, tuple) or len(output) < 2:
            raise RuntimeError("mutual backbone should return at least (logits, feature)")

        features = [
            item
            for item in output
            if torch.is_tensor(item) and item.dim() == 2 and item.size(1) == self.feature_dim
        ]
        if len(features) != 1:
            shapes = [tuple(item.shape) for item in output if torch.is_tensor(item)]
            raise RuntimeError(
                f"expected exactly one {self.feature_dim}-d feature tensor from mutual backbone, got {shapes}"
            )
        return features[0]


class MultiModalTokenProjector(nn.Module):
    """Project final feature maps into aligned token sequences.

    Input feature maps are expected to be:
        WiFi:   [B, 1024, 32]
        RFID:   [B, 1024, 5]
        mmWave: [B, 1024, 8, 4]
    Output tokens:
        each modality: [B, 32, d]
    """

    def __init__(self, feature_dim=1024, token_dim=512):
        super().__init__()
        self.linear_proj = linear_projector(input_dim=feature_dim, output_dim=token_dim)

    def forward(self, wifi_map, rfid_map, mmwave_map):
        wifi_tokens = self.linear_proj.wifi_linear_projection(wifi_map).transpose(1, 2)
        rfid_tokens = self.linear_proj.rfid_linear_projection(rfid_map).transpose(1, 2)
        mmwave_tokens = self.linear_proj.mmwave_linear_projection(mmwave_map).flatten(2).transpose(1, 2)
        return {
            "wifi": wifi_tokens,      # [B, 32, d]
            "rfid": rfid_tokens,      # [B, 32, d]
            "mmwave": mmwave_tokens,  # [B, 32, d]
        }


class LightweightSepDualCrossModalGenerator(nn.Module):
    """Separate-encoder dual-source target token generator.

    Each source modality is encoded independently by the same source encoder.
    Two encoded source memories are concatenated only for cross-attention K/V.
    """

    def __init__(
        self,
        token_dim=512,
        num_tokens=32,
        num_heads=8,
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
        self.token_dim = token_dim
        self.num_tokens = num_tokens
        self.drop_prob = drop_prob
        self.scale_low = scale_low
        self.scale_high = scale_high
        self.noise_std = noise_std
        self.span_mask = span_mask
        self.span_min = span_min
        self.span_max = span_max
        self.mod_to_id = {name: index for index, name in enumerate(MODALITY_NAMES)}

        self.type_embed = nn.Parameter(torch.zeros(3, 1, token_dim))
        self.pos_embed = nn.Parameter(torch.zeros(3, num_tokens, token_dim))
        self.target_query = nn.Parameter(torch.zeros(3, 1, token_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=num_heads,
            dim_feedforward=4 * token_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.source_encoder = nn.TransformerEncoder(encoder_layer, num_layers=encoder_depth)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=token_dim,
            num_heads=num_heads,
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
        trunc_normal_(self.type_embed, std=0.02)
        trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.target_query, std=0.02)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.weight, 1.0)
            nn.init.constant_(module.bias, 0)

    def add_source_embedding(self, tokens, mod):
        mod_id = self.mod_to_id[mod]
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
        """Encode one augmented source modality.

        Args:
            tokens: {"wifi"/"rfid"/"mmwave": [B, 32, d]}
            source_mod: source modality name.
        Returns:
            source_context: [B, 32, d]
        """
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
        source_tokens = self.add_source_embedding(source_tokens, source_mod)  # [B, 32, d]
        return self.source_encoder(source_tokens)  # [B, 32, d]

    def forward_one_target(self, tokens, target_mod):
        """Predict one clean target modality from the other two sources.

        Args:
            tokens: {"wifi"/"rfid"/"mmwave": [B, 32, d]}
            target_mod: target modality name.
        Returns:
            pred_target: [B, 32, d]
        """

        source_mods = [mod for mod in MODALITY_NAMES if mod != target_mod]
        source_context_1 = self.encode_source(tokens, source_mods[0])  # [B, 32, d]
        source_context_2 = self.encode_source(tokens, source_mods[1])  # [B, 32, d]
        source_context = torch.cat([source_context_1, source_context_2], dim=1)  # [B, 64, d]

        batch_size = source_context.size(0)
        target_queries = self.make_target_queries(
            batch_size=batch_size,
            target_mod=target_mod,
            device=source_context.device,
            dtype=source_context.dtype,
        )  # [B, 32, d]

        attn_out, _ = self.cross_attn(
            query=target_queries,  # [B, 32, d]
            key=source_context,    # [B, 64, d]
            value=source_context,  # [B, 64, d]
            need_weights=False,
        )
        hidden = target_queries + self.attn_dropout(attn_out)  # [B, 32, d]
        hidden = hidden + self.query_mlp(self.query_norm(hidden))  # [B, 32, d]
        return self.pred_head(hidden)  # [B, 32, d]

    def forward(self, tokens):
        return {
            target_mod: self.forward_one_target(tokens, target_mod)
            for target_mod in REC_DIRECTIONS
        }


def build_models(device, feature_dim=1024):
    models = [
        LargeMutualFeatureBranch(resnet1d.resnet18_mutual(), feature_dim=feature_dim),
        LargeMutualFeatureBranch(resnet1d_rfid.resnet18_mutual(), feature_dim=feature_dim),
        LargeMutualFeatureBranch(resnet2d.resnet18_mutual(), feature_dim=feature_dim),
    ]
    return [model.to(device) for model in models]


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


def pool_feature_maps(models, wifi_map, rfid_map, mmwave_map):
    wifi_feature = models[0].backbone.avg_pool(wifi_map).flatten(1)
    rfid_feature = models[1].backbone.avg_pool(rfid_map).flatten(1)
    mmwave_feature = models[2].backbone.avgpool(mmwave_map).flatten(1)
    return wifi_feature, rfid_feature, mmwave_feature


def forward_three_modalities(
    models,
    projector,
    token_projector,
    samples_wifi,
    samples_rfid,
    samples_mmwave,
):
    wifi_map = forward_wifi_map(models[0].backbone, samples_wifi)
    rfid_map = forward_rfid_map(models[1].backbone, samples_rfid)
    mmwave_map = forward_mmwave_map(models[2].backbone, samples_mmwave)
    model_features = pool_feature_maps(models, wifi_map, rfid_map, mmwave_map)
    model_zs = projector(*model_features)
    clean_tokens = token_projector(wifi_map, rfid_map, mmwave_map)
    return model_features, model_zs, clean_tokens


def collect_trainable_params(
    models,
    projector,
    token_projector,
    contrastive_loss_fn,
    matching_loss_fn,
    matrix_kl_loss_fn,
    cross_modal_generator,
):
    params = []
    for model in models:
        params.extend(model.parameters())
    params.extend(projector.parameters())
    params.extend(token_projector.parameters())
    params.extend(contrastive_loss_fn.parameters())
    params.extend(matching_loss_fn.parameters())
    params.extend(matrix_kl_loss_fn.parameters())
    params.extend(cross_modal_generator.parameters())
    return [param for param in params if param.requires_grad]


def build_optimizer(
    args,
    models,
    projector,
    token_projector,
    contrastive_loss_fn,
    matching_loss_fn,
    matrix_kl_loss_fn,
    cross_modal_generator,
):
    params = collect_trainable_params(
        models,
        projector,
        token_projector,
        contrastive_loss_fn,
        matching_loss_fn,
        matrix_kl_loss_fn,
        cross_modal_generator,
    )
    return torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)


def save_checkpoint(
    save_path,
    epoch,
    args,
    models,
    projector,
    token_projector,
    contrastive_loss_fn,
    matching_loss_fn,
    matrix_kl_loss_fn,
    cross_modal_generator,
    optimizer,
    scheduler,
):
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "args": vars(args),
            "pretrain_type": "large_sepdual",
            "backbone_type": "resnet18_mutual",
            "feature_dim": args.feature_dim,
            "projection_dim": args.projection_dim,
            "token_dim": args.token_dim,
            "num_tokens": args.num_tokens,
            "models": [model.state_dict() for model in models],
            "projector": projector.state_dict(),
            "token_projector": token_projector.linear_proj.state_dict(),
            "contrastive_loss": contrastive_loss_fn.state_dict(),
            "matching_loss": matching_loss_fn.state_dict(),
            "matrix_kl_loss": matrix_kl_loss_fn.state_dict(),
            "cross_modal_generator": cross_modal_generator.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
        },
        save_path,
    )


def load_checkpoint(
    checkpoint_path,
    device,
    models,
    projector,
    token_projector,
    contrastive_loss_fn,
    matching_loss_fn,
    matrix_kl_loss_fn,
    cross_modal_generator,
    optimizer,
    scheduler,
):
    checkpoint = torch.load(Path(checkpoint_path).expanduser(), map_location=device)
    for model, state_dict in zip(models, checkpoint["models"]):
        model.load_state_dict(state_dict)
    projector.load_state_dict(checkpoint["projector"])
    if "token_projector" in checkpoint:
        token_projector.linear_proj.load_state_dict(checkpoint["token_projector"])
    contrastive_loss_fn.load_state_dict(checkpoint["contrastive_loss"])
    matching_loss_fn.load_state_dict(checkpoint["matching_loss"])
    if "matrix_kl_loss" in checkpoint:
        matrix_kl_loss_fn.load_state_dict(checkpoint["matrix_kl_loss"])
    if "cross_modal_generator" in checkpoint:
        cross_modal_generator.load_state_dict(checkpoint["cross_modal_generator"])

    optimizer_loaded = True
    try:
        optimizer.load_state_dict(checkpoint["optimizer"])
    except ValueError as exc:
        optimizer_loaded = False
        print(f"warning: skip optimizer state because parameter groups changed: {exc}")
    if optimizer_loaded and scheduler is not None and checkpoint.get("scheduler") is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])
    return int(checkpoint["epoch"]) + 1


def train_one_epoch(
    epoch,
    args,
    device,
    train_loader,
    models,
    projector,
    token_projector,
    contrastive_loss_fn,
    matching_loss_fn,
    matrix_kl_loss_fn,
    cross_modal_generator,
    optimizer,
):
    set_train_mode(
        [
            *models,
            projector,
            token_projector,
            contrastive_loss_fn,
            matching_loss_fn,
            matrix_kl_loss_fn,
            cross_modal_generator,
        ]
    )
    meters = {
        "loss": 0.0,
        "con": 0.0,
        "mat": 0.0,
        "kl": 0.0,
        "rec": 0.0,
    }
    for target_mod in REC_DIRECTIONS:
        meters[rec_detail_key(target_mod)] = 0.0
    seen = 0

    for step, (samples_wifi, samples_rfid, samples_mmwave, _) in enumerate(train_loader, start=1):
        batch_size = samples_wifi.size(0)
        if batch_size < 2:
            continue

        samples_wifi = samples_wifi.to(device, non_blocking=True)
        samples_rfid = samples_rfid.to(device, non_blocking=True)
        samples_mmwave = samples_mmwave.to(device, non_blocking=True)

        _, model_zs, clean_tokens = forward_three_modalities(
            models,
            projector,
            token_projector,
            samples_wifi,
            samples_rfid,
            samples_mmwave,
        )

        loss_con, _ = contrastive_loss_fn(model_zs)
        temperature = contrastive_loss_fn.temp.detach()
        loss_mat, _ = matching_loss_fn(model_zs, temperature)
        loss_kl, _ = matrix_kl_loss_fn(model_zs, temperature)
        preds = cross_modal_generator(clean_tokens)
        loss_rec, rec_details = cross_modal_generation_loss(
            preds,
            clean_tokens,
            rec_loss=args.rec_loss,
            global_weight=args.global_weight,
            global_loss=args.global_loss,
        )
        loss = (
            args.con_weight * loss_con
            + args.mat_weight * loss_mat
            + args.kl_weight * loss_kl
            + args.rec_weight * loss_rec
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip > 0:
            params = collect_trainable_params(
                models,
                projector,
                token_projector,
                contrastive_loss_fn,
                matching_loss_fn,
                matrix_kl_loss_fn,
                cross_modal_generator,
            )
            torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
        optimizer.step()

        if args.learn_temperature:
            with torch.no_grad():
                contrastive_loss_fn.temp.clamp_(min=args.min_temperature, max=args.max_temperature)

        seen += batch_size
        meters["loss"] += loss.item() * batch_size
        meters["con"] += loss_con.item() * batch_size
        meters["mat"] += loss_mat.item() * batch_size
        meters["kl"] += loss_kl.item() * batch_size
        meters["rec"] += loss_rec.item() * batch_size
        for key, value in rec_details.items():
            meters[key] += float(value.cpu()) * batch_size

        if args.log_interval > 0 and step % args.log_interval == 0:
            denom = max(seen, 1)
            rec_items = rec_log_items(meters, denom)
            print(
                f"epoch {epoch:03d} step {step:05d}/{len(train_loader):05d} "
                f"loss {meters['loss'] / denom:.4f} "
                f"con {meters['con'] / denom:.4f} "
                f"mat {meters['mat'] / denom:.4f} "
                f"kl {meters['kl'] / denom:.4f} "
                f"rec {meters['rec'] / denom:.4f} "
                f"{rec_items} "
                f"temp {float(contrastive_loss_fn.temp.detach().cpu()):.4f}"
            )

    denom = max(seen, 1)
    return {key: value / denom for key, value in meters.items()}


def parse_args():
    parser = argparse.ArgumentParser(
        description="XRF55 large pretraining with con/mat/KL and separate-encoder dual-source token generation"
    )
    parser.add_argument("--data_dir", type=str, default="~/XRF55-repo/dataset/XRF_dataset")
    parser.add_argument("--scene", type=str, default="dml")
    parser.add_argument(
        "--use_all_source_data",
        action="store_true",
        help="Use both source train and val splits for self-supervised pre-training.",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--min_lr", type=float, default=0.0)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--learn_temperature", dest="learn_temperature", action="store_true", default=True)
    parser.add_argument("--fix_temperature", dest="learn_temperature", action="store_false")
    parser.add_argument("--min_temperature", type=float, default=0.03)
    parser.add_argument("--max_temperature", type=float, default=1.0)
    parser.add_argument("--feature_dim", type=int, default=1024)
    parser.add_argument("--projection_dim", type=int, default=512)
    parser.add_argument("--token_dim", type=int, default=512)
    parser.add_argument("--num_tokens", type=int, default=32)
    parser.add_argument("--generator_depth", type=int, default=2)
    parser.add_argument("--generator_heads", type=int, default=8)
    parser.add_argument("--generator_dropout", type=float, default=0.1)
    parser.add_argument("--token_drop_prob", type=float, default=0)
    parser.add_argument("--span_min", type=int, default=4)
    parser.add_argument("--span_max", type=int, default=8)
    parser.add_argument("--scale_low", type=float, default=0.9)
    parser.add_argument("--scale_high", type=float, default=1.1)
    parser.add_argument("--noise_std", type=float, default=0.02)
    parser.add_argument("--con_label_smoothing", type=float, default=0.1)
    parser.add_argument("--con_weight", type=float, default=1.0)
    parser.add_argument("--mat_weight", type=float, default=1.0)
    parser.add_argument("--kl_weight", type=float, default=1.0)
    parser.add_argument("--rec_weight", type=float, default=1.0)
    parser.add_argument("--rec_loss", type=str, default="mse", choices=("cos", "mse"))
    parser.add_argument("--global_weight", type=float, default=0)
    parser.add_argument("--global_loss", type=str, default="cos", choices=("cos", "mse"))
    parser.add_argument("--grad_clip", type=float, default=0.0)
    parser.add_argument("--save_dir", type=str, default="./result/pretrain_sepdual")
    parser.add_argument("--save_interval", type=int, default=10)
    parser.add_argument("--log_interval", type=int, default=20)
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda:0")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    requested_device = torch.device(args.device)
    if requested_device.type == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = requested_device

    train_dataset = XRFMultiModalDataset(
        file_path=args.data_dir,
        is_train=True,
        scene=args.scene,
        split="all" if args.use_all_source_data else "train",
    )
    train_loader = Data.DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=device.type == "cuda",
        num_workers=args.num_workers,
        drop_last=True,
    )

    models = build_models(device, feature_dim=args.feature_dim)
    projector = MultiModalProjector(feature_dim=args.feature_dim, projection_dim=args.projection_dim).to(device)
    token_projector = MultiModalTokenProjector(feature_dim=args.feature_dim, token_dim=args.token_dim).to(device)
    contrastive_loss_fn = MultiModalContrastiveLoss(
        temperature=args.temperature,
        label_smoothing=args.con_label_smoothing,
    ).to(device)
    contrastive_loss_fn.temp.requires_grad_(args.learn_temperature)
    matching_loss_fn = MultiModalMatchingLoss(projection_dim=args.projection_dim).to(device)
    matrix_kl_loss_fn = MultiModalMatrixKLLoss().to(device)
    cross_modal_generator = LightweightSepDualCrossModalGenerator(
        token_dim=args.token_dim,
        num_tokens=args.num_tokens,
        num_heads=args.generator_heads,
        encoder_depth=args.generator_depth,
        dropout=args.generator_dropout,
        drop_prob=args.token_drop_prob,
        scale_low=args.scale_low,
        scale_high=args.scale_high,
        noise_std=args.noise_std,
        span_min=args.span_min,
        span_max=args.span_max,
    ).to(device)

    optimizer = build_optimizer(
        args,
        models,
        projector,
        token_projector,
        contrastive_loss_fn,
        matching_loss_fn,
        matrix_kl_loss_fn,
        cross_modal_generator,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.min_lr,
    )

    start_epoch = 0
    if args.resume:
        start_epoch = load_checkpoint(
            args.resume,
            device,
            models,
            projector,
            token_projector,
            contrastive_loss_fn,
            matching_loss_fn,
            matrix_kl_loss_fn,
            cross_modal_generator,
            optimizer,
            scheduler,
        )

    with torch.no_grad():
        if args.learn_temperature:
            contrastive_loss_fn.temp.clamp_(min=args.min_temperature, max=args.max_temperature)
        else:
            contrastive_loss_fn.temp.fill_(args.temperature)

    save_dir = Path(args.save_dir).expanduser()
    print(f"train samples: {len(train_dataset)}")
    print(f"batches per epoch: {len(train_loader)}")
    print(f"device: {device}")
    print(f"save dir: {save_dir}")
    print("backbone: resnet18_mutual")
    print("pretrain type: large_sepdual")
    print(f"feature dim: {args.feature_dim}")
    print(f"projection dim: {args.projection_dim}")
    print(f"token dim: {args.token_dim}")
    print(
        "generator: "
        f"depth {args.generator_depth} heads {args.generator_heads} dropout {args.generator_dropout}"
    )
    print(
        "source augmentation: "
        f"drop {args.token_drop_prob} span [{args.span_min}, {args.span_max}] "
        f"scale [{args.scale_low}, {args.scale_high}] noise {args.noise_std}"
    )
    print(
        "loss weights: "
        f"con {args.con_weight} mat {args.mat_weight} kl {args.kl_weight} rec {args.rec_weight}"
    )
    print(
        f"reconstruction loss: {args.rec_loss} "
        f"global_weight {args.global_weight} global_loss {args.global_loss}"
    )
    print(
        f"temperature: {args.temperature} learn_temperature: {args.learn_temperature} "
        f"clamp [{args.min_temperature}, {args.max_temperature}]"
    )

    for epoch in range(start_epoch, args.epochs):
        stats = train_one_epoch(
            epoch,
            args,
            device,
            train_loader,
            models,
            projector,
            token_projector,
            contrastive_loss_fn,
            matching_loss_fn,
            matrix_kl_loss_fn,
            cross_modal_generator,
            optimizer,
        )
        scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        rec_items = rec_log_items(stats, 1.0)
        print(
            f"epoch {epoch:03d} done "
            f"loss {stats['loss']:.4f} "
            f"con {stats['con']:.4f} "
            f"mat {stats['mat']:.4f} "
            f"kl {stats['kl']:.4f} "
            f"rec {stats['rec']:.4f} "
            f"{rec_items} "
            f"lr {current_lr:.6g}"
        )

        latest_path = save_dir / "checkpoint_latest.pth"
        save_checkpoint(
            latest_path,
            epoch,
            args,
            models,
            projector,
            token_projector,
            contrastive_loss_fn,
            matching_loss_fn,
            matrix_kl_loss_fn,
            cross_modal_generator,
            optimizer,
            scheduler,
        )
        if args.save_interval > 0 and (epoch + 1) % args.save_interval == 0:
            save_checkpoint(
                save_dir / f"checkpoint_epoch{epoch:03d}.pth",
                epoch,
                args,
                models,
                projector,
                token_projector,
                contrastive_loss_fn,
                matching_loss_fn,
                matrix_kl_loss_fn,
                cross_modal_generator,
                optimizer,
                scheduler,
            )


if __name__ == "__main__":
    main()

