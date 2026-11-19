import argparse
from pathlib import Path

import torch
import torch.nn as nn


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
from model.fusion.gated_iin import Gate, build_mask, linear_projector
from xrf55_finetune import (
    LargeCleanGenEncoderZGuidedIIN,
    build_clean_gen_encoder,
    checkpoint_config,
    gate_stats,
    gate_text,
    new_gate_meter,
    split_logits_gate,
    update_gate_meter,
)
from xrf55_pretrain import MultiModalTokenProjector

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
class LargeZGuidedGatedIIN(nn.Module):
    def __init__(
        self,
        feature_dim=1024,
        d_l=256,
        projection_dim=512,
        z_hidden_dim=512,
        sharing_num=4,
        dropout=0.1,
        cls_gamma_init=0.1,
        return_g=False,
    ):
        super().__init__()
        self.d_l = d_l
        self.feature_dim = feature_dim
        self.sharing_num = sharing_num
        self.return_g = return_g

        self.linear_proj = linear_projector(input_dim=feature_dim, output_dim=d_l)
        self.proj_wifi = self.linear_proj.wifi_linear_projection
        self.proj_rfid = self.linear_proj.rfid_linear_projection
        self.proj_mmwave = self.linear_proj.mmwave_linear_projection

        self.z_to_cls = nn.Sequential(
            nn.LayerNorm(projection_dim * 3),
            nn.Linear(projection_dim * 3, z_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(z_hidden_dim, d_l),
            nn.LayerNorm(d_l),
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_l))
        self.cls_gamma = nn.Parameter(torch.tensor(float(cls_gamma_init)))
        self.cls_norm = nn.LayerNorm(d_l)
        trunc_normal_(self.cls_token, std=0.02)

        self.mte = nn.Embedding(3, d_l)
        self.pos_embed = nn.Parameter(torch.zeros(1, 97, d_l))
        trunc_normal_(self.pos_embed, std=0.02)

        self.inter_unit = nn.TransformerEncoderLayer(
            d_model=d_l,
            nhead=8,
            dim_feedforward=d_l * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.intra_unit = nn.TransformerEncoderLayer(
            d_model=d_l,
            nhead=8,
            dim_feedforward=d_l * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.gate = Gate(d_l, return_g=return_g)
        self.mlp_head = nn.Sequential(
            nn.Linear(d_l, d_l // 2, bias=False),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_l // 2, 55, bias=True),
        )

    def make_cls_token(self, model_zs):
        z_cm = self.z_to_cls(torch.cat(model_zs, dim=1))
        cls_token = self.cls_token.expand(z_cm.size(0), -1, -1)
        return self.cls_norm(cls_token + self.cls_gamma * z_cm.unsqueeze(1))

    def forward(self, model_zs, wifi_f, rfid_f, mmwave_f, return_g=None):
        if return_g is None:
            return_g = self.return_g

        wifi_proj = self.proj_wifi(wifi_f).transpose(1, 2)
        rfid_proj = self.proj_rfid(rfid_f).transpose(1, 2)
        mmwave_proj = self.proj_mmwave(mmwave_f).transpose(1, 2)

        device = wifi_f.device
        wifi_embed = wifi_proj + self.mte(torch.tensor(0, device=device)).view(1, 1, -1)
        rfid_embed = rfid_proj + self.mte(torch.tensor(1, device=device)).view(1, 1, -1)
        mmwave_embed = mmwave_proj + self.mte(torch.tensor(2, device=device)).view(1, 1, -1)

        cls_token = self.make_cls_token(model_zs)
        x = torch.cat([cls_token, wifi_embed, rfid_embed, mmwave_embed], dim=1)
        x = x + self.pos_embed[:, : x.size(1), :]

        inter_mask, intra_mask = build_mask(
            len_wifi=wifi_proj.size(1),
            len_rfid=rfid_proj.size(1),
            len_mmwave=mmwave_proj.size(1),
            device=device,
        )

        x_inter = x
        x_intra = x
        for _ in range(self.sharing_num):
            x_inter = self.inter_unit(x_inter, src_mask=inter_mask)
            x_intra = self.intra_unit(x_intra, src_mask=intra_mask)

        cls_inter = x_inter[:, 0, :]
        cls_intra = x_intra[:, 0, :]

        if return_g:
            fused_cls, g = self.gate(cls_intra, cls_inter, return_g=True)
            return self.mlp_head(fused_cls), g
        fused_cls = self.gate(cls_intra, cls_inter, return_g=False)
        return self.mlp_head(fused_cls)
def load_mmwave(path):
    record = np.load(path)
    if record.shape == (1, 17, 256, 128):
        record = record.squeeze(0)
    if record.shape == (256, 128, 17):
        record = record.transpose(2, 0, 1)
    if record.shape != (17, 256, 128):
        raise ValueError(f"expected shape {(17, 256, 128)} or {(256, 128, 17)}, got {record.shape}")
    return torch.from_numpy(np.ascontiguousarray(record)).float()
def to_channels_first_1d(record, channels, time_steps):
    if record.shape == (time_steps, channels):
        record = record.transpose(1, 0)
    if record.shape != (channels, time_steps):
        raise ValueError(
            f"expected shape {(channels, time_steps)} or {(time_steps, channels)}, got {record.shape}"
        )
    return np.ascontiguousarray(record)
def load_rfid(path):
    record = np.load(path)
    record = to_channels_first_1d(record, channels=23, time_steps=148)
    return torch.from_numpy(record).float()
def load_wifi(path):
    record = np.load(path)
    record = to_channels_first_1d(record, channels=270, time_steps=1000)
    return torch.from_numpy(record).float()
class CrossSceneFileDataset(Data.Dataset):
    def __init__(self, data_root, entries):
        super().__init__()
        self.data_root = Path(data_root).expanduser()
        self.entries = entries

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        file_name, label = self.entries[index]
        wifi_data = load_wifi(self.data_root / "WiFi" / f"{file_name}.npy")
        rfid_data = load_rfid(self.data_root / "RFID" / f"{file_name}.npy")
        mmwave_data = load_mmwave(self.data_root / "mmWave" / f"{file_name}.npy")
        return wifi_data, rfid_data, mmwave_data, label
SHOT_DIRS = {
    "oneshot": "one_shot",
    "twoshot": "two_shot",
}
def read_txt_entries(txt_path):
    entries = []
    with Path(txt_path).expanduser().open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            file_name, _, action_id = line.strip().split(",")
            entries.append((file_name, int(action_id) - 1))
    return entries
def build_few_shot_datasets(args):
    if args.split_dir:
        split_dir = Path(args.split_dir).expanduser()
    else:
        shot_dir = args.shot_dir or SHOT_DIRS[args.strategy]
        split_dir = Path(args.split_root).expanduser() / args.scene / shot_dir

    train_txt = split_dir / f"{args.txt_name}_train.txt"
    test_txt = split_dir / f"{args.txt_name}_val.txt"
    if not train_txt.exists() or not test_txt.exists():
        raise FileNotFoundError(
            f"expected {train_txt} and {test_txt}; create the split first or pass --split_dir"
        )

    train_dataset = CrossSceneFileDataset(
        split_dir / "train_data",
        read_txt_entries(train_txt),
    )
    test_dataset = CrossSceneFileDataset(
        split_dir / "test_data",
        read_txt_entries(test_txt),
    )
    return train_dataset, test_dataset
def numeric_name_key(file_name):
    key = []
    for part in file_name.split("_"):
        key.append(int(part) if part.isdigit() else part)
    return key
def parse_action_label(file_name):
    parts = file_name.split("_")
    if len(parts) < 2:
        raise ValueError(f"expected file name like subject_action_time, got {file_name}")
    return int(parts[1]) - 1
def read_raw_entries(scene_root):
    scene_root = Path(scene_root).expanduser()
    modality_files = []
    for modality in ("WiFi", "RFID", "mmWave"):
        modality_dir = scene_root / modality
        files = {path.stem for path in modality_dir.glob("*.npy")}
        if not files:
            raise FileNotFoundError(f"no npy files found in {modality_dir}")
        modality_files.append(files)

    shared_files = sorted(set.intersection(*modality_files), key=numeric_name_key)
    if not shared_files:
        raise RuntimeError(f"no shared samples across WiFi/RFID/mmWave in {scene_root}")
    return [(file_name, parse_action_label(file_name)) for file_name in shared_files]
def build_zero_shot_dataset(args):
    if args.eval_split_dir:
        split_dir = Path(args.eval_split_dir).expanduser()
        entries = read_txt_entries(split_dir / f"{args.txt_name}_val.txt")
        data_root = split_dir / "test_data"
    else:
        data_root = Path(args.raw_root).expanduser() / args.scene
        entries = read_raw_entries(data_root)
    return CrossSceneFileDataset(data_root, entries)
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
def forward_feature_maps(models, samples_wifi, samples_rfid, samples_mmwave):
    wifi_map = forward_wifi_map(models[0].backbone, samples_wifi)
    rfid_map = forward_rfid_map(models[1].backbone, samples_rfid)
    mmwave_map = forward_mmwave_map(models[2].backbone, samples_mmwave)
    return wifi_map, rfid_map, mmwave_map
def pool_feature_maps(models, wifi_map, rfid_map, mmwave_map):
    wifi_feature = models[0].backbone.avg_pool(wifi_map).flatten(1)
    rfid_feature = models[1].backbone.avg_pool(rfid_map).flatten(1)
    mmwave_feature = models[2].backbone.avgpool(mmwave_map).flatten(1)
    return wifi_feature, rfid_feature, mmwave_feature
def forward_projected_inputs(models, projector, samples_wifi, samples_rfid, samples_mmwave):
    wifi_map, rfid_map, mmwave_map = forward_feature_maps(models, samples_wifi, samples_rfid, samples_mmwave)
    pooled_features = pool_feature_maps(models, wifi_map, rfid_map, mmwave_map)
    model_zs = projector(*pooled_features)
    return model_zs, wifi_map, rfid_map, mmwave_map
def make_loader(dataset, args, shuffle, drop_last=False):
    return Data.DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        drop_last=drop_last,
    )
def set_mode(modules, train):
    for module in modules:
        module.train(train)
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)



def pick_config(cli_value, checkpoint, checkpoint_args, name, default):
    if cli_value is not None:
        return cli_value
    value = checkpoint.get(name, None)
    if value is not None:
        return value
    value = checkpoint_args.get(name, None)
    if value is not None:
        return value
    return default


def infer_use_clean_gen_encoder(args, checkpoint):
    if args.clean_gen_encoder == "on":
        return True
    if args.clean_gen_encoder == "off":
        return False
    if "use_clean_gen_encoder" in checkpoint:
        return bool(checkpoint["use_clean_gen_encoder"])
    checkpoint_args = checkpoint.get("args", {})
    if "use_clean_gen_encoder" in checkpoint_args:
        return bool(checkpoint_args["use_clean_gen_encoder"])
    if "no_clean_gen_encoder" in checkpoint_args:
        return not bool(checkpoint_args["no_clean_gen_encoder"])
    return "clean_gen_encoder" in checkpoint


def build_classifier_from_checkpoint(args, checkpoint, device, config):
    checkpoint_args = checkpoint.get("args", {})
    z_hidden_dim = pick_config(
        args.z_hidden_dim,
        checkpoint,
        checkpoint_args,
        "z_hidden_dim",
        config["projection_dim"],
    )
    sharing_num = pick_config(args.sharing_num, checkpoint, checkpoint_args, "sharing_num", 4)
    dropout = pick_config(args.dropout, checkpoint, checkpoint_args, "dropout", 0.1)
    cls_gamma_init = pick_config(args.cls_gamma_init, checkpoint, checkpoint_args, "cls_gamma_init", 0.1)
    add_iin_modality_embed = bool(args.add_iin_modality_embed or checkpoint_args.get("add_iin_modality_embed", False))

    classifier = LargeCleanGenEncoderZGuidedIIN(
        token_dim=config["token_dim"],
        projection_dim=config["projection_dim"],
        z_hidden_dim=int(z_hidden_dim),
        sharing_num=int(sharing_num),
        dropout=float(dropout),
        cls_gamma_init=float(cls_gamma_init),
        return_g=args.return_g,
        add_iin_modality_embed=add_iin_modality_embed,
    ).to(device)
    return classifier, {
        **config,
        "z_hidden_dim": int(z_hidden_dim),
        "sharing_num": int(sharing_num),
        "dropout": float(dropout),
        "cls_gamma_init": float(cls_gamma_init),
        "add_iin_modality_embed": add_iin_modality_embed,
    }


def infer_legacy_config(args, checkpoint):
    checkpoint_args = checkpoint.get("args", {})
    classifier_state = checkpoint["classifier"]
    feature_dim = pick_config(args.feature_dim, checkpoint, checkpoint_args, "feature_dim", 1024)
    projection_dim = pick_config(args.projection_dim, checkpoint, checkpoint_args, "projection_dim", 512)
    iin_dim = pick_config(args.iin_dim, checkpoint, checkpoint_args, "iin_dim", None)
    if iin_dim is None:
        iin_dim = pick_config(args.token_dim, checkpoint, checkpoint_args, "token_dim", None)
    if iin_dim is None and "pos_embed" in classifier_state:
        iin_dim = classifier_state["pos_embed"].shape[-1]
    if iin_dim is None:
        iin_dim = 256

    z_hidden_dim = pick_config(args.z_hidden_dim, checkpoint, checkpoint_args, "z_hidden_dim", None)
    if z_hidden_dim is None and "z_to_cls.1.weight" in classifier_state:
        z_hidden_dim = classifier_state["z_to_cls.1.weight"].shape[0]
    if z_hidden_dim is None:
        z_hidden_dim = projection_dim

    sharing_num = pick_config(args.sharing_num, checkpoint, checkpoint_args, "sharing_num", 4)
    dropout = pick_config(args.dropout, checkpoint, checkpoint_args, "dropout", 0.1)
    cls_gamma_init = pick_config(args.cls_gamma_init, checkpoint, checkpoint_args, "cls_gamma_init", 0.1)
    return {
        "feature_dim": int(feature_dim),
        "projection_dim": int(projection_dim),
        "iin_dim": int(iin_dim),
        "z_hidden_dim": int(z_hidden_dim),
        "sharing_num": int(sharing_num),
        "dropout": float(dropout),
        "cls_gamma_init": float(cls_gamma_init),
        "legacy_featuremap_iin": True,
    }


def build_legacy_classifier_from_checkpoint(args, checkpoint, device):
    config = infer_legacy_config(args, checkpoint)
    classifier = LargeZGuidedGatedIIN(
        feature_dim=config["feature_dim"],
        d_l=config["iin_dim"],
        projection_dim=config["projection_dim"],
        z_hidden_dim=config["z_hidden_dim"],
        sharing_num=config["sharing_num"],
        dropout=config["dropout"],
        cls_gamma_init=config["cls_gamma_init"],
        return_g=args.return_g,
    ).to(device)
    return classifier, config


def load_classification_checkpoint(checkpoint_path, device, models, args):
    checkpoint = torch.load(Path(checkpoint_path).expanduser(), map_location=device)
    required_keys = ("models", "projector", "classifier")
    missing_keys = [key for key in required_keys if key not in checkpoint]
    if missing_keys:
        raise KeyError(
            f"checkpoint must contain {required_keys}; missing {missing_keys}. "
                "Use finetune_sepdual.py checkpoint_best/latest.pth."
        )

    if "token_projector" not in checkpoint:
        if args.clean_gen_encoder == "on":
            raise KeyError(
                "this checkpoint has no 'token_projector', so it cannot use sepdual encoder mode. "
                "Use --clean_gen_encoder auto/off, or run the current xrf55_cross_domain.py entrypoint."
            )
        args.use_clean_gen_encoder = False
        args.legacy_featuremap_iin = True
        classifier, config = build_legacy_classifier_from_checkpoint(args, checkpoint, device)
        projector = MultiModalProjector(
            feature_dim=config["feature_dim"],
            projection_dim=config["projection_dim"],
        ).to(device)
        for model, state_dict in zip(models, checkpoint["models"]):
            model.load_state_dict(state_dict)
        projector.load_state_dict(checkpoint["projector"])
        classifier.load_state_dict(checkpoint["classifier"])
        return checkpoint, projector, None, None, classifier, config

    args.legacy_featuremap_iin = False
    args.use_clean_gen_encoder = infer_use_clean_gen_encoder(args, checkpoint)
    if args.use_clean_gen_encoder and "clean_gen_encoder" not in checkpoint:
        raise KeyError(
            "this checkpoint/config requests sepdual encoder, but checkpoint has no 'clean_gen_encoder'. "
            "Pass --clean_gen_encoder off if this checkpoint was trained without it."
        )
    allowed_classifier_types = (
        "large_sepdual_z_iin",
        "large_sepdual_nopretrain_z_iin",
    )
    if args.use_clean_gen_encoder and checkpoint.get("classifier_type") not in allowed_classifier_types:
        raise ValueError(
            "cross_scene_sepdual.py expects a sepdual checkpoint with "
            f"classifier_type in {allowed_classifier_types}, "
            f"got {checkpoint.get('classifier_type', 'unknown')!r}. "
            "Pass --clean_gen_encoder off only for raw-token ablations."
        )

    config = checkpoint_config(checkpoint, args)
    classifier, config = build_classifier_from_checkpoint(args, checkpoint, device, config)
    projector = MultiModalProjector(
        feature_dim=config["feature_dim"],
        projection_dim=config["projection_dim"],
    ).to(device)
    token_projector = MultiModalTokenProjector(
        feature_dim=config["feature_dim"],
        token_dim=config["token_dim"],
    ).to(device)
    clean_gen_encoder = build_clean_gen_encoder(config).to(device) if args.use_clean_gen_encoder else None

    for model, state_dict in zip(models, checkpoint["models"]):
        model.load_state_dict(state_dict)
    projector.load_state_dict(checkpoint["projector"])
    token_projector.linear_proj.load_state_dict(checkpoint["token_projector"])
    if clean_gen_encoder is not None:
        clean_gen_encoder.load_state_dict(checkpoint["clean_gen_encoder"])
    classifier.load_state_dict(checkpoint["classifier"])
    return checkpoint, projector, token_projector, clean_gen_encoder, classifier, config


def encode_tokens(clean_gen_encoder, raw_tokens):
    if clean_gen_encoder is None:
        return raw_tokens
    return clean_gen_encoder(raw_tokens)


def module_list(models, projector, token_projector, clean_gen_encoder, classifier=None):
    modules = [*models, projector]
    if token_projector is not None:
        modules.append(token_projector)
    if clean_gen_encoder is not None:
        modules.append(clean_gen_encoder)
    if classifier is not None:
        modules.append(classifier)
    return modules


def build_optimizer(args, models, projector, token_projector, clean_gen_encoder, classifier):
    pretrained_params = []
    for module in module_list(models, projector, token_projector, clean_gen_encoder):
        pretrained_params.extend(module.parameters())
    return torch.optim.AdamW(
        [
            {"params": pretrained_params, "lr": args.lr},
            {"params": classifier.parameters(), "lr": args.head_lr},
        ],
        weight_decay=args.weight_decay,
    )


def forward_logits(
    args,
    models,
    projector,
    token_projector,
    clean_gen_encoder,
    classifier,
    samples_wifi,
    samples_rfid,
    samples_mmwave,
):
    model_zs, wifi_map, rfid_map, mmwave_map = forward_projected_inputs(
        models, projector, samples_wifi, samples_rfid, samples_mmwave
    )
    if getattr(args, "legacy_featuremap_iin", False):
        return classifier(model_zs, wifi_map, rfid_map, mmwave_map, return_g=args.return_g)

    raw_tokens = token_projector(wifi_map, rfid_map, mmwave_map)
    encoded_tokens = encode_tokens(clean_gen_encoder, raw_tokens)
    return classifier(model_zs, encoded_tokens, return_g=args.return_g)


def train_one_epoch(
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
    set_mode(module_list(models, projector, token_projector, clean_gen_encoder, classifier), True)

    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    gate_meter = new_gate_meter()

    for samples_wifi, samples_rfid, samples_mmwave, labels in train_loader:
        samples_wifi = samples_wifi.to(device, non_blocking=True)
        samples_rfid = samples_rfid.to(device, non_blocking=True)
        samples_mmwave = samples_mmwave.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits, gate = split_logits_gate(
            forward_logits(
                args,
                models,
                projector,
                token_projector,
                clean_gen_encoder,
                classifier,
                samples_wifi,
                samples_rfid,
                samples_mmwave,
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

    return total_loss / total_samples, 100.0 * total_correct / total_samples, gate_stats(gate_meter)


@torch.no_grad()
def evaluate(
    args,
    device,
    data_loader,
    models,
    projector,
    token_projector,
    clean_gen_encoder,
    classifier,
    criterion,
):
    set_mode(module_list(models, projector, token_projector, clean_gen_encoder, classifier), False)
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    gate_meter = new_gate_meter()

    for samples_wifi, samples_rfid, samples_mmwave, labels in data_loader:
        samples_wifi = samples_wifi.to(device, non_blocking=True)
        samples_rfid = samples_rfid.to(device, non_blocking=True)
        samples_mmwave = samples_mmwave.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits, gate = split_logits_gate(
            forward_logits(
                args,
                models,
                projector,
                token_projector,
                clean_gen_encoder,
                classifier,
                samples_wifi,
                samples_rfid,
                samples_mmwave,
            )
        )
        loss = criterion(logits, labels)

        batch_size = labels.size(0)
        total_samples += batch_size
        total_loss += loss.item() * batch_size
        total_correct += logits.argmax(dim=1).eq(labels).sum().item()
        update_gate_meter(gate_meter, gate)

    return total_loss / total_samples, 100.0 * total_correct / total_samples, gate_stats(gate_meter)


def parse_args():
    parser = argparse.ArgumentParser(description="Cross-scene zero/few-shot evaluation for sepdual z-guided IIN")
    parser.add_argument("--ckpt", type=str, required=True, help="finetune_sepdual checkpoint")
    parser.add_argument("--raw_root", type=str, default="~/XRF55-repo/dataset/Raw_dataset")
    parser.add_argument("--split_root", type=str, default="~/XRF55-repo/dataset/XRF_dataset")
    parser.add_argument("--scene", type=str, default="Scene2")
    parser.add_argument("--strategy", type=str, default="zeroshot", choices=("zeroshot", "oneshot", "twoshot"))
    parser.add_argument("--shot_dir", type=str, default="", help="Override one_shot/two_shot directory name")
    parser.add_argument("--split_dir", type=str, default="", help="Direct path to a split dir with train_data/test_data")
    parser.add_argument("--eval_split_dir", type=str, default="", help="For zeroshot: evaluate a split dir's test_data")
    parser.add_argument("--txt_name", type=str, default="dml")
    parser.add_argument("--adapt_mode", choices=("full",), default="full", help=argparse.SUPPRESS)
    parser.add_argument("--adapt_epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=12)
    parser.add_argument("--lr", type=float, default=1.5e-4)
    parser.add_argument("--head_lr", type=float, default=1e-4)
    parser.add_argument("--min_lr", type=float, default=0.0)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--label_smoothing", "--labelsmooth", dest="label_smoothing", type=float, default=0.0)
    parser.add_argument("--feature_dim", type=int, default=None)
    parser.add_argument("--projection_dim", type=int, default=None)
    parser.add_argument("--iin_dim", type=int, default=None)
    parser.add_argument("--token_dim", type=int, default=None)
    parser.add_argument("--num_tokens_per_modality", type=int, default=None)
    parser.add_argument("--encoder_depth", type=int, default=None)
    parser.add_argument("--encoder_heads", type=int, default=None)
    parser.add_argument("--encoder_mlp_ratio", type=float, default=None)
    parser.add_argument("--drop_rate", type=float, default=None)
    parser.add_argument("--z_hidden_dim", type=int, default=None)
    parser.add_argument("--sharing_num", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--cls_gamma_init", type=float, default=None)
    parser.add_argument("--clean_gen_encoder", type=str, default="auto", choices=("auto", "on", "off"))
    parser.add_argument("--add_iin_modality_embed", action="store_true")
    parser.add_argument("--return_g", action="store_true")
    parser.add_argument("--log_interval", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--no_pin_memory", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    args.pin_memory = not args.no_pin_memory
    set_seed(args.seed)

    requested_device = torch.device(args.device)
    if requested_device.type == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = requested_device

    models = build_models(device)
    checkpoint, projector, token_projector, clean_gen_encoder, classifier, config = load_classification_checkpoint(
        args.ckpt,
        device,
        models,
        args,
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing).to(device)

    if args.legacy_featuremap_iin:
        token_path = "legacy: featuremap -> classifier.linear_proj -> z_guided_iin"
    elif args.use_clean_gen_encoder:
        token_path = "featuremap -> token_projector -> sepdual type/pos -> shared source_encoder(separate modalities) -> z_guided_iin"
    else:
        token_path = "featuremap -> token_projector(raw tokens) -> z_guided_iin"
    print(f"loaded checkpoint: {args.ckpt}")
    print(f"checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"classifier type: {checkpoint.get('classifier_type', 'unknown')}")
    print(f"classifier config: {config}")
    print(f"legacy featuremap iin: {args.legacy_featuremap_iin}")
    print(f"use sepdual encoder: {args.use_clean_gen_encoder}")
    print(f"token path: {token_path}")
    print(f"scene: {args.scene}")
    print(f"strategy: {args.strategy}")
    print(f"device: {device}")
    print(f"label smoothing: {args.label_smoothing}")

    if args.strategy == "zeroshot":
        test_dataset = build_zero_shot_dataset(args)
        test_loader = make_loader(test_dataset, args, shuffle=False)
        test_loss, test_acc, test_gate = evaluate(
            args,
            device,
            test_loader,
            models,
            projector,
            token_projector,
            clean_gen_encoder,
            classifier,
            criterion,
        )
        print(f"test samples: {len(test_dataset)}")
        print(f"test_loss {test_loss:.4f} test_acc {test_acc:.2f}{gate_text(test_gate)}")
        return

    train_dataset, test_dataset = build_few_shot_datasets(args)
    train_loader = make_loader(train_dataset, args, shuffle=True)
    test_loader = make_loader(test_dataset, args, shuffle=False)
    optimizer = build_optimizer(args, models, projector, token_projector, clean_gen_encoder, classifier)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.adapt_epochs,
        eta_min=args.min_lr,
    )

    initial_test_loss, initial_test_acc, initial_test_gate = evaluate(
        args,
        device,
        test_loader,
        models,
        projector,
        token_projector,
        clean_gen_encoder,
        classifier,
        criterion,
    )
    print(f"adapt mode: {args.adapt_mode}")
    print(f"train/support samples: {len(train_dataset)}")
    print(f"test/query samples: {len(test_dataset)}")
    print(
        f"before_adapt test_loss {initial_test_loss:.4f} "
        f"test_acc {initial_test_acc:.2f}{gate_text(initial_test_gate)}"
    )

    best_test_acc = initial_test_acc
    best_epoch = -1
    for epoch in range(args.adapt_epochs):
        train_loss, train_acc, train_gate = train_one_epoch(
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
        test_loss, test_acc, test_gate = evaluate(
            args,
            device,
            test_loader,
            models,
            projector,
            token_projector,
            clean_gen_encoder,
            classifier,
            criterion,
        )
        scheduler.step()
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_epoch = epoch

        if args.log_interval > 0 and (epoch + 1) % args.log_interval == 0:
            print(
                f"epoch {epoch:03d} done "
                f"train_loss {train_loss:.4f} train_acc {train_acc:.2f} "
                f"test_loss {test_loss:.4f} test_acc {test_acc:.2f} "
                f"train{gate_text(train_gate)} "
                f"test{gate_text(test_gate)} "
                f"cls_gamma {classifier.cls_gamma.item():.4f} "
                f"lr {optimizer.param_groups[0]['lr']:.6g}"
            )

    final_test_loss, final_test_acc, final_test_gate = evaluate(
        args,
        device,
        test_loader,
        models,
        projector,
        token_projector,
        clean_gen_encoder,
        classifier,
        criterion,
    )
    print(f"best_epoch {best_epoch} best_test_acc {best_test_acc:.2f}")
    print(f"final_test_loss {final_test_loss:.4f} final_test_acc {final_test_acc:.2f}{gate_text(final_test_gate)}")


if __name__ == "__main__":
    main()

