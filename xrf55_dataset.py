import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data.dataset import Dataset


log = logging.getLogger(__name__)


class XRFMultiModalDataset(Dataset):
    def __init__(self, file_path="~/XRF55-repo/dataset/XRF_dataset", is_train=True, scene="dml", split=None):
        super(XRFMultiModalDataset, self).__init__()
        self.file_path = Path(file_path).expanduser()
        self.is_train = is_train
        self.scene = scene
        split = split or ("train" if is_train else "val")
        if split not in ("train", "val", "all"):
            raise ValueError("split must be 'train', 'val', or 'all'")
        split_names = ("train", "val") if split == "all" else (split,)
        self.files = [self.file_path / f"{self.scene}_{name}.txt" for name in split_names]

        self.data = {
            "file_name": [],
            "label": [],
            "is_train": [],
        }
        for split_name, file in zip(split_names, self.files):
            with file.open("r", encoding="utf-8") as f:
                for line in f:
                    file_name, _, action_id = line.strip().split(",")
                    self.data["file_name"].append(file_name)
                    self.data["label"].append(int(action_id) - 1)
                    self.data["is_train"].append(split_name == "train")

        log.info("load XRF dataset from %s", ", ".join(str(file) for file in self.files))

    def __len__(self):
        return len(self.data["label"])

    def __getitem__(self, idx):
        file_name = self.data["file_name"][idx]
        label = self.data["label"][idx]
        is_train = self.data["is_train"][idx]

        wifi_data = load_wifi(file_name, is_train, path=self.file_path)
        rfid_data = load_rfid(file_name, is_train, path=self.file_path)
        mmwave_data = load_mmwave(file_name, is_train, path=self.file_path)

        return wifi_data, rfid_data, mmwave_data, label


def _split_dir(is_train):
    return "train_data" if is_train else "test_data"


def _to_channels_first_1d(record, channels, time_steps):
    if record.shape == (time_steps, channels):
        record = record.transpose(1, 0)
    if record.shape != (channels, time_steps):
        raise ValueError(
            f"expected shape {(channels, time_steps)} or {(time_steps, channels)}, got {record.shape}"
        )
    return np.ascontiguousarray(record)


def load_rfid(filename, is_train, path="~/XRF55-repo/dataset/XRF_dataset"):
    root = Path(path).expanduser() / _split_dir(is_train)
    record = np.load(root / "RFID" / f"{filename}.npy")
    record = _to_channels_first_1d(record, channels=23, time_steps=148)
    return torch.from_numpy(record).float()


def load_wifi(filename, is_train, path="~/XRF55-repo/dataset/XRF_dataset"):
    root = Path(path).expanduser() / _split_dir(is_train)
    record = np.load(root / "WiFi" / f"{filename}.npy")
    record = _to_channels_first_1d(record, channels=270, time_steps=1000)
    return torch.from_numpy(record).float()


def load_mmwave(filename, is_train, path="~/XRF55-repo/dataset/XRF_dataset"):
    root = Path(path).expanduser() / _split_dir(is_train)
    record = np.load(root / "mmWave" / f"{filename}.npy")
    if record.shape == (1, 17, 256, 128):
        record = record.squeeze(0)
    if record.shape == (256, 128, 17):
        record = record.transpose(2, 0, 1)
    if record.shape != (17, 256, 128):
        raise ValueError(f"expected shape {(17, 256, 128)} or {(256, 128, 17)}, got {record.shape}")
    return torch.from_numpy(np.ascontiguousarray(record)).float()


