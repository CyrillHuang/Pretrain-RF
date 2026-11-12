import csv, random, re
from pathlib import Path
import numpy as np
import torch
from scipy.io import loadmat
from torch.utils.data import Dataset

FRAME_RE = re.compile(r"frame(\d+)", re.I)

def _frame_id(path):
    m = FRAME_RE.search(path.stem)
    if m is None: raise ValueError(f"cannot parse frame id: {path}")
    return int(m.group(1))

def read_mmwave(path):
    x = np.fromfile(path, dtype=np.float64)
    if x.size % 5: raise ValueError(f"invalid mmWave frame: {path}")
    return np.nan_to_num(x.reshape(-1, 5).astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

def read_wifi_amp(path):
    x = np.asarray(loadmat(path)["CSIamp"], dtype=np.float32)
    if x.shape != (3, 114, 10): raise ValueError(f"unexpected CSIamp shape {x.shape} in {path}")
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    lo, hi = float(x.min()), float(x.max())
    return ((x - lo) / (hi - lo)).astype(np.float32) if hi > lo else np.zeros_like(x)

def read_segment_rows(csv_path):
    rows = []
    with Path(csv_path).expanduser().open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            segs = []
            for text in row["Segments"].split(";"):
                text = text.strip()
                if text:
                    a, b = text.split("-", 1); segs.append((int(a), int(b)))
            rows.append((row["Environment"], row["Student"], row["Action"], segs))
    return rows

class MMFiWindowDataset(Dataset):
    """CSV segment samples with stratified 70/30 train/test splitting."""
    def __init__(self, root="~/MMFi/MMFi_Dataset", segments_csv="~/MMFi/MMFi_action_segments.csv",
                 environments=None, students=None, actions=None, window_size=32, stride=16,
                 max_points=128, split=None, train_ratio=0.7, seed=42, drop_last=False):
        self.root = Path(root).expanduser(); self.window_size = int(window_size)
        self.stride = int(stride); self.max_points = int(max_points)
        if split not in (None, "all", "train", "test"): raise ValueError("split must be None, all, train, or test")
        envs, studs, acts = map(lambda x: None if x is None else set(x), (environments, students, actions))
        candidates = []
        for env, student, action, segments in read_segment_rows(segments_csv):
            if (envs is not None and env not in envs) or (studs is not None and student not in studs) or (acts is not None and action not in acts): continue
            base = self.root / env / student / action
            mm = {_frame_id(p): p for p in (base / "mmwave").glob("frame*.bin")}
            wi = {_frame_id(p): p for p in (base / "wifi-csi").glob("frame*.mat")}
            common = set(mm) & set(wi)
            for seg_id, (start, end) in enumerate(segments):
                frames = sorted(i for i in common if start <= i <= end)
                if frames: candidates.append({"env":env,"student":student,"action":action,"segment":seg_id,"frames":frames,"mm":mm,"wi":wi})
        if split in ("train", "test"):
            groups = {}
            for item in candidates: groups.setdefault(item["action"], []).append(item)
            rng = random.Random(seed); selected = []
            for action in sorted(groups):
                group = groups[action]; rng.shuffle(group)
                cut = min(max(int(len(group) * train_ratio), 1), len(group)-1)
                selected.extend(group[:cut] if split == "train" else group[cut:])
            candidates = selected
        if not candidates: raise RuntimeError("No MM-Fi segments found for requested filters/split")
        self.samples = candidates

    def __len__(self): return len(self.samples)

    def __getitem__(self, index):
        item = self.samples[index]; frames = item["frames"]
        if len(frames) > self.window_size:
            start = (len(frames)-self.window_size)//2; frames = frames[start:start+self.window_size]
        wifi = np.zeros((self.window_size,3,114,10), dtype=np.float32)
        mmwave = np.zeros((self.window_size,self.max_points,5), dtype=np.float32)
        for t, frame in enumerate(frames):
            wifi[t] = read_wifi_amp(item["wi"][frame]); points = read_mmwave(item["mm"][frame])
            mmwave[t,:min(len(points),self.max_points)] = points[:self.max_points]
        meta = {"environment":item["env"],"student":item["student"],"action":item["action"],"segment":item["segment"],"valid_frames":len(frames)}
        return torch.from_numpy(wifi), torch.from_numpy(mmwave), torch.tensor(int(item["action"][1:])-1), meta

if __name__ == "__main__":
    ds = MMFiWindowDataset(); x = ds[0]
    print("samples:",len(ds),"wifi:",tuple(x[0].shape),"mmwave:",tuple(x[1].shape),"label:",int(x[2]))


