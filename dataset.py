import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from collections import defaultdict

from canon_v import canonicalize_json


TARGET = {0: 1, 1: 6, 2: 3, 3: 5, 4: 4, 5: 4, 6: 2}
LEVELS_IN_ORDER = [0, 1, 2, 3, 4, 5, 6]

J = sum(TARGET.values())
K = J * 3


def build_role_slots(canon_data):
    signature = canon_data["signature"]
    level_map = defaultdict(list)

    for row in signature:
        jid = row[0]
        lvl = row[1]
        degree = row[2]
        subtree = row[3]
        level_map[lvl].append((jid, subtree, degree))

    slots = []

    for lvl in LEVELS_IN_ORDER:
        joints = level_map.get(lvl, [])
        quota = TARGET[lvl]

        if len(joints) == 0:
            continue

        joints_sorted = sorted(
            joints,
            key=lambda x: (-x[1], -x[2])
        )

        keep = joints_sorted[:quota]
        slots.extend([j[0] for j in keep])

    if len(slots) < J:
        return None, None

    slots = slots[:J]
    root = slots[0]

    return slots, root


def load_cam3d(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)

    cam3d = data.get("points", {}).get("cam_3d", None)

    if cam3d is None:
        return None

    arr = np.asarray(cam3d, dtype=np.float32)

    if arr.ndim != 3 or arr.shape[-1] != 3:
        return None

    return arr


def normalize_to_slots(x_tj3, slots, root_jid):
    T, Jraw, _ = x_tj3.shape

    if max(slots) >= Jraw or root_jid >= Jraw:
        return None

    x = x_tj3[:, slots, :]
    root = x_tj3[:, root_jid, :].reshape(T, 1, 3)

    return x - root


def gather_jsons_recursive(root_dir):
    out = []

    for r, _, files in os.walk(root_dir):
        for f in files:
            if f.endswith(".json"):
                out.append(os.path.join(r, f))

    return sorted(out)


class WindowMaskDataset(Dataset):

    def __init__(self, root_dir, in_len=25, out_len=25):
        self.samples = []

        L = in_len + out_len
        json_paths = gather_jsons_recursive(root_dir)

        for jp in json_paths:
            try:
                canon_data = canonicalize_json(jp)
                slots, root = build_role_slots(canon_data)

                if slots is None:
                    continue

                x = load_cam3d(jp)

                if x is None:
                    continue

                x = normalize_to_slots(x, slots, root)

                if x is None:
                    continue

                T = x.shape[0]

                if T < L:
                    continue

                x = x.reshape(T, K)

                json_name = os.path.splitext(os.path.basename(jp))[0]

                for s in range(T - L + 1):
                    win = x[s:s + L]

                    pose = win.astype(np.float32)

                    mask = np.zeros((L, K), dtype=np.float32)
                    mask[:in_len, :] = 1.0

                    tp = np.arange(L, dtype=np.float32)

                    self.samples.append({
                        "pose": pose,
                        "mask": mask,
                        "timepoints": tp,
                        "json_path": jp,
                        "json_name": json_name,
                        "window_start": s,
                    })

            except Exception as e:
                print("Skipping:", jp)
                print("Error:", repr(e))
                continue

        print(f"[dataset] windows = {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]

        return {
            "pose": torch.from_numpy(item["pose"]),
            "mask": torch.from_numpy(item["mask"]),
            "timepoints": torch.from_numpy(item["timepoints"]),
            "json_path": item["json_path"],
            "json_name": item["json_name"],
            "window_start": item["window_start"],
        }