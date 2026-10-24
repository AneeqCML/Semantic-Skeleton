import os
import json
import hashlib
from collections import defaultdict, deque

import numpy as np
import torch
from torch.utils.data import Dataset

from canon_v import canonicalize_json


TARGET = {0: 1, 1: 6, 2: 3, 3: 5, 4: 4, 5: 4, 6: 2}
LEVELS_IN_ORDER = [0, 1, 2, 3, 4, 5, 6]

J = sum(TARGET.values())
K = J * 3


def stable_int_seed(*parts, base_seed=0):
    s = "::".join(str(p) for p in parts)
    h = hashlib.md5(s.encode("utf-8")).hexdigest()
    return (int(h[:8], 16) + int(base_seed)) % (2**32 - 1)


def rng_for(*parts, base_seed=0):
    return np.random.default_rng(stable_int_seed(*parts, base_seed=base_seed))


# ============================================================
# Basic file / JSON helpers
# ============================================================

def gather_jsons_recursive(root_dir):
    out = []

    for r, _, files in os.walk(root_dir):
        for f in files:
            if f.endswith(".json"):
                out.append(os.path.join(r, f))

    return sorted(out)


def detect_animal(json_path):
    return os.path.basename(os.path.dirname(json_path))


def load_json(json_path):
    with open(json_path, "r") as f:
        return json.load(f)


def extract_cam3d_from_data(data):
    """
    Supports both:
      data["points"]["cam_3d"]
      data["cam_3d"]

    Returns np.float32 array with shape (T, J, 3), or None.
    """

    cam3d = None

    if "points" in data and isinstance(data["points"], dict):
        cam3d = data["points"].get("cam_3d", None)

    if cam3d is None:
        cam3d = data.get("cam_3d", None)

    if cam3d is None:
        return None

    arr = np.asarray(cam3d, dtype=np.float32)

    if arr.ndim == 2 and arr.shape[-1] == 3:
        arr = arr[None, :, :]

    if arr.ndim != 3 or arr.shape[-1] != 3:
        return None

    return arr


def load_cam3d(json_path):
    data = load_json(json_path)
    return extract_cam3d_from_data(data)


# ============================================================
# Edge helpers for eval metrics only
# ============================================================

FALLBACK_EDGES_25 = [
    [2, 0], [3, 0], [4, 0], [5, 0], [6, 0], [1, 0],
    [1, 8], [1, 9],
    [8, 12], [12, 17], [17, 21],
    [9, 13], [13, 18], [18, 22],
    [1, 7],
    [7, 10], [7, 11],
    [10, 15], [15, 19], [19, 23],
    [11, 16], [16, 20], [20, 24],
    [7, 14],
]


def extract_connections_from_data(data):
    """
    Supports common edge locations:
      data["connections"]
      data["skeleton"]
      data["edges"]
      data["points"]["connections"]

    Returns raw-joint edges as list[[a, b], ...].
    """

    connections = data.get("connections", None)

    if connections is None:
        connections = data.get("skeleton", None)

    if connections is None:
        connections = data.get("edges", None)

    if connections is None and "points" in data and isinstance(data["points"], dict):
        connections = data["points"].get("connections", None)

    if connections is None:
        return []

    out = []

    for e in connections:
        if not isinstance(e, (list, tuple)) or len(e) != 2:
            continue

        try:
            a = int(e[0])
            b = int(e[1])
        except Exception:
            continue

        if a != b:
            out.append([a, b])

    return out


def load_connections(json_path):
    data = load_json(json_path)
    return extract_connections_from_data(data)


def unique_edges(edges):
    seen = set()
    out = []

    for e in edges:
        if len(e) != 2:
            continue

        a, b = int(e[0]), int(e[1])

        if a == b:
            continue

        key = tuple(sorted((a, b)))

        if key in seen:
            continue

        seen.add(key)
        out.append([a, b])

    return out


def remap_edges_from_slots(raw_connections, slots, *, fallback=True):
    """
    Convert raw JSON edges into final 25-slot coordinates.

    slots means:
      final slot index -> joint id in the same index space as raw_connections

    Example:
      slots[0] = 24 means joint 24 is now final joint 0.

    Returns:
      np.int64 array [E, 2], with edge indices in 0..24.
    """

    if slots is None or len(slots) != J:
        if fallback:
            return np.asarray(FALLBACK_EDGES_25, dtype=np.int64)
        return np.zeros((0, 2), dtype=np.int64)

    raw_to_slot = {
        int(raw_jid): int(slot_idx)
        for slot_idx, raw_jid in enumerate(slots)
    }

    edges = []

    for e in raw_connections:
        if len(e) != 2:
            continue

        a_raw = int(e[0])
        b_raw = int(e[1])

        if a_raw not in raw_to_slot or b_raw not in raw_to_slot:
            continue

        a_new = raw_to_slot[a_raw]
        b_new = raw_to_slot[b_raw]

        if a_new != b_new:
            edges.append([a_new, b_new])

    edges = unique_edges(edges)

    if len(edges) == 0 and fallback:
        edges = FALLBACK_EDGES_25

    return np.asarray(edges, dtype=np.int64)


def reconstruct_edges_for_result(json_path, slots):
    """
    Reconstruct final 25-joint edges for:
      canonical slots
      random_perm_drop kept_raw slots

    Edges follow the same joint mapping as pose_tj3.
    """

    raw_connections = load_connections(json_path)

    return remap_edges_from_slots(
        raw_connections,
        slots,
        fallback=True,
    )


def build_role_slots(
    canon_data,
    *,
    random_within_levels=False,
    seed=0,
    json_path=None,
):
    """
    OG canonical slot builder.

    canon_v.py signature rows are:
        (jid, lvl, degree, subtree, mld, fb, lr)

    The old loader used only:
        jid, lvl, degree, subtree

    Default canonical:
        group by level
        sort each level by subtree desc, degree desc
        take TARGET quota
        root = slots[0]

    random_within_levels:
        group by level
        randomly choose TARGET quota inside each level
        root still comes from level 0
    """

    signature = canon_data["signature"]
    level_map = defaultdict(list)

    for row in signature:
        jid = int(row[0])
        lvl = int(row[1])
        degree = int(row[2])
        subtree = int(row[3])

        level_map[lvl].append((jid, subtree, degree))

    slots = []

    r = rng_for(
        "canonical_random_within_levels",
        json_path or "",
        base_seed=seed,
    )

    for lvl in LEVELS_IN_ORDER:
        joints = list(level_map.get(lvl, []))
        quota = TARGET[lvl]

        if len(joints) == 0:
            continue

        if random_within_levels:
            if len(joints) <= quota:
                keep = joints
            else:
                idx = r.permutation(len(joints))[:quota]
                keep = [joints[i] for i in idx]

            # OG behavior: sort selected joints by current joint id.
            keep = sorted(keep, key=lambda x: x[0])

        else:
            # OG behavior: subtree desc, degree desc.
            # No explicit joint-id tie-breaker.
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


def normalize_to_slots(x_tj3, slots, root_jid, *, root_norm=True):
    T, Jraw, _ = x_tj3.shape

    if len(slots) != J:
        return None

    if max(slots) >= Jraw or root_jid >= Jraw:
        return None

    x = x_tj3[:, slots, :].astype(np.float32)

    if root_norm:
        root = x_tj3[:, root_jid, :].reshape(T, 1, 3)
        x = x - root

    return x.astype(np.float32)


def canonicalize_og_loader_style(
    json_path,
    *,
    root_norm=True,
    random_within_levels=False,
    seed=0,
):
    """
    Canonical modes:

      canonical
      canonical + no_root_norm
      canonical + random_within_levels
      canonical + random_within_levels + no_root_norm

    Uses canon_v.py directly on the raw JSON.
    No animal-specific permutation.
    """

    canon_data = canonicalize_json(json_path)

    slots, root = build_role_slots(
        canon_data,
        random_within_levels=random_within_levels,
        seed=seed,
        json_path=json_path,
    )

    if slots is None:
        return None

    x_raw = load_cam3d(json_path)

    if x_raw is None:
        return None

    x = normalize_to_slots(
        x_raw,
        slots,
        root,
        root_norm=root_norm,
    )

    if x is None:
        return None

    edges = reconstruct_edges_for_result(json_path, slots)

    mode = "canonical"

    if random_within_levels:
        mode += "_random_within_levels"

    if not root_norm:
        mode += "_no_root_norm"

    return {
        "pose_tj3": x.astype(np.float32),
        "slots": slots,
        "root": root,
        "mode": mode,
        "root_norm": bool(root_norm),
        "edges": edges,
    }


# ============================================================
# Random permute-drop mode
# ============================================================

def build_random_perm_0based(
    raw_joint_count,
    *,
    animal,
    json_path,
    seed=0,
    scope="animal",
):
    """
    Random baseline:

      raw joints
      -> deterministic random order over ALL raw joints
      -> keep first 25
      -> drop everything beyond 25

    scope="animal":
        same random order for all files from same animal with same raw_joint_count

    scope="file":
        different deterministic random order per JSON file
    """

    if raw_joint_count < J:
        return None

    if scope == "animal":
        r = rng_for(
            "random_perm_drop_animal",
            animal,
            raw_joint_count,
            base_seed=seed,
        )
    elif scope == "file":
        r = rng_for(
            "random_perm_drop_file",
            animal,
            json_path,
            raw_joint_count,
            base_seed=seed,
        )
    else:
        raise ValueError(f"Unknown random_perm_scope: {scope}")

    return list(r.permutation(raw_joint_count).astype(int))


def random_perm_drop(
    json_path,
    *,
    root_norm=True,
    seed=0,
    random_perm_scope="animal",
):
    """
    Random permute-drop modes:

      random_perm_drop
      random_perm_drop + no_root_norm

    No canonicalization.
    No animal-specific permutation table.

    Root normalization:
      root = joint 0 after random keep/drop.
      So after random ordering, the first kept joint is treated as root.
    """

    animal = detect_animal(json_path)
    x_raw = load_cam3d(json_path)

    if x_raw is None:
        return None

    T, raw_joint_count, _ = x_raw.shape

    perm = build_random_perm_0based(
        raw_joint_count,
        animal=animal,
        json_path=json_path,
        seed=seed,
        scope=random_perm_scope,
    )

    if perm is None:
        return None

    kept_raw = perm[:J]

    if len(kept_raw) != J:
        return None

    x = x_raw[:, kept_raw, :].astype(np.float32)

    root = 0
    root_raw = kept_raw[root]

    if root_norm:
        root_xyz = x[:, root, :].reshape(T, 1, 3)
        x = x - root_xyz

    edges = reconstruct_edges_for_result(json_path, kept_raw)

    mode = "random_perm_drop"

    if not root_norm:
        mode += "_no_root_norm"

    return {
        "pose_tj3": x.astype(np.float32),
        "slots": kept_raw,
        "root": root,
        "root_raw": root_raw,
        "mode": mode,
        "root_norm": bool(root_norm),
        "edges": edges,
    }



def build_graph(connections, joint_count):
    graph = defaultdict(list)

    for e in connections:
        if len(e) != 2:
            continue

        a = int(e[0])
        b = int(e[1])

        if a < 0 or b < 0:
            continue

        if a >= joint_count or b >= joint_count:
            continue

        if a == b:
            continue

        graph[a].append(b)
        graph[b].append(a)

    return graph


def compute_subtree_sizes(root, children):
    subtree = {}

    def dfs(u):
        total = 1

        for v in children[u]:
            total += dfs(v)

        subtree[u] = total
        return total

    dfs(root)
    return subtree


def choose_raw_root_from_graph(raw_connections, raw_joint_count):
    """
    Pick the same style of root as canon_v.py:
      highest degree hub, smallest raw id as tie-breaker.

    This avoids changing the root just because joint labels are permuted.
    """

    graph = build_graph(raw_connections, raw_joint_count)

    if raw_joint_count <= 0:
        return None

    return max(range(raw_joint_count), key=lambda j: (len(graph[j]), -j))


def permute_points_full_from_data(data, perm):
    """
    Applies a full permutation to cam_3d, 3d, and 2d if present.

    perm means:
      perm[new_index] = old_raw_index
    """

    out = json.loads(json.dumps(data))

    if "points" in out and isinstance(out["points"], dict):
        for key in ["cam_3d", "3d", "2d"]:
            if key not in out["points"]:
                continue

            arr = np.asarray(out["points"][key], dtype=np.float32)

            if arr.ndim != 3:
                continue

            if arr.shape[1] <= max(perm):
                continue

            out["points"][key] = arr[:, perm, :].tolist()

    if "cam_3d" in out:
        arr = np.asarray(out["cam_3d"], dtype=np.float32)

        if arr.ndim == 3 and arr.shape[1] > max(perm):
            out["cam_3d"] = arr[:, perm, :].tolist()

        elif arr.ndim == 2 and arr.shape[-1] == 3 and arr.shape[0] > max(perm):
            out["cam_3d"] = arr[perm, :].tolist()

    return out


def remap_connections_full_random(connections, perm):
    """
    Remap raw connection indices into random-permuted index space.

    perm[new_index] = old_raw_index

    Therefore:
      old_to_new[old_raw_index] = new_index
    """

    old_to_new = {
        int(old): int(new)
        for new, old in enumerate(perm)
    }

    remapped = []

    for e in connections:
        if len(e) != 2:
            continue

        a = int(e[0])
        b = int(e[1])

        if a not in old_to_new or b not in old_to_new:
            continue

        na = old_to_new[a]
        nb = old_to_new[b]

        if na == nb:
            continue

        remapped.append([na, nb])

    return unique_edges(remapped)


def make_full_random_permuted_data(
    json_path,
    *,
    seed=0,
    random_perm_scope="animal",
):
    """
    Creates a full randomly-permuted JSON-like dict.

    Important:
      - keeps ALL raw joints
      - remaps connections
      - stores _perm_0based
      - stores _root_permuted_idx
      - does NOT drop before canonicalization

    perm means:
      perm[permuted_index] = original_raw_joint_id
    """

    animal = detect_animal(json_path)
    data = load_json(json_path)
    x_raw = extract_cam3d_from_data(data)

    if x_raw is None:
        return None

    T, raw_joint_count, _ = x_raw.shape

    perm = build_random_perm_0based(
        raw_joint_count,
        animal=animal,
        json_path=json_path,
        seed=seed,
        scope=random_perm_scope,
    )

    if perm is None:
        return None

    perm = list(map(int, perm))

    raw_connections = extract_connections_from_data(data)

    root_raw = choose_raw_root_from_graph(
        raw_connections,
        raw_joint_count,
    )

    if root_raw is None:
        return None

    if root_raw not in perm:
        return None

    root_permuted_idx = int(perm.index(root_raw))

    out = permute_points_full_from_data(data, perm)

    remapped_connections = remap_connections_full_random(
        raw_connections,
        perm,
    )

    out["connections"] = remapped_connections

    if "points" in out and isinstance(out["points"], dict):
        if "connections" in out["points"]:
            out["points"]["connections"] = remapped_connections

    out["_animal"] = animal
    out["_perm_0based"] = perm
    out["_root_permuted_idx"] = root_permuted_idx
    out["_root_raw_0based"] = int(root_raw)
    out["_mode"] = "full_random_permutation"

    return out


def canonicalize_full_permuted_data(
    data,
    *,
    root_norm=True,
    random_within_levels=False,
    seed=0,
    json_path=None,
):
    """
    Canonicalizes a FULL permuted skeleton.

    Pipeline:
      full permuted data
      -> build graph
      -> BFS from stored root
      -> group by level
      -> select TARGET slots
      -> optional root norm

    Important:
      This intentionally matches OG canonical slot tie-breaking:

        default:
          sort by subtree desc, degree desc

        random_within_levels:
          randomly choose within level, then sort by current joint id

      No original-raw-id tie-breaker is used here.
    """

    cam3d = extract_cam3d_from_data(data)

    if cam3d is None:
        return None

    T, joint_count, _ = cam3d.shape

    if joint_count < J:
        return None

    connections = extract_connections_from_data(data)
    graph = build_graph(connections, joint_count)

    if joint_count == 0:
        return None

    root = data.get("_root_permuted_idx", None)

    if root is not None:
        root = int(root)

        if root < 0 or root >= joint_count:
            root = None

    if root is None:
        root = max(range(joint_count), key=lambda j: (len(graph[j]), -j))

    visited = set([root])
    parent = {root: -1}
    level = {root: 0}
    children = defaultdict(list)

    q = deque([root])

    while q:
        u = q.popleft()

        for v in graph[u]:
            if v in visited:
                continue

            visited.add(v)
            parent[v] = u
            level[v] = level[u] + 1
            children[u].append(v)
            q.append(v)

    if len(visited) < J:
        print(
            "[canonicalize_full_permuted_data] visited fewer than J joints:",
            "visited =", len(visited),
            "J =", J,
            "root =", root,
            "root_raw =", data.get("_root_raw_0based", None),
            "animal =", data.get("_animal", None),
        )
        return None

    subtree = compute_subtree_sizes(root, children)

    signature = []

    for jid in sorted(visited):
        lvl = int(level[jid])
        degree = int(len(graph[jid]))
        st = int(subtree.get(jid, 1))
        signature.append([jid, lvl, degree, st])

    level_map = defaultdict(list)

    for jid, lvl, degree, st in signature:
        level_map[lvl].append((jid, st, degree))

    perm = data.get("_perm_0based", None)

    def raw_id_for(permuted_jid):
        if perm is None:
            return int(permuted_jid)

        return int(perm[int(permuted_jid)])

    r = rng_for(
        "random_perm_canonical_random_within_levels",
        json_path or "",
        base_seed=seed,
    )

    slots = []

    for lvl in LEVELS_IN_ORDER:
        quota = TARGET[lvl]
        joints = list(level_map.get(lvl, []))

        if len(joints) == 0:
            continue

        if random_within_levels:
            if len(joints) <= quota:
                keep = joints
            else:
                idx = r.permutation(len(joints))[:quota]
                keep = [joints[i] for i in idx]

            # Match OG canonical behavior exactly:
            # after random selection, sort by current joint index.
            keep = sorted(keep, key=lambda x: x[0])

        else:
            # Match OG canonical behavior exactly:
            # subtree desc, degree desc.
            # No raw_id_for tie-breaker.
            joints_sorted = sorted(
                joints,
                key=lambda x: (-x[1], -x[2])
            )

            keep = joints_sorted[:quota]

        slots.extend([int(x[0]) for x in keep])

    if len(slots) != J:
        print(
            "[canonicalize_full_permuted_data] slot count mismatch:",
            "slots =", len(slots),
            "J =", J,
            "root =", root,
            "animal =", data.get("_animal", None),
        )
        return None

    x = cam3d[:, slots, :].astype(np.float32)

    if root_norm:
        root_xyz = cam3d[:, root, :].reshape(T, 1, 3)
        x = x - root_xyz

    edges = remap_edges_from_slots(
        connections,
        slots,
        fallback=True,
    )

    slots_original_raw = [
        raw_id_for(slot)
        for slot in slots
    ]

    mode = "random_perm_canonical"

    if random_within_levels:
        mode += "_random_within_levels"

    if not root_norm:
        mode += "_no_root_norm"

    return {
        "pose_tj3": x.astype(np.float32),

        # Original raw IDs of final selected canonical slots.
        "slots": slots_original_raw,

        # Permuted-space IDs of final selected slots.
        "slots_perm": slots,

        # The actual graph root is root_perm/root_raw.
        # Dataset stores root=0 for compatibility with final 25-slot pose convention.
        "root": 0,
        "root_perm": int(root),
        "root_raw": int(data.get("_root_raw_0based", -1)),

        "mode": mode,
        "root_norm": bool(root_norm),
        "edges": edges.astype(np.int64),
        "signature": signature,
        "perm_0based": perm,
    }


def random_perm_then_canonicalize(
    json_path,
    *,
    root_norm=True,
    random_within_levels=False,
    seed=0,
    random_perm_scope="animal",
):

    full_perm_data = make_full_random_permuted_data(
        json_path,
        seed=seed,
        random_perm_scope=random_perm_scope,
    )

    if full_perm_data is None:
        return None

    return canonicalize_full_permuted_data(
        full_perm_data,
        root_norm=root_norm,
        random_within_levels=random_within_levels,
        seed=seed,
        json_path=json_path,
    )


class WindowMaskDataset(Dataset):
    def __init__(
        self,
        root_dir,
        in_len=25,
        out_len=25,
        *,
        ablation_mode="canonical",
        root_norm=True,
        random_within_levels=False,
        random_perm_scope="animal",
        seed=0,
        skip_rate=1,
        return_metadata=True,
    ):
        """
        Supported ablation setups:

          canonical
          canonical + no_root_norm
          canonical + random_within_levels
          canonical + random_within_levels + no_root_norm

          random_perm_drop
          random_perm_drop + no_root_norm

          random_perm_canonical
          random_perm_canonical + no_root_norm
          random_perm_canonical + random_within_levels
          random_perm_canonical + random_within_levels + no_root_norm

        Args:
          ablation_mode:
            "canonical", "random_perm_drop", or "random_perm_canonical"

          root_norm:
            True by default.
            False gives no-root-normalization ablation.

          random_within_levels:
            Meaningful for canonical and random_perm_canonical.

          random_perm_scope:
            Meaningful for random_perm_drop and random_perm_canonical.

            "animal" = one deterministic random order per animal.
            "file" = one deterministic random order per JSON.

          seed:
            Base seed for deterministic ablations.

          skip_rate:
            Sliding-window stride.

          return_metadata:
            If False, __getitem__ returns only pose/mask/timepoints.
        """

        if ablation_mode not in ["canonical", "random_perm_drop", "random_perm_canonical"]:
            raise ValueError(
                f"Unknown ablation_mode={ablation_mode}. "
                f"Expected 'canonical', 'random_perm_drop', or 'random_perm_canonical'."
            )

        self.samples = []
        self.root_dir = root_dir
        self.in_len = int(in_len)
        self.out_len = int(out_len)
        self.ablation_mode = ablation_mode
        self.root_norm = bool(root_norm)
        self.random_within_levels = bool(random_within_levels)
        self.random_perm_scope = random_perm_scope
        self.seed = int(seed)
        self.skip_rate = max(1, int(skip_rate))
        self.return_metadata = bool(return_metadata)

        L = self.in_len + self.out_len
        json_paths = gather_jsons_recursive(root_dir)

        skipped = 0

        for jp in json_paths:
            try:
                if ablation_mode == "canonical":
                    result = canonicalize_og_loader_style(
                        jp,
                        root_norm=self.root_norm,
                        random_within_levels=self.random_within_levels,
                        seed=self.seed,
                    )

                elif ablation_mode == "random_perm_drop":
                    result = random_perm_drop(
                        jp,
                        root_norm=self.root_norm,
                        seed=self.seed,
                        random_perm_scope=self.random_perm_scope,
                    )

                elif ablation_mode == "random_perm_canonical":
                    result = random_perm_then_canonicalize(
                        jp,
                        root_norm=self.root_norm,
                        random_within_levels=self.random_within_levels,
                        seed=self.seed,
                        random_perm_scope=self.random_perm_scope,
                    )

                else:
                    raise ValueError(f"Unknown ablation_mode: {ablation_mode}")

                if result is None:
                    print("Skipping ablation fail:", jp)
                    skipped += 1
                    continue

                x = result["pose_tj3"]

                if x.shape[1] != J or x.shape[2] != 3:
                    print("Skipping bad shape:", jp, x.shape)
                    skipped += 1
                    continue

                if not np.isfinite(x).all():
                    print("Skipping non-finite pose:", jp)
                    skipped += 1
                    continue

                T = x.shape[0]

                if T < L:
                    skipped += 1
                    continue

                x = x.reshape(T, K)

                json_name = os.path.splitext(os.path.basename(jp))[0]
                animal = detect_animal(jp)

                edges = result.get(
                    "edges",
                    np.asarray(FALLBACK_EDGES_25, dtype=np.int64),
                )

                for s in range(0, T - L + 1, self.skip_rate):
                    win = x[s:s + L]
                    pose = win.astype(np.float32)

                    mask = np.zeros((L, K), dtype=np.float32)
                    mask[:self.in_len, :] = 1.0

                    tp = np.arange(L, dtype=np.float32)

                    sample = {
                        "pose": pose,
                        "mask": mask,
                        "timepoints": tp,
                        "json_path": jp,
                        "json_name": json_name,
                        "animal": animal,
                        "window_start": s,
                        "ablation_mode": result["mode"],
                        "slots": list(result["slots"]),
                        "root": int(result["root"]),
                        "edges": edges.astype(np.int64),
                    }

                    if "slots_perm" in result:
                        sample["slots_perm"] = list(result["slots_perm"])

                    if "root_perm" in result:
                        sample["root_perm"] = int(result["root_perm"])

                    if "root_raw" in result:
                        sample["root_raw"] = int(result["root_raw"])

                    self.samples.append(sample)

            except Exception as e:
                print("Skipping:", jp)
                print("Error:", repr(e))
                skipped += 1
                continue

        print(f"[dataset] mode = {ablation_mode}")
        print(f"[dataset] root_norm = {self.root_norm}")
        print(f"[dataset] random_within_levels = {self.random_within_levels}")
        print(f"[dataset] random_perm_scope = {self.random_perm_scope}")
        print(f"[dataset] seed = {self.seed}")
        print(f"[dataset] skip_rate = {self.skip_rate}")
        print(f"[dataset] return_metadata = {self.return_metadata}")
        print(f"[dataset] windows = {len(self.samples)}")
        print(f"[dataset] skipped jsons = {skipped}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]

        out = {
            "pose": torch.from_numpy(item["pose"]),
            "mask": torch.from_numpy(item["mask"]),
            "timepoints": torch.from_numpy(item["timepoints"]),
        }

        if self.return_metadata:
            out.update({
                "json_path": item["json_path"],
                "json_name": item["json_name"],
                "animal": item["animal"],
                "window_start": item["window_start"],
                "ablation_mode": item["ablation_mode"],
                "slots": torch.tensor(item["slots"], dtype=torch.long),
                "root": torch.tensor(item["root"], dtype=torch.long),
                "edges": torch.from_numpy(item["edges"]).long(),
            })

            if "slots_perm" in item:
                out["slots_perm"] = torch.tensor(item["slots_perm"], dtype=torch.long)

            if "root_perm" in item:
                out["root_perm"] = torch.tensor(item["root_perm"], dtype=torch.long)

            if "root_raw" in item:
                out["root_raw"] = torch.tensor(item["root_raw"], dtype=torch.long)

        return out



if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(
        "OG-style canonical/random-perm-drop/random-perm-canonical dataset"
    )

    p.add_argument("--data_dir", required=True)
    p.add_argument("--input_n", type=int, default=20)
    p.add_argument("--output_n", type=int, default=15)
    p.add_argument("--skip_rate", type=int, default=1)

    p.add_argument(
        "--ablation_mode",
        choices=["canonical", "random_perm_drop", "random_perm_canonical"],
        default="canonical",
    )

    p.add_argument("--no_root_norm", action="store_true")
    p.add_argument("--random_within_levels", action="store_true")
    p.add_argument(
        "--random_perm_scope",
        choices=["animal", "file"],
        default="animal",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no_metadata", action="store_true")

    args = p.parse_args()

    ds = WindowMaskDataset(
        args.data_dir,
        in_len=args.input_n,
        out_len=args.output_n,
        ablation_mode=args.ablation_mode,
        root_norm=not args.no_root_norm,
        random_within_levels=args.random_within_levels,
        random_perm_scope=args.random_perm_scope,
        seed=args.seed,
        skip_rate=args.skip_rate,
        return_metadata=not args.no_metadata,
    )

    print("len(ds) =", len(ds))

    if len(ds) > 0:
        item = ds[0]
        print("pose shape:", tuple(item["pose"].shape))
        print("mask shape:", tuple(item["mask"].shape))
        print("timepoints shape:", tuple(item["timepoints"].shape))

        if "animal" in item:
            print("animal:", item["animal"])
            print("mode:", item["ablation_mode"])
            print("json:", item["json_path"])
            print("slots shape:", tuple(item["slots"].shape))
            print("slots:", item["slots"].tolist())
            print("root:", int(item["root"]))
            print("edges shape:", tuple(item["edges"].shape))
            print("first edges:", item["edges"][:10])

            if "slots_perm" in item:
                print("slots_perm:", item["slots_perm"].tolist())

            if "root_perm" in item:
                print("root_perm:", int(item["root_perm"]))

            if "root_raw" in item:
                print("root_raw:", int(item["root_raw"]))
