#!/usr/bin/env python3
#Canonizer script , which dataset scripts resort to 
import os, json, argparse
import numpy as np
from collections import defaultdict, deque

def build_adj(connections):
    adj = defaultdict(list)
    for a, b in connections:
        adj[a].append(b)
        adj[b].append(a)
    return adj

def bfs_levels_parents(adj, root):
    lvl = {root: 0}
    par = {root: -1}
    q = deque([root])
    while q:
        u = q.popleft()
        for v in adj[u]:
            if v not in lvl:
                lvl[v] = lvl[u] + 1
                par[v] = u
                q.append(v)
    return lvl, par

def subtree_sizes_from_parent(par, root):
    ch = defaultdict(list)
    for v, p in par.items():
        if p != -1:
            ch[p].append(v)
    sub = {}
    def dfs(u):
        t = 1
        for c in ch[u]:
            t += dfs(c)
        sub[u] = t
        return t
    dfs(root)
    return sub

def mean_leaf_distance(adj, V):
    leaves = [v for v in V if len(adj[v]) == 1]
    d = {}
    for v in V:
        q = deque([(v, 0)])
        vis = {v}
        total = 0
        cnt = 0
        while q:
            u, dist = q.popleft()
            if u in leaves:
                total += dist
                cnt += 1
            for w in adj[u]:
                if w not in vis:
                    vis.add(w)
                    q.append((w, dist + 1))
        d[v] = total / max(cnt, 1)
    return d


def extract_cam3d(data):
    if "cam_3d" in data:
        return np.asarray(data["cam_3d"], dtype=np.float64)
    if "points" in data and isinstance(data["points"], dict) and "cam_3d" in data["points"]:
        return np.asarray(data["points"]["cam_3d"], dtype=np.float64)
    return None

# PCA-based orientation is computed from the available pose sequence
# to describe spatial joint relationships.
#
# Forecasting predictions are made from observed XYZ coordinates.
# The downstream 25-joint selection explicitly ranks joints using
# graph-level structure (subtree size and degree), not PCA scores.
#
# Note: PCA may indirectly affect ordering when structural scores tie. This wasn't used in AnimalSyn3D work, was just included as a potential tie breaker in the future. 
def pca_axes(X):
    # X: (N,3) centered
    C = (X.T @ X) / max(len(X), 1)
    evals, evecs = np.linalg.eigh(C)  # ascending
    u_small = evecs[:, 0]
    u_mid   = evecs[:, 1]
    u_large = evecs[:, 2]
    u_small /= (np.linalg.norm(u_small) + 1e-12)
    u_mid   /= (np.linalg.norm(u_mid)   + 1e-12)
    u_large /= (np.linalg.norm(u_large) + 1e-12)
    return u_small, u_mid, u_large

def principal_axis_from_vel(cam_TJ3, center=None, eps=1e-12):
    if cam_TJ3.shape[0] < 2:
        X = cam_TJ3[0] - (np.mean(cam_TJ3[0], axis=0) if center is None else center)
        _, _, f = pca_axes(X)
        return f

    V = cam_TJ3[1:] - cam_TJ3[:-1]  # (T-1,J,3)
    V = V.reshape(-1, 3)
    sp = np.linalg.norm(V, axis=1)
    V = V[sp > 1e-6]
    if len(V) < 10:
        X = cam_TJ3[0] - (np.mean(cam_TJ3[0], axis=0) if center is None else center)
        _, _, f = pca_axes(X)
        return f

    C = (V.T @ V) / max(len(V), 1)
    evals, evecs = np.linalg.eigh(C)
    f = evecs[:, 2]  # largest
    f = f / (np.linalg.norm(f) + eps)
    return f

def build_body_frame(cam_TJ3, lvl, adj, root, frame_for_shape=0):
    T, J, _ = cam_TJ3.shape
    frame_for_shape = int(np.clip(frame_for_shape, 0, T - 1))
    P0 = cam_TJ3[frame_for_shape]  # (J,3)
    c = np.mean(P0, axis=0)

    f = principal_axis_from_vel(cam_TJ3, center=c)
    if cam_TJ3.shape[0] >= 2:
        avg_motion = np.mean(
            [(cam_TJ3[t+1] - cam_TJ3[t]).reshape(-1, 3).mean(axis=0)
             for t in range(cam_TJ3.shape[0]-1)],
            axis=0
        )
        if np.dot(avg_motion, f) < 0:
            f = -f
    X = P0 - c
    u0, r0, _ = pca_axes(X)

    u = u0 - np.dot(u0, f) * f
    if np.linalg.norm(u) < 1e-6:
        u = np.cross(f, r0)
    u = u / (np.linalg.norm(u) + 1e-12)

    r = np.cross(u, f)
    r = r / (np.linalg.norm(r) + 1e-12)

    leaves = [v for v in lvl.keys() if len(adj[v]) == 1]
    if leaves:
        maxlvl = max(lvl[v] for v in leaves if v in lvl)
        far = [v for v in leaves if lvl.get(v, -1) == maxlvl]
        if far:
            vec = np.mean([X[v] for v in far], axis=0)
            if np.dot(vec, f) < 0:
                f = -f
                r = -r


    lvl1 = [v for v in lvl if lvl[v] == 1]
    if len(lvl1) >= 2:
        lateral = np.mean([(P0[v] - c) for v in lvl1], axis=0)
        if np.dot(lateral, r) < 0:
            r = -r

  
    u = np.cross(f, r)
    u = u / (np.linalg.norm(u) + 1e-12)

    return c, f, r, u


def canonicalize_json(path, frame_for_shape=0):
    data = json.load(open(path, "r"))
    if "connections" not in data:
        raise ValueError("missing 'connections'")

    adj = build_adj(data["connections"])
    V = sorted(adj.keys())
    if not V:
        raise ValueError("empty graph")

   
    root = max(V, key=lambda v: (len(adj[v]), -v))
    lvl, par = bfs_levels_parents(adj, root)
    sub = subtree_sizes_from_parent(par, root)
    mld = mean_leaf_distance(adj, V)

    cam = extract_cam3d(data)
    used_motion = False
    if cam is not None:
        if cam.ndim == 2:
            cam = cam[None, ...]  # (1,J,3)
        if cam.ndim == 3 and cam.shape[1] > max(V):
            c, f, r, u = build_body_frame(cam, lvl, adj, root, frame_for_shape=frame_for_shape)
            used_motion = (cam.shape[0] >= 2)
        else:
            cam = None
    else:
        cam = None

  
    fb = {v: 0.0 for v in V}
    lr = {v: 0.0 for v in V}
    ud = {v: 0.0 for v in V}
    if cam is not None:
        P0 = cam[int(np.clip(frame_for_shape, 0, cam.shape[0]-1))]
        for v in V:
            rel = P0[v] - c
            fb[v] = float(np.dot(rel, f))  # front/back
            lr[v] = float(np.dot(rel, r))  # left/right
            ud[v] = float(np.dot(rel, u))  # up/down

    def key(v):
        return (
            lvl.get(v, 10**9),
            -len(adj[v]),
            -sub.get(v, 0),
            mld.get(v, 0.0),
            -round(fb[v], 6),
            round(lr[v], 6),
            -round(ud[v], 6),
            v
        )

    order = sorted(V, key=key)

    sig = []
    for v in order:
        sig.append((
            v,
            lvl.get(v, -1),
            len(adj[v]),
            sub.get(v, 0),
            round(mld.get(v, 0.0), 2),
            round(fb[v], 3),
            round(lr[v], 3),
        ))

    return {
        "root": root,
        "used_motion": used_motion,
        "canonical_order": order,
        "signature": sig,
    }

def process_dir(dirpath, write=False, frame_for_shape=0):
    for name in sorted(os.listdir(dirpath)):
        if not name.endswith(".json"):
            continue
        path = os.path.join(dirpath, name)
        try:
            out = canonicalize_json(path, frame_for_shape=frame_for_shape)
            print(f"{name}  root={out['root']}  motion={'ON' if out['used_motion'] else 'OFF'}")
            # print first ~30 rows
            for row in out["signature"][:30]:
                print(" ", row)
            print()

            if write:
                data = json.load(open(path, "r"))
                data["canonical_order"] = out["canonical_order"]
                data["canonical_signature"] = out["signature"]
                data["canonical_root"] = out["root"]
                data["canonical_used_motion"] = out["used_motion"]

                out_path = path[:-5] + "_canon.json"
                with open(out_path, "w") as f:
                    json.dump(data, f, indent=2)
        except Exception as e:
            print(f"FAILED {name}: {e}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="infile", default=None, help="single json file")
    ap.add_argument("--dir", default=None, help="directory of json files")
    ap.add_argument("--write", action="store_true", help="write *_canon.json")
    ap.add_argument("--frame", type=int, default=0, help="frame index to use for pose projections (default 0)")
    args = ap.parse_args()

    if args.infile:
        out = canonicalize_json(args.infile, frame_for_shape=args.frame)
        print("canonical_order =", out["canonical_order"])
        print("root =", out["root"], "motion =", out["used_motion"])
        print("signature (first 30):")
        for row in out["signature"][:30]:
            print(" ", row)

        if args.write:
            data = json.load(open(args.infile, "r"))
            data["canonical_order"] = out["canonical_order"]
            data["canonical_signature"] = out["signature"]
            data["canonical_root"] = out["root"]
            data["canonical_used_motion"] = out["used_motion"]
            out_path = args.infile[:-5] + "_canon.json"
            with open(out_path, "w") as f:
                json.dump(data, f, indent=2)
            print("wrote", out_path)
        return

    if args.dir:
        process_dir(args.dir, write=args.write, frame_for_shape=args.frame)
        return

    raise SystemExit("Provide --in <file.json> or --dir <folder>")

if __name__ == "__main__":
    main()
