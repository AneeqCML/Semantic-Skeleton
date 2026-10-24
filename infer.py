import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import copy

import torch
from torch.utils.data import DataLoader

from model import ModelMain
from dataset import WindowMaskDataset


parser = argparse.ArgumentParser()

parser.add_argument("--joints", type=int, default=25)
parser.add_argument("--input_n", type=int, default=50)
parser.add_argument("--output_n", type=int, default=25)

parser.add_argument("--model_s", type=str, required=True)
parser.add_argument("--model_l", type=str, required=True)

parser.add_argument("--data_dir", type=str, required=True)
parser.add_argument("--save_dir", type=str, default="eval_results")

parser.add_argument("--nsample", type=int, default=5)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--pcp_alpha", type=float, default=0.5)

args = parser.parse_args()


base_config = {
    "diffusion": {
        "layers": 11,
        "channels": 64,
        "nheads": 8,
        "diffusion_embedding_dim": 128,
        "beta_start": 0.0001,
        "beta_end": 0.5,
        "num_steps": 50,
        "schedule": "cosine",
    },
    "model": {
        "is_unconditional": 0,
        "timeemb": 128,
        "featureemb": 16,
    },
}


OLD_CONNECTIONS = [
    [5, 25], [1, 25], [2, 25], [3, 25], [4, 25], [0, 25],
    [0, 6], [0, 7], [6, 8], [8, 9], [9, 10],
    [7, 11], [11, 12], [12, 13],
    [0, 14], [14, 15], [14, 16],
    [15, 17], [17, 18], [18, 21],
    [16, 19], [19, 20], [20, 22],
    [14, 23], [23, 24],
]


OLD_TO_NEW = {
    25: 0,
    0: 1,
    5: 2,
    1: 3,
    2: 4,
    4: 5,
    3: 6,
    14: 7,
    7: 8,
    6: 9,
    15: 10,
    16: 11,
    11: 12,
    8: 13,
    23: 14,
    17: 15,
    19: 16,
    12: 17,
    9: 18,
    18: 19,
    20: 20,
    13: 21,
    10: 22,
    21: 23,
    22: 24,
}


def build_canonical_edges():
    edges = []

    for a, b in OLD_CONNECTIONS:
        if a in OLD_TO_NEW and b in OLD_TO_NEW:
            edges.append([OLD_TO_NEW[a], OLD_TO_NEW[b]])

    return np.asarray(edges, dtype=np.int64)


CANONICAL_EDGES = build_canonical_edges()


def get_animal_from_path(path):
    parts = os.path.normpath(str(path)).split(os.sep)

    known_animals = [
        "foxZED",
        "elkML",
        "pumaRW",
        "tigerD8H",
        "bucksYJL",
    ]

    for p in parts:
        if p in known_animals:
            return p

    return "unknown"


def mpjpe_error(pred, gt):
    pred = pred.contiguous().view(-1, 3)
    gt = gt.contiguous().view(-1, 3)

    return torch.mean(torch.norm(gt - pred, dim=1))


def compute_pose_metrics(pred, gt, edges, alpha=0.5):
    pred_s = pred[:, edges[:, 0], :]
    pred_e = pred[:, edges[:, 1], :]

    gt_s = gt[:, edges[:, 0], :]
    gt_e = gt[:, edges[:, 1], :]

    start_err = np.linalg.norm(gt_s - pred_s, axis=2)
    end_err = np.linalg.norm(gt_e - pred_e, axis=2)

    endpoint_err = (start_err + end_err) / 2.0

    gt_bone_len = np.linalg.norm(gt_s - gt_e, axis=2)
    pred_bone_len = np.linalg.norm(pred_s - pred_e, axis=2)

    eps = 1e-8
    correct = endpoint_err < alpha * (gt_bone_len + eps)

    return {
        "pcp": float(np.mean(correct)),
        "bone_error": float(np.mean(endpoint_err)),
        "bone_length_error": float(np.mean(np.abs(pred_bone_len - gt_bone_len))),
        "bone_std": float(np.mean(np.std(pred_bone_len, axis=0))),
    }


def safe_name(path_or_name):
    base = os.path.basename(str(path_or_name))
    base = os.path.splitext(base)[0]

    return base.replace(" ", "_")


def evaluate():
    os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    print("Joints:", args.joints)
    print("Input/output:", args.input_n, "->", args.output_n)

    config_short = copy.deepcopy(base_config)
    config_long = copy.deepcopy(base_config)

    config_short["diffusion"]["layers"] = 4
    config_long["diffusion"]["layers"] = 12

    model_s = ModelMain(config_short, device, target_dim=args.joints * 3)
    model_l = ModelMain(config_long, device, target_dim=args.joints * 3)

    model_s.load_state_dict(
        torch.load(f"{args.model_s}/model.pth", map_location=device)
    )
    model_l.load_state_dict(
        torch.load(f"{args.model_l}/model.pth", map_location=device)
    )

    model_s.to(device)
    model_l.to(device)

    model_s.eval()
    model_l.eval()

    dataset = WindowMaskDataset(args.data_dir, args.input_n, args.output_n)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    all_preds = []
    all_gt = []
    rows = []

    with torch.no_grad():
        with tqdm(loader) as it:
            for batch_no, batch in enumerate(it, start=1):
                json_paths = batch["json_path"]
                json_names = batch["json_name"]
                window_starts = batch["window_start"]

                pose = batch["pose"].to(device)
                mask = batch["mask"].to(device)
                timepoints = batch["timepoints"].to(device)

                pose_np = pose.cpu().numpy()

                diff = pose_np[:, 1:] - pose_np[:, :-1]
                diff = diff.reshape(diff.shape[0], diff.shape[1], args.joints, 3)
                mag = np.linalg.norm(diff, axis=3)
                motion_mag = mag.mean(axis=(1, 2))

                short_batch = {
                    "pose": pose.clone()[:, :args.input_n + 5],
                    "mask": mask.clone()[:, :args.input_n + 5],
                    "timepoints": timepoints.clone()[:, :args.input_n + 5],
                }

                samples_short, _, _, _ = model_s.evaluate(short_batch, args.nsample)

                samples_short = samples_short.permute(0, 1, 3, 2)
                samples_mean_short = np.mean(samples_short.cpu().numpy(), axis=1)

                short_mpjpes = []

                for i in range(len(samples_mean_short)):
                    short_pred = samples_mean_short[i][args.input_n:args.input_n + 5]
                    short_gt = pose_np[i][args.input_n:args.input_n + 5]

                    short_pred_tj3 = short_pred.reshape(5, args.joints, 3)
                    short_gt_tj3 = short_gt.reshape(5, args.joints, 3)

                    short_mpjpe = mpjpe_error(
                        torch.from_numpy(short_pred_tj3),
                        torch.from_numpy(short_gt_tj3),
                    ).item()

                    short_mpjpes.append(short_mpjpe)

                pose_for_long = pose.clone()
                mask_for_long = mask.clone()

                pose_for_long[:, :args.input_n + 5] = torch.from_numpy(
                    samples_mean_short
                ).to(device)

                mask_for_long[:, :args.input_n + 5] = 1

                long_batch = {
                    "pose": pose_for_long,
                    "mask": mask_for_long,
                    "timepoints": timepoints,
                }

                samples_long, c_target, _, _ = model_l.evaluate(
                    long_batch,
                    args.nsample,
                )

                samples_long = samples_long.permute(0, 1, 3, 2)
                c_target = c_target.permute(0, 2, 1)

                samples_mean_long = np.mean(samples_long.cpu().numpy(), axis=1)
                c_target_np = c_target.cpu().numpy()

                for i in range(len(samples_mean_long)):
                    pred = samples_mean_long[i][args.input_n + 5:]
                    gt = c_target_np[i][args.input_n + 5:]

                    pred_tj3 = pred.reshape(args.output_n - 5, args.joints, 3)
                    gt_tj3 = gt.reshape(args.output_n - 5, args.joints, 3)

                    mpjpe_after5 = mpjpe_error(
                        torch.from_numpy(pred_tj3),
                        torch.from_numpy(gt_tj3),
                    ).item()

                    extra = compute_pose_metrics(
                        pred_tj3,
                        gt_tj3,
                        CANONICAL_EDGES,
                        alpha=args.pcp_alpha,
                    )

                    json_name = safe_name(json_names[i])
                    json_path = json_paths[i]
                    animal = get_animal_from_path(json_path)

                    window_start = (
                        int(window_starts[i].item())
                        if torch.is_tensor(window_starts[i])
                        else int(window_starts[i])
                    )

                    out_dir = os.path.join(args.save_dir, str(animal), json_name)
                    os.makedirs(out_dir, exist_ok=True)

                    out_file = os.path.join(
                        out_dir,
                        f"window_{window_start:06d}.npz",
                    )

                    np.savez(
                        out_file,
                        prediction_after5=pred_tj3,
                        ground_truth_after5=gt_tj3,
                        source_json=json_path,
                        json_name=json_names[i],
                        animal=animal,
                        window_start=window_start,
                        short_mpjpe_first5=short_mpjpes[i],
                        mpjpe_after5=mpjpe_after5,
                        motion=motion_mag[i],
                        pcp_after5=extra["pcp"],
                        bone_error_after5=extra["bone_error"],
                        bone_length_error_after5=extra["bone_length_error"],
                        bone_std_after5=extra["bone_std"],
                        canonical_edges=CANONICAL_EDGES,
                    )

                    all_preds.append(pred_tj3)
                    all_gt.append(gt_tj3)

                    rows.append({
                        "sequence_id": len(rows),
                        "animal": animal,
                        "json_name": json_names[i],
                        "json_path": json_path,
                        "window_start": window_start,
                        "short_mpjpe_first5": short_mpjpes[i],
                        "mpjpe_after5": mpjpe_after5,
                        "motion": motion_mag[i],
                        "pcp_after5": extra["pcp"],
                        "bone_error_after5": extra["bone_error"],
                        "bone_length_error_after5": extra["bone_length_error"],
                        "bone_std_after5": extra["bone_std"],
                        "saved_file": out_file,
                    })

                avg = np.mean([r["mpjpe_after5"] for r in rows]) if rows else 0.0
                it.set_postfix({"avg_mpjpe_after5": avg})

    df = pd.DataFrame(rows)

    metrics_path = os.path.join(args.save_dir, "metrics_per_window.csv")
    df.to_csv(metrics_path, index=False)

    preds = np.array(all_preds)
    gts = np.array(all_gt)

    np.savez(
        os.path.join(args.save_dir, "predictions_all_after5.npz"),
        predictions_after5=preds,
        ground_truth_after5=gts,
        canonical_edges=CANONICAL_EDGES,
    )

    overall_summary = pd.DataFrame([df.mean(numeric_only=True)])
    overall_summary.to_csv(
        os.path.join(args.save_dir, "summary_overall.csv"),
        index=False,
    )

    animal_summary = df.groupby("animal").mean(numeric_only=True)
    animal_summary.to_csv(os.path.join(args.save_dir, "summary_by_animal.csv"))

    print("\n========== OVERALL METRICS ==========")
    print(overall_summary)

    print("\n========== METRICS BY ANIMAL ==========")
    print(animal_summary)

    print("\nSaved:", metrics_path)
    print("Saved:", os.path.join(args.save_dir, "summary_overall.csv"))
    print("Saved:", os.path.join(args.save_dir, "summary_by_animal.csv"))

    motions = df["motion"].values

    p25 = np.percentile(motions, 25)
    p50 = np.percentile(motions, 50)
    p65 = np.percentile(motions, 65)
    p75 = np.percentile(motions, 75)
    p85 = np.percentile(motions, 85)
    p95 = np.percentile(motions, 95)

    def bucket_motion(m):
        if m <= p25:
            return "P25"
        elif m <= p50:
            return "P50"
        elif m <= p65:
            return "P65"
        elif m <= p75:
            return "P75"
        elif m <= p85:
            return "P85"
        elif m <= p95:
            return "P95"
        else:
            return "P100"

    df["motion_bucket"] = df["motion"].apply(bucket_motion)

    bucket_summary = df.groupby("motion_bucket").agg({
        "short_mpjpe_first5": "mean",
        "mpjpe_after5": "mean",
        "pcp_after5": "mean",
        "bone_error_after5": "mean",
        "bone_length_error_after5": "mean",
        "bone_std_after5": "mean",
        "sequence_id": "count",
    }).rename(columns={"sequence_id": "n"})

    bucket_path = os.path.join(args.save_dir, "metrics_by_motion_bucket.csv")
    bucket_summary.to_csv(bucket_path)

    print("\n========== METRICS BY MOTION BUCKET ==========")
    print(bucket_summary)

    print("\nSaved:", bucket_path)
    print("Saved results to:", args.save_dir)


if __name__ == "__main__":
    evaluate()
