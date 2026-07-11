import os
from tqdm import tqdm
import numpy as np
import pandas as pd
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam

from model import ModelMain
from dataset import WindowMaskDataset


parser = argparse.ArgumentParser(description="Train/test short-term diffusion model")

parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--skip_rate_train", type=int, default=5)
parser.add_argument("--skip_rate_val", type=int, default=15)

parser.add_argument("--joints", type=int, default=25)

parser.add_argument("--input_n", type=int, default=20)
parser.add_argument("--output_n", type=int, default=5)

parser.add_argument("--mode", type=str, default="train", choices=["train", "test"])
parser.add_argument("--resume", action="store_true")

parser.add_argument("--output_dir", type=str, default="default")
parser.add_argument("--data_dir", type=str, default="./data")
parser.add_argument("--val_dir", type=str, default=None)
parser.add_argument(
    "--ablation_mode",
    type=str,
    default="canonical",
    choices=["canonical", "random_perm_drop"],
    help=(
        "canonical = old canon_v-based canonical 25-joint loader; "
        "random_perm_drop = random per-animal raw joint permutation, keep first 25"
    ),
)

parser.add_argument(
    "--no_root_norm",
    action="store_true",
    help="Disable root normalization. By default, root normalization is ON.",
)

parser.add_argument(
    "--random_within_levels",
    action="store_true",
    help="For canonical mode: randomly choose joints within each BFS level quota.",
)

parser.add_argument(
    "--random_perm_scope",
    type=str,
    default="animal",
    choices=["animal", "file"],
    help=(
        "For random_perm_drop: animal = one random permutation per animal folder; "
        "file = one random permutation per JSON file."
    ),
)

parser.add_argument(
    "--seed",
    type=int,
    default=0,
    help="Seed for deterministic random ablations.",
)

args = parser.parse_args()
print(args)


config = {
    "train": {
        "epochs": 50,
        "batch_size": 32,
        "batch_size_test": 32,
        "lr": 1.0e-3,
    },
    "diffusion": {
        "layers": 4,
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


def save_csv_log(head, value, is_create=False, file_name="test"):
    if len(value.shape) < 2:
        value = np.expand_dims(value, axis=0)

    df = pd.DataFrame(value)
    file_path = f"{output_dir}/{file_name}.csv"

    if not os.path.exists(file_path) or is_create:
        df.to_csv(file_path, header=head, index=False)
    else:
        with open(file_path, "a") as f:
            df.to_csv(f, header=False, index=False)


def save_state(model, optimizer, scheduler, epoch_no, foldername):
    params = {
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch_no,
    }

    if isinstance(model, nn.DataParallel):
        torch.save(model.module.state_dict(), foldername + "/model.pth")
    else:
        torch.save(model.state_dict(), foldername + "/model.pth")

    torch.save(params, foldername + "/params.pth")


def train(
    model,
    config,
    train_loader,
    valid_loader=None,
    valid_epoch_interval=5,
    foldername="",
    load_state=False,
):
    optimizer = Adam(model.parameters(), lr=config["lr"], weight_decay=1e-6)

    p1 = int(0.75 * config["epochs"])
    p2 = int(0.9 * config["epochs"])

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[p1, p2],
        gamma=0.1,
    )

    best_valid_loss = 1e10

    for epoch_no in range(config["epochs"]):
        avg_loss = 0.0
        model.train()

        with tqdm(train_loader, mininterval=5.0, maxinterval=50.0) as it:
            for batch_no, train_batch in enumerate(it, start=1):
                batch = train_batch

                optimizer.zero_grad()

                loss = model(batch).mean()

                loss.backward()
                optimizer.step()

                avg_loss += loss.item()

                it.set_postfix({
                    "avg_epoch_loss": avg_loss / batch_no,
                    "epoch": epoch_no,
                })

        lr_scheduler.step()

        if valid_loader is not None and (epoch_no + 1) % valid_epoch_interval == 0:
            model.eval()
            avg_loss_valid = 0.0

            with torch.no_grad():
                with tqdm(valid_loader, mininterval=5.0, maxinterval=50.0) as it:
                    for batch_no, valid_batch in enumerate(it, start=1):
                        batch = valid_batch

                        loss = model(batch, is_train=0).mean()
                        avg_loss_valid += loss.item()

                        it.set_postfix({
                            "valid_avg_epoch_loss": avg_loss_valid / batch_no,
                            "epoch": epoch_no,
                        })

            valid_epoch_loss = avg_loss_valid / batch_no

            if best_valid_loss > valid_epoch_loss:
                best_valid_loss = valid_epoch_loss

                print("\nBest validation loss updated:", valid_epoch_loss)

                save_state(model, optimizer, lr_scheduler, epoch_no, foldername)

    save_state(model, optimizer, lr_scheduler, config["epochs"], foldername)


def mpjpe_error(batch_imp, batch_gt):
    batch_imp = batch_imp.contiguous().view(-1, 3)
    batch_gt = batch_gt.contiguous().view(-1, 3)

    return torch.mean(torch.norm(batch_gt - batch_imp, 2, 1))


def evaluate(model, loader, nsample=5, sample_strategy="best"):
    with torch.no_grad():
        model.eval()

        mpjpe_total = 0.0

        with tqdm(loader, mininterval=5.0, maxinterval=50.0) as it:
            for batch_no, test_batch in enumerate(it, start=1):
                batch = test_batch

                if isinstance(model, nn.DataParallel):
                    output = model.module.evaluate(batch, nsample)
                else:
                    output = model.evaluate(batch, nsample)

                samples, c_target, eval_points, observed_time = output

                samples = samples.permute(0, 1, 3, 2)
                c_target = c_target.permute(0, 2, 1)

                samples_np = samples.cpu().numpy()
                c_target_np = c_target.cpu().numpy()

                samples_mean = np.mean(samples_np, axis=1)

                renorm_pose = []
                renorm_c_target = []

                for i in range(len(samples_mean)):
                    renorm_c_target_i = c_target_np[i][input_n:input_n + output_n]

                    if sample_strategy == "best":
                        best_pose = None
                        best_error = float("inf")

                        for j in range(nsample):
                            pose_j = samples_np[i][j][input_n:input_n + output_n]

                            error = mpjpe_error(
                                torch.from_numpy(pose_j).view(output_n, args.joints, 3),
                                torch.from_numpy(renorm_c_target_i).view(output_n, args.joints, 3),
                            )

                            if error.item() < best_error:
                                best_error = error.item()
                                best_pose = pose_j
                    else:
                        best_pose = samples_mean[i][input_n:input_n + output_n]

                    renorm_pose.append(best_pose)
                    renorm_c_target.append(renorm_c_target_i)

                renorm_pose = torch.from_numpy(np.array(renorm_pose))
                renorm_c_target = torch.from_numpy(np.array(renorm_c_target))

                mpjpe_current = mpjpe_error(
                    renorm_pose.view(-1, output_n, args.joints, 3),
                    renorm_c_target.view(-1, output_n, args.joints, 3),
                )

                mpjpe_total += mpjpe_current.item()

                it.set_postfix({
                    "mpjpe_total": mpjpe_total / batch_no,
                })

        print("Average MPJPE:", mpjpe_total / batch_no)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Using device:", device)

    data_dir = args.data_dir
    output_dir = args.output_dir

    input_n = args.input_n
    output_n = args.output_n

    config["train"]["epochs"] = args.epochs

    os.makedirs(output_dir, exist_ok=True)

    model = ModelMain(config, device, target_dim=(args.joints * 3))

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    model.to(device)

    if args.mode == "train":
        dataset = WindowMaskDataset(
            data_dir,
            in_len=input_n,
            out_len=output_n,
            ablation_mode=args.ablation_mode,
            root_norm=not args.no_root_norm,
            random_within_levels=args.random_within_levels,
            random_perm_scope=args.random_perm_scope,
            seed=args.seed,
        )
        print(">>> Training dataset length:", len(dataset))

        if len(dataset) == 0:
            raise RuntimeError(
                "No training windows found. Check data_dir, input_n/output_n, and sequence lengths."
            )

        train_loader = DataLoader(
            dataset,
            batch_size=config["train"]["batch_size"],
            shuffle=True,
            num_workers=0,
            pin_memory=True,
        )

        valid_loader = None

        if args.val_dir is not None:
            valid_dataset = WindowMaskDataset(args.val_dir, in_len=input_n, out_len=output_n)

            print(">>> Validation dataset length:", len(valid_dataset))

            if len(valid_dataset) == 0:
                print("WARNING: Validation dataset has 0 windows. Disabling validation.")
            else:
                valid_loader = DataLoader(
                    valid_dataset,
                    batch_size=config["train"]["batch_size"],
                    shuffle=False,
                    num_workers=0,
                    pin_memory=True,
                )

        train(
            model,
            config["train"],
            train_loader,
            valid_loader=valid_loader,
            foldername=output_dir,
            load_state=args.resume,
        )

    elif args.mode == "test":
        test_dataset = WindowMaskDataset(data_dir, in_len=input_n, out_len=output_n)

        print(">>> Test dataset length:", len(test_dataset))

        if len(test_dataset) == 0:
            raise RuntimeError(
                "No test windows found. Check data_dir, input_n/output_n, and sequence lengths."
            )

        test_loader = DataLoader(
            test_dataset,
            batch_size=config["train"]["batch_size_test"],
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )

        model_path = f"{output_dir}/model.pth"

        if isinstance(model, nn.DataParallel):
            model.module.load_state_dict(torch.load(model_path, map_location=device))
        else:
            model.load_state_dict(torch.load(model_path, map_location=device))

        evaluate(model, test_loader)