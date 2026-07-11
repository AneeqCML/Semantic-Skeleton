import os
from tqdm import tqdm
import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam

from model import ModelMain
from dataset import WindowMaskDataset


parser = argparse.ArgumentParser(description='Arguments for running the scripts')

parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--skip_rate_train", type=int, default=1)
parser.add_argument("--skip_rate_val", type=int, default=25)
parser.add_argument("--joints", type=int, default=25)
parser.add_argument("--input_n", type=int, default=25)
parser.add_argument("--output_n", type=int, default=25)

parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
parser.add_argument('--resume', action='store_true')

parser.add_argument('--output_dir', type=str, default='model_s')
parser.add_argument('--model_s', type=str, default='model_s')
parser.add_argument('--model_l', type=str, default='model_l')

parser.add_argument('--data_dir', type=str, default='../foxZED')
parser.add_argument(
    "--ablation_mode",
    choices=["canonical", "random_perm_drop", "random_perm_canonical"],
    default="canonical",
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
    'train': {
        'epochs': 50,
        'batch_size': 32,
        'batch_size_test': 32,
        'lr': 1.0e-3
    },
    'diffusion': {
        'layers': 12,
        'channels': 64,
        'nheads': 8,
        'diffusion_embedding_dim': 128,
        'beta_start': 0.0001,
        'beta_end': 0.5,
        'num_steps': 50,
        'schedule': "cosine"
    },
    'model': {
        'is_unconditional': 0,
        'timeemb': 128,
        'featureemb': 16
    }
}


def mpjpe_error(batch_imp, batch_gt):
    batch_imp = batch_imp.contiguous().view(-1, 3)
    batch_gt = batch_gt.contiguous().view(-1, 3)
    return torch.mean(torch.norm(batch_gt - batch_imp, 2, 1))


def train(model, config, train_loader, valid_loader=None, foldername="", load_state=False):

    optimizer = Adam(model.parameters(), lr=config["lr"], weight_decay=1e-6)

    p1 = int(0.75 * config["epochs"])
    p2 = int(0.9 * config["epochs"])

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[p1, p2], gamma=0.1
    )

    best_valid_loss = 1e10

    for epoch_no in range(config["epochs"]):

        avg_loss = 0
        model.train()

        with tqdm(train_loader) as it:

            for batch_no, batch in enumerate(it, start=1):

                optimizer.zero_grad()

                loss = model(batch).mean()

                loss.backward()

                avg_loss += loss.item()

                optimizer.step()

                it.set_postfix({
                    "avg_epoch_loss": avg_loss / batch_no,
                    "epoch": epoch_no,
                })

        lr_scheduler.step()
        os.makedirs(foldername, exist_ok=True)
        torch.save(model.state_dict(), foldername + "/model.pth")
        if valid_loader is not None:

            model.eval()

            avg_loss_valid = 0

            with torch.no_grad():

                for batch_no, batch in enumerate(valid_loader, start=1):

                    loss = model(batch, is_train=0).mean()

                    avg_loss_valid += loss.item()

            if best_valid_loss > avg_loss_valid:

                best_valid_loss = avg_loss_valid

                torch.save(model.state_dict(), foldername + "/model.pth")


def evaluate(model_s, model_l, loader, nsample=5):

    with torch.no_grad():

        model_s.eval()
        model_l.eval()

        mpjpe_total = 0

        titles = np.array(range(output_n)) + 1
        m_p3d = np.zeros([output_n])

        n = 0

        with tqdm(loader) as it:

            for batch_no, batch in enumerate(it, start=1):

                batch_size = batch["pose"].shape[0]

                n += batch_size

                s = {
                    "pose": batch["pose"].clone()[:, :input_n + 5],
                    "mask": batch["mask"].clone()[:, :input_n + 5],
                    "timepoints": batch["timepoints"].clone()[:, :input_n + 5]
                }

                output = model_s.evaluate(s, nsample)

                samples, _, _, _ = output

                samples = samples.permute(0, 1, 3, 2)

                samples_mean = np.mean(samples.cpu().numpy(), axis=1)

                batch["pose"][:, :input_n + 5] = torch.from_numpy(samples_mean)

                batch["mask"][:, :input_n + 5] = 1

                output = model_l.evaluate(batch, nsample)

                samples, c_target, eval_points, observed_time = output

                samples = samples.permute(0, 1, 3, 2)

                c_target = c_target.permute(0, 2, 1)

                samples_mean = np.mean(samples.cpu().numpy(), axis=1)

                renorm_pose = []
                renorm_c_target = []

                for i in range(len(samples_mean)):

                    renorm_pose_i = samples_mean[i][input_n:]
                    renorm_target_i = c_target.cpu().numpy()[i][input_n:]

                    renorm_pose.append(renorm_pose_i)
                    renorm_c_target.append(renorm_target_i)

                renorm_pose = torch.from_numpy(np.array(renorm_pose))
                renorm_c_target = torch.from_numpy(np.array(renorm_c_target))

                mpjpe_current = mpjpe_error(
                    renorm_pose.view(-1, output_n, args.joints, 3),
                    renorm_c_target.view(-1, output_n, args.joints, 3)
                )

                mpjpe_total += mpjpe_current.item()

                it.set_postfix({
                    "average_mpjpe": mpjpe_total / batch_no
                })

        print("Average MPJPE:", mpjpe_total / batch_no)


if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data_dir = args.data_dir
    output_dir = args.output_dir

    input_n = args.input_n
    output_n = args.output_n

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    model = ModelMain(config, device, target_dim=(args.joints * 3))

    model.to(device)

    if args.mode == 'train':

        dataset = WindowMaskDataset(
            data_dir,
            in_len=input_n,
            out_len=output_n,
            ablation_mode=args.ablation_mode,
            root_norm=not args.no_root_norm,
            random_within_levels=args.random_within_levels,
            random_perm_scope=args.random_perm_scope,
            seed=args.seed,
            return_metadata=False,
        )

        train_loader = DataLoader(dataset, batch_size=config["train"]["batch_size"], shuffle=True)

        train(model, config["train"], train_loader, foldername=output_dir)

    elif args.mode == 'test':

        model_s = ModelMain(config, device, target_dim=(args.joints * 3))
        model_l = ModelMain(config, device, target_dim=(args.joints * 3))

        model_s.load_state_dict(torch.load(f'{args.model_s}/model.pth'))
        model_l.load_state_dict(torch.load(f'{args.model_l}/model.pth'))

        model_s.to(device)
        model_l.to(device)
        dataset = WindowMaskDataset(data_dir, input_n, output_n)

        test_loader = DataLoader(dataset, batch_size=config["train"]["batch_size_test"], shuffle=False)

        evaluate(model_s, model_l, test_loader)