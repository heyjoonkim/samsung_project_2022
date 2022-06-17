import argparse
import logging
import os
import random
import time
import json
import math
import torchvision.transforms as transforms
import wandb
import torch
from torch.utils.data import DataLoader
from torch import optim
from torch.optim.lr_scheduler import LinearLR
from tqdm.auto import tqdm

from data import SEMDataset
from models.unet import UNet

from torchvision.utils import save_image, make_grid
from PIL import Image
def parse_args():
    parser = argparse.ArgumentParser(description="Test a U-net model.")

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="A seed for reproducible training."
    )
    parser.add_argument(
        '--dataset_dir',
        default='./dataset/AI_challenge_data',
        help='Dataset directory.'
    )
    parser.add_argument(
        '--output_dir',
        default='./output',
        help='Output directory to store checkpoints.'
    )
    parser.add_argument(
        '--result_dir',
        default='./results/test/Depth',
        help='Result directory to save depth images.'
    )

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    # for reproducibility
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # check dataset directory
    assert os.path.isdir(args.dataset_dir), f'{args.dataset_dir} is not a directory.'

    # load dataset
    test_dataset = SEMDataset(path=args.dataset_dir, mode="Test")

    # make results directory.
    print(f'Generate results directory : {args.result_dir}')
    if not os.path.isdir(args.result_dir):
        os.makedirs(args.result_dir, exist_ok=True)

    # init data loader
    test_loader = DataLoader(test_dataset, batch_size=1)

    model = UNet(n_channels=1, n_classes=1, bilinear=False).to('cuda')
    model.load_state_dict(torch.load(f"{args.output_dir}/checkpoint.pth"))

    transform = transforms.ToPILImage()

    for test_sample in tqdm(test_loader, desc=f"TEST"):
        # shape : (batch, 1, w, h)
        sem, filename = test_sample['sem'], test_sample['filename'][0]
        assert sem.shape[1] == 1

        # shape : (batch, 1, w, h)
        sem = sem.to('cuda', dtype=torch.float32)

        with torch.no_grad():
            # shape : (batch, 1, w, h)
            prediction = model(sem)

        img = transform(prediction[0].cpu())
        img.convert("L").save(f"{args.result_dir}/{filename}")
        


if __name__ == "__main__":
    start_time = time.time()
    main()

    end_time = time.time()
    print(f'Total runtime : {end_time - start_time} sec.')