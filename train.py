import argparse
import logging
import os
import random
import time
import json
import math

import wandb
import torch
from torch.utils.data import DataLoader
from torch import optim
from torch.optim.lr_scheduler import LinearLR
from tqdm.auto import tqdm

from data import SEMDataset
from models.unet import UNet

logger = logging.getLogger(__name__)

optimizers = {
    'Adam' : optim.Adam, 
    'AdamW' : optim.AdamW,
    'RMSprop' : optim.RMSprop,
    'SGD' : optim.SGD,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Train a U-net model.")
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for training.",
    )
    parser.add_argument(
        "--lr", 
        type=float, 
        default=1e-5, 
        help="Learning rate for training."
    )
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
        "--early_stop", 
        type=int, 
        default=5, 
        help="Number of epoch for early stopping."
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default='Adam',
        help="The name of the optmizer for training.",
        choices=['Adam', 'AdamW', 'RMSprop', 'SGD'],
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        default="linear",
        help="The scheduler type to use during training.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", 
        type=int, 
        default=1000, 
        help="Number of steps for the warmup in the lr scheduler."
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
    train_dataset = SEMDataset(path=args.dataset_dir, mode="Train")
    validation_dataset = SEMDataset(path=args.dataset_dir, mode="Validation")

    num_validation = len(validation_dataset)

    # make output directory to save checkpoints, etc.
    print(f'Generate output directory : {args.output_dir}')
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    # save configs
    config_file = os.path.join(args.output_dir, 'config.json')
    with open(config_file, 'w') as file:
        json.dump(vars(args), file)
    
    # set logger config
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )

    logging_output_file = os.path.join(args.output_dir, "output.log")
    file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    file_handler = logging.FileHandler(logging_output_file)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    # set logger until here

    # init data loader
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size)
    validation_loader = DataLoader(validation_dataset, batch_size=args.batch_size)

    logger.info(f'# train samples      : {len(train_loader)} * {args.batch_size} = {len(train_loader) * args.batch_size}')
    logger.info(f'# validation samples : {len(validation_loader)} * {args.batch_size} = {len(validation_loader) * args.batch_size}')

    # init model
    logger.info('Init model....')
    model = UNet(n_channels=1, n_classes=1, bilinear=False).to('cuda')

    # print # trained params
    total_param = 0
    for name, param in model.named_parameters():
        total_param += param.numel()
    logger.info(f'# trained param      : {total_param}')
    
    # init wandb
    db = wandb.init(project="U-net", resume="allow", anonymous='must')
    db.config.update(dict(
        epochs=args.epochs, 
        batch_size=args.batch_size,
        learning_rate=args.lr
    ))

    optimizer_class = optimizers.get(args.optimizer, None)
    assert optimizer_class, f'Optimizer {args.optimizer} not supported.'
    optimizer = optimizer_class(model.parameters(), lr=args.lr, weight_decay=1e-8)
    # mse loss?
    # criterion = torch.nn.CrossEntropyLoss()
    criterion = torch.nn.MSELoss()
    rmse_criterion = torch.nn.MSELoss(reduction='sum')
    global_step = 0
    best_eval_loss = float('inf')
    early_stop_count = 0

    # TRAIN!
    for epoch in range(1, args.epochs+1):
        model.train()
        epoch_loss = 0
        for train_sample in tqdm(train_loader, desc=f"TRAIN EPOCH {epoch}"):
            # shape : (batch, 1, w, h)
            sem, depth = train_sample['sem'], train_sample['depth']
            assert sem.shape[1] == 1
            assert depth.shape[1] == 1

            # shape : (batch, 1, w, h)
            sem = sem.to('cuda', dtype=torch.float32)
            depth = depth.to('cuda', dtype=torch.float32)

            # shape : (batch, 1, w, h)
            prediction = model(sem)

            optimizer.zero_grad()
            loss = criterion(prediction, depth)
            loss.backward()
            optimizer.step()

            global_step += 1
            epoch_loss += loss.item()

            db.log({'train loss': loss.item()})

        # EVALUATE !
        model.eval()
        # log weights
        histograms = {}
        for name, param in model.named_parameters():
            name = name.replace('/', '.')
            histograms['Weights/' + name] = wandb.Histogram(param.data.cpu())
            histograms['Gradients/' + name] = wandb.Histogram(param.grad.data.cpu())

        eval_loss = 0
        for eval_sample in tqdm(validation_loader, desc=f"EVAL"):
            # shape : (batch, 1, w, h)
            sem, depth = eval_sample['sem'], eval_sample['depth']
            assert sem.shape[1] == 1
            assert depth.shape[1] == 1

            # shape : (batch, 1, w, h)
            sem = sem.to('cuda', dtype=torch.float32)
            depth = depth.to('cuda', dtype=torch.float32)

            with torch.no_grad():
                # shape : (batch, 1, w, h)
                prediction = model(sem)
                # calculate the squared error (mean x)
                loss = rmse_criterion(prediction, depth)
                eval_loss += loss.item()
        
        # calculate rmse
        eval_loss = math.sqrt(eval_loss / num_validation)

        db.log({
            'learning rate': optimizer.param_groups[0]['lr'],
            'validation RMSE': eval_loss,
            'sem': wandb.Image(sem[0].cpu()),
            'masks': {
                'depth': wandb.Image(depth[0].float().cpu()),
                'prediction': wandb.Image(prediction[0].float().cpu()),
            },
            'step': global_step,
            'epoch': epoch,
            **histograms
        })

        # for early stopping
        if best_eval_loss > eval_loss:
            best_eval_loss = eval_loss
            early_stop_count = 0

            print('Best model! Saving model...')
            output_file = os.path.join(args.output_dir, f'checkpoint.pth')
            # if file exits, remove (overwrite)
            if os.path.isfile(output_file):
                os.remove(output_file)
            # save model
            torch.save(model.state_dict(), output_file)
        else:
            early_stop_count += 1
            print(f'Early stop : {early_stop_count} / {args.early_stop}')

            if early_stop_count > args.early_stop:
                print('Early Stop.')
                break

        print(f'Best RMSE loss : {best_eval_loss}')







if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print(f'Total runtime : {end_time - start_time} sec.')