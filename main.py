# -*- coding: utf-8 -*-
#!/home/gezhipeng/anaconda3/envs/gtorch/bin/python
"""
@author: gehipeng @ 20230411
@file: main.py
@brief: main
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.models as models
import glob
import time
import os
import numpy as np
import argparse
import importlib
from src.utils import utils
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import logging
from functools import wraps

logging.getLogger().setLevel(logging.INFO)
from PIL import Image
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau


def log_fun(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        logging.info(f"{' ' + 'start ' + func.__name__ + ' ':=^100}")
        res = func(*args, **kwargs)
        logging.info(f"{' ' + 'end ' + func.__name__ + ' ':=^100}")
        return res

    return wrapper


@log_fun
def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch Training")
    parser.add_argument(
        "--config_path",
        type=str,
        default="config/17flowers_resnet18.py",
        help="config path",
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="local rank")
    parser.add_argument("--test", action="store_true", help="test")
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images"
        'or a single glob pattern such as "directory/*.jpg"',
    )
    parser.add_argument("--checkpoint", type=str, help="checkpoint path")
    args = parser.parse_args()
    return args


def init_state(args, train=True):
    config_path = args.config_path
    assert os.path.exists(config_path), f"config path: {config_path} not exists"
    config_spec = importlib.util.spec_from_file_location("config", config_path)
    config_module = importlib.util.module_from_spec(config_spec)
    config_spec.loader.exec_module(config_module)
    config = config_module.config
    if not train:
        return config
    torch.manual_seed(config.SEED)
    torch.cuda.manual_seed_all(config.SEED)
    torch.backends.cudnn.deterministic = True
    dist.init_process_group(
        backend="nccl",  # communication backend
        world_size=torch.cuda.device_count(),  # number of processes: gpus num in current node
        rank=args.local_rank,  # rank of the current process: local_rank in current node
    )
    torch.cuda.set_device(args.local_rank)
    return config


@log_fun
def get_model(config, args):
    model_config = config.MODEL_CONFIG
    model_name = model_config["name"]
    if model_name == "resnet18":
        model = models.resnet18(weights=model_config["weights"])
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, model_config["num_classes"])
    elif model_name == "resnet50":
        model = models.resnet50(weights=model_config["weights"])
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, model_config["num_classes"])
    else:
        raise NotImplementedError
    if not args.test:
        return model.to(args.local_rank)
    else:
        return model.to(device=0)


@log_fun
def get_latest_checkpoint_path(config, resume=False):
    if not resume:
        return None
    model_config = config.MODEL_CONFIG
    model_name = model_config["name"]
    checkpoint_dirs = glob.glob(
        os.path.join(
            config.CHECKPOINT_DIR, f"{os.path.basename(config.DATA_DIR)}_{model_name}_*"
        )
    )
    if len(checkpoint_dirs) > 0:
        checkpoint_dirs.sort()
        chekpoint_dir = checkpoint_dirs[-1]
        saved_models = glob.glob(os.path.join(chekpoint_dir, "model_*.ckpt"))
        if len(saved_models) == 0:
            logging.warning(f"no checkpoint found in {chekpoint_dir}")
            return None
        else:
            saved_models.sort()
            checkpoint_path = saved_models[-1]
            logging.info(f"load model from: {checkpoint_path}")
            return checkpoint_path
    else:
        logging.warning(f"no checkpoint dir found in {config.CHECKPOINT_DIR}")
        return None


@log_fun
def load_state_dict(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    return checkpoint


def train(config, args):
    checkpoint_dir_base = config.CHECKPOINT_DIR
    model_config = config.MODEL_CONFIG
    model_name = model_config["name"]
    train_transforms = utils.get_transforms(config.TRANSFORMS["train"])
    val_transforms = utils.get_transforms(config.TRANSFORMS["val"])
    train_set = datasets.ImageFolder(
        root=os.path.join(config.DATA_DIR, "train"), transform=train_transforms
    )
    val_set = datasets.ImageFolder(
        root=os.path.join(config.DATA_DIR, "val"), transform=val_transforms
    )
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
    train_loader = DataLoader(
        train_set,
        batch_size=config.BATCH_SIZE,
        sampler=train_sampler,
        num_workers=config.NUM_WORKERS,
    )
    val_loader = DataLoader(
        val_set, batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS
    )
    model = get_model(config, args)

    start_num_epoch = 0
    checkpoint = None
    checkpoint_dir = None
    resume_path = get_latest_checkpoint_path(config, config.RESUME)
    if resume_path:
        if dist.get_rank() == 0:
            checkpoint_dir = os.path.dirname(resume_path)
            checkpoint = load_state_dict(resume_path)
            model.load_state_dict(checkpoint["model_state_dict"])
            start_num_epoch = checkpoint["epoch"] + 1
            checkpoint_dir = os.path.dirname(resume_path)
    elif dist.get_rank() == 0:
        logging.info(f"train from scratch")
        checkpoint_dir = os.path.join(
            checkpoint_dir_base,
            "{}_{}_{}".format(
                os.path.basename(config.DATA_DIR),
                model_name,
                utils.get_current_time_str_simple(),
            ),
        )
        os.makedirs(checkpoint_dir, exist_ok=True)
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[args.local_rank], output_device=args.local_rank
    )
    writer = SummaryWriter(f"{checkpoint_dir}/logs")
    # optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
    # scheduler = StepLR(optimizer, step_size=5, gamma=0.8)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=5, verbose=True
    )
    if checkpoint is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    criterion = nn.CrossEntropyLoss()
    total_steps = len(train_loader)
    num_steps_to_show = 1
    for epoch in range(start_num_epoch, config.NUM_EPOCHS):
        model.train()
        for i, (images, labels) in enumerate(train_loader):
            t0 = time.time()
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            writer.add_scalar("loss", loss.item(), len(train_loader) * epoch + i)
            writer.add_scalar(
                "lr", optimizer.param_groups[0]["lr"], len(train_loader) * epoch + i
            )
            t1 = time.time()
            if (i + 1) % num_steps_to_show == 0 and args.local_rank == 0:
                print(
                    "RANK {}: Epoch [{}/{}], Step [{}/{}], Loss: {:.4f} | {:.3f} images/s | {:.2f}s/iter".format(
                        args.local_rank,
                        epoch + 1,
                        config.NUM_EPOCHS,
                        i + 1,
                        total_steps,
                        loss.item(),
                        len(images) / (time.time() - t0 + 1e-6),
                        t1 - t0,
                    )
                )
        if dist.get_rank() == 0:
            model.eval()
            with torch.no_grad():
                correct = 0
                total = 0
                val_loss_scalar = 0.0
                for images, labels in val_loader:
                    images = images.cuda(non_blocking=True)
                    labels = labels.cuda(non_blocking=True)
                    outputs = model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

                    val_loss = criterion(outputs, labels)
                    val_loss_scalar += val_loss.item()
                scheduler.step(val_loss_scalar / len(val_loader))
                acc = correct / total
                print(
                    "Accuracy of the model on the {} validation images: {:.3f} %".format(
                        total, 100 * acc
                    )
                )
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.module.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": loss.item(),
                },
                os.path.join(
                    checkpoint_dir,
                    "model_{}.ckpt".format(
                        str(epoch + 1).zfill(len(str(config.NUM_EPOCHS)))
                    ),
                ),
            )

        # scheduler.step()
    writer.close()


def test(config, args):
    checkpoint_path = args.checkpoint
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"checkpoint path: {checkpoint_path} not exists")
    model_config = config.MODEL_CONFIG
    model_name = model_config["name"]
    model = get_model(config, args)
    checkpoint = load_state_dict(checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    transform = utils.get_transforms(config.TRANSFORMS["val"])
    for img_path in args.input:
        img = utils.load_image(img_path)
        img_pil = Image.fromarray(img)
        tensor = transform(img_pil).unsqueeze(0).cuda(0)
        model.eval()
        with torch.no_grad():
            outputs = model(tensor)
            confidence_distribution = F.softmax(outputs, dim=-1)
            score, predicted = torch.max(confidence_distribution, 1)
            print(
                f"img_path: {img_path}, predicted: {predicted.item()}, confidence: {score.item():.2f}"
            )


def main():
    args = parse_args()
    if args.test:
        config = init_state(args, train=False)
        test(config, args)
    else:
        config = init_state(args, train=True)
        train(config, args)
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
