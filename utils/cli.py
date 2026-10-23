import argparse

from torch.utils.data import DataLoader

from data_utils.vinafood_dataset import collate_fn


def parse_args(description: str, lr: float, batch_size: int, epochs: int, patience: int):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--data_dir", default="./dataset/VinaFood21")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=batch_size)
    parser.add_argument("--epochs", type=int, default=epochs)
    parser.add_argument("--patience", type=int, default=patience)
    parser.add_argument("--lr", type=float, default=lr)
    parser.add_argument("--dev_ratio", type=float, default=0.1)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_amp", action="store_true")
    return parser


def build_dataloaders(train_dataset, dev_dataset, test_dataset, batch_size: int, num_workers: int, pin_memory: bool):
    common = dict(batch_size=batch_size, collate_fn=collate_fn,
                  num_workers=num_workers, pin_memory=pin_memory)
    return (
        DataLoader(train_dataset, shuffle=True, **common),
        DataLoader(dev_dataset, shuffle=False, **common),
        DataLoader(test_dataset, shuffle=False, **common),
    )
