import os

import torch
from torch import optim

from data_utils.vinafood_dataset import build_vinafood_datasets
from models.googlenet import GoogLeNet
from utils.cli import build_dataloaders, parse_args
from utils.engine import fit, report_test, set_seed

CHECKPOINT_DIR = "./checkpoints/assignment-2"

if __name__ == "__main__":
    args = parse_args("Bài 2 - GoogLeNet", lr=1e-4, batch_size=32, epochs=30, patience=5).parse_args()
    set_seed(args.seed)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_dataset, dev_dataset, test_dataset, classes = build_vinafood_datasets(
        args.data_dir, image_size=args.image_size, dev_ratio=args.dev_ratio, seed=args.seed
    )
    print(f"{len(classes)} lớp | train {len(train_dataset)} | dev {len(dev_dataset)} | test {len(test_dataset)}")
    train_dataloader, dev_dataloader, test_dataloader = build_dataloaders(
        train_dataset, dev_dataset, test_dataset, args.batch_size, args.num_workers, device == "cuda"
    )

    model = GoogLeNet(num_classes=len(classes)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    checkpoint_path = os.path.join(CHECKPOINT_DIR, "best_model.pth")
    history = fit(
        model, train_dataloader, dev_dataloader, optimizer, device,
        epochs=args.epochs, patience=args.patience,
        save_fn=lambda: torch.save(model.state_dict(), checkpoint_path),
        use_amp=not args.no_amp,
    )

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    report_test(model, test_dataloader, device, classes, CHECKPOINT_DIR, history)
