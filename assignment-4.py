import os

import torch
from torch import optim
from transformers import AutoImageProcessor, AutoModelForImageClassification

from data_utils.vinafood_dataset import build_vinafood_datasets
from utils.cli import build_dataloaders, parse_args
from utils.engine import fit, report_test, set_seed

CHECKPOINT_DIR = "./checkpoints/assignment-4"

if __name__ == "__main__":
    parser = parse_args("Bài 4 - Fine-tune ResNet50 (HuggingFace)", lr=1e-4, batch_size=32, epochs=10, patience=3)
    parser.add_argument("--model_name", default="microsoft/resnet-50")
    args = parser.parse_args()
    set_seed(args.seed)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # chuẩn hoá ảnh theo mean/std của model pretrained
    processor = AutoImageProcessor.from_pretrained(args.model_name)
    train_dataset, dev_dataset, test_dataset, classes = build_vinafood_datasets(
        args.data_dir, image_size=args.image_size, dev_ratio=args.dev_ratio, seed=args.seed,
        mean=processor.image_mean, std=processor.image_std,
    )
    print(f"{len(classes)} lớp | train {len(train_dataset)} | dev {len(dev_dataset)} | test {len(test_dataset)}")
    train_dataloader, dev_dataloader, test_dataloader = build_dataloaders(
        train_dataset, dev_dataset, test_dataset, args.batch_size, args.num_workers, device == "cuda"
    )

    # thay head 1000 lớp ImageNet bằng head 21 lớp
    model = AutoModelForImageClassification.from_pretrained(
        args.model_name,
        num_labels=len(classes),
        id2label={i: c for i, c in enumerate(classes)},
        label2id={c: i for i, c in enumerate(classes)},
        ignore_mismatched_sizes=True,
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_dir = os.path.join(CHECKPOINT_DIR, "best_model")
    history = fit(
        model, train_dataloader, dev_dataloader, optimizer, device,
        epochs=args.epochs, patience=args.patience,
        save_fn=lambda: model.save_pretrained(best_dir),
        use_amp=not args.no_amp,
    )

    model = AutoModelForImageClassification.from_pretrained(best_dir).to(device)
    report_test(model, test_dataloader, device, classes, CHECKPOINT_DIR, history)
