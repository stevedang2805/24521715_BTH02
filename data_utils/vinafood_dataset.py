import os
import random

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def collate_fn(items: list[dict]) -> dict[str, torch.Tensor]:
    return {
        "image": torch.stack([item["image"] for item in items], dim=0),  # (bs, 3, H, W)
        "label": torch.tensor([item["label"] for item in items], dtype=torch.long),  # (bs, )
    }


def get_classes(root: str) -> list[str]:
    classes = sorted(
        d for d in os.listdir(root)
        if os.path.isdir(os.path.join(root, d)) and not d.startswith(".")
    )
    if not classes:
        raise FileNotFoundError(f"Không tìm thấy thư mục lớp nào trong {root}")
    return classes


def load_samples(root: str, classes: list[str]) -> list[tuple[str, int]]:
    # root/<tên lớp>/<ảnh>
    samples = []
    for label, class_name in enumerate(classes):
        class_dir = os.path.join(root, class_name)
        if not os.path.isdir(class_dir):
            continue
        for dirpath, _, filenames in os.walk(class_dir):
            for filename in sorted(filenames):
                if filename.lower().endswith(IMAGE_EXTENSIONS):
                    samples.append((os.path.join(dirpath, filename), label))
    return samples


def split_samples(samples: list, dev_ratio: float = 0.1, seed: int = 42) -> tuple[list, list]:
    # tách dev từ train theo từng lớp, dùng dev để chọn checkpoint thay vì dùng test
    rng = random.Random(seed)
    by_label: dict[int, list] = {}
    for sample in samples:
        by_label.setdefault(sample[1], []).append(sample)

    train, dev = [], []
    for label in sorted(by_label):
        items = by_label[label][:]
        rng.shuffle(items)
        n_dev = max(1, int(len(items) * dev_ratio))
        dev.extend(items[:n_dev])
        train.extend(items[n_dev:])
    return train, dev


def build_transform(image_size: int = 224, train: bool = False,
                    mean=IMAGENET_MEAN, std=IMAGENET_STD) -> transforms.Compose:
    steps = [transforms.Resize((image_size, image_size))]
    if train:
        steps.append(transforms.RandomHorizontalFlip())
    steps += [transforms.ToTensor(), transforms.Normalize(mean, std)]
    return transforms.Compose(steps)


class VinaFood21Dataset(Dataset):
    def __init__(self, samples: list[tuple[str, int]], transform):
        self._samples = samples
        self._transform = transform

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> dict:
        path, label = self._samples[idx]
        image = Image.open(path).convert("RGB")  # có ảnh xám / RGBA trong dataset
        return {
            "image": self._transform(image),  # (3, H, W)
            "label": label,
        }


def build_vinafood_datasets(data_dir: str, image_size: int = 224, dev_ratio: float = 0.1,
                            seed: int = 42, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    train_root = os.path.join(data_dir, "train")
    test_root = os.path.join(data_dir, "test")
    classes = get_classes(train_root)

    train_samples, dev_samples = split_samples(load_samples(train_root, classes), dev_ratio, seed)
    test_samples = load_samples(test_root, classes)

    train_tf = build_transform(image_size, train=True, mean=mean, std=std)
    eval_tf = build_transform(image_size, train=False, mean=mean, std=std)

    return (
        VinaFood21Dataset(train_samples, train_tf),
        VinaFood21Dataset(dev_samples, eval_tf),
        VinaFood21Dataset(test_samples, eval_tf),
        classes,
    )
