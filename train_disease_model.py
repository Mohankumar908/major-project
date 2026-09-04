import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from torchvision import transforms
from torchvision import datasets
from torchvision import models

from torchvision.models import ResNet18_Weights

from torch.utils.data import DataLoader
from torch.utils.data import Subset

MODEL_DIR = Path("ml_models")


def find_train_valid(data_dir):
    """
    Recursively locate the actual train/valid folders inside data_dir.

    Handles the well-known quirk of this exact Kaggle dataset: the zip
    extracts into a DOUBLY-NESTED folder, e.g.
        dataset/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train
        dataset/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/valid
    rather than dataset/train + dataset/valid directly. This searches at any
    depth for a folder literally named 'train' with a 'valid' (or 'val')
    sibling, so you can point --data straight at wherever you extracted the
    zip without manually moving folders around.
    """
    data_dir = Path(data_dir)
    for train_dir in sorted(data_dir.rglob("train")):
        if not train_dir.is_dir():
            continue
        for val_name in ("valid", "val"):
            val_dir = train_dir.parent / val_name
            if val_dir.is_dir():
                return train_dir, val_dir
    return None, None


def build_splits(data_dir):
    """
    Build train/val splits WITHOUT leaking near-duplicate augmented images
    between the two sets, and WITH class balance preserved (stratified).

    Two modes:
    1) If a train/valid (or train/val) pair is found anywhere under
       data_dir (see find_train_valid) — as the Kaggle "New Plant Diseases
       Dataset (Augmented)" download provides — use THOSE folders directly.
       Do not re-split them yourself; the dataset's own split already keeps
       augmented siblings on one side.
    2) If data_dir is a single flat folder of class subfolders (your own
       collected Green Gram photos, for example), do a STRATIFIED split so
       every class is represented proportionally in val, instead of the
       previous plain random_split (which can leave some classes with too
       few/zero val samples -> noisy, unstable "best model" selection).
    """
    data_dir = Path(data_dir)
    train_dir, val_dir = find_train_valid(data_dir)

    if train_dir and val_dir:
        print(f"\nFound dataset's own split:\n  train: {train_dir}\n  valid: {val_dir}")
        return "prebuilt", train_dir, val_dir

    print(f"\nNo train/valid folders found anywhere under {data_dir} — "
          f"treating it as a flat folder of class subfolders and doing a "
          f"stratified split instead.")
    return "flat", data_dir, data_dir


def stratified_indices(full_dataset, val_fraction=0.2, seed=42):
    """Per-class split so every class contributes ~val_fraction to val."""
    rng = np.random.RandomState(seed)
    targets = np.array(full_dataset.targets)
    train_idx, val_idx = [], []
    for cls in np.unique(targets):
        cls_idx = np.where(targets == cls)[0]
        rng.shuffle(cls_idx)
        n_val = max(1, int(len(cls_idx) * val_fraction))
        val_idx.extend(cls_idx[:n_val])
        train_idx.extend(cls_idx[n_val:])
    return train_idx, val_idx


def train(data_dir, epochs=30, batch_size=32, lr=1e-4):

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(25),

        transforms.ColorJitter(
            brightness=0.3,
            contrast=0.3,
            saturation=0.3
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),

        transforms.ToTensor(),

        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])

    mode, train_src, val_src = build_splits(data_dir)

    if mode == "prebuilt":
        # Dataset's own train/valid folders — no leakage, no re-split needed.
        train_dataset = datasets.ImageFolder(train_src, transform=train_transform)
        val_dataset   = datasets.ImageFolder(val_src,   transform=val_transform)
        class_names   = train_dataset.classes
        assert class_names == val_dataset.classes, \
            "train/valid class folders don't match! Check dataset structure."
        train_targets = train_dataset.targets
    else:
        # Flat folder -> stratified split by class, same underlying files
        # loaded twice with different transforms (train aug vs clean val).
        base_for_split = datasets.ImageFolder(train_src)
        class_names = base_for_split.classes
        train_indices, val_indices = stratified_indices(base_for_split)

        train_dataset = Subset(
            datasets.ImageFolder(train_src, transform=train_transform),
            train_indices
        )
        val_dataset = Subset(
            datasets.ImageFolder(val_src, transform=val_transform),
            val_indices
        )
        train_targets = [base_for_split.targets[i] for i in train_indices]

    print("\nClasses Found:", len(class_names))
    print("Train Images:", len(train_dataset), " Val Images:", len(val_dataset))

    # Per-class counts -> class-balanced sampling so rare classes aren't
    # drowned out (a common cause of the model defaulting to majority/
    # "healthy"-like classes).
    counts = Counter(train_targets)
    class_weights = {c: 1.0 / n for c, n in counts.items()}
    sample_weights = [class_weights[t] for t in train_targets]
    sampler = torch.utils.data.WeightedRandomSampler(
        sample_weights, num_samples=len(sample_weights), replacement=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=4
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )

    num_classes = len(class_names)

    print("\nLoading ResNet18...")

    weights = ResNet18_Weights.DEFAULT

    model = models.resnet18(
        weights=weights
    )

    for param in model.parameters():
        param.requires_grad = False

    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(
            model.fc.in_features,
            num_classes
        )
    )

    model = model.to(device)

    # Label smoothing: your Phase-2 model hit 99.7% "val" accuracy with
    # overconfident softmax outputs, which is a symptom of overfitting to
    # near-duplicate augmented images. Smoothing discourages the model from
    # driving probabilities to 0/1 and tends to generalize better to
    # real-world (non-studio) photos.
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    optimizer_fc = torch.optim.Adam(
        model.fc.parameters(),
        lr=lr
    )

    optimizer_all = torch.optim.Adam(
        model.parameters(),
        lr=lr * 0.1,
        weight_decay=1e-4
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_all,
        mode='max',
        patience=2,
        factor=0.5
    )

    best_acc = 0
    patience = 5
    early_stop_counter = 0

    print("\nTraining Started\n")

    for epoch in range(epochs):

        if epoch == 3:
            print("\nUnfreezing entire network...\n")

            for param in model.parameters():
                param.requires_grad = True

        optimizer = (
            optimizer_fc
            if epoch < 3
            else optimizer_all
        )

        model.train()

        running_loss = 0

        for images, labels in train_loader:

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

        model.eval()

        total = 0
        all_preds, all_labels = [], []

        with torch.no_grad():

            for images, labels in val_loader:

                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)

                _, preds = torch.max(
                    outputs,
                    1
                )

                total += labels.size(0)
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        accuracy = 100 * (all_preds == all_labels).sum() / total

        # Macro-averaged per-class recall ("balanced accuracy"). Using this
        # instead of raw accuracy to pick the "best" checkpoint stops the
        # model from being rewarded for nailing common/easy classes while
        # quietly failing rare ones (e.g. always guessing "healthy").
        per_class_recall = []
        for cls in range(num_classes):
            mask = all_labels == cls
            if mask.sum() > 0:
                per_class_recall.append((all_preds[mask] == cls).mean())
        balanced_acc = 100 * float(np.mean(per_class_recall))

        avg_loss = (
            running_loss /
            len(train_loader)
        )

        print(
            f"Epoch [{epoch+1}/{epochs}] "
            f"Loss: {avg_loss:.4f} "
            f"Val Acc: {accuracy:.2f}% "
            f"Balanced Acc: {balanced_acc:.2f}%"
        )

        if epoch >= 3:
            scheduler.step(balanced_acc)

        if balanced_acc > best_acc:

            best_acc = balanced_acc

            early_stop_counter = 0

            MODEL_DIR.mkdir(
                exist_ok=True
            )

            torch.save(
                model.state_dict(),
                MODEL_DIR /
                "disease_model_best.pth"
            )

        else:
            early_stop_counter += 1

        if early_stop_counter >= patience:

            print(
                "\nEarly stopping triggered."
            )

            break

    print(
        f"\nBest Balanced Accuracy (macro avg per-class recall): "
        f"{best_acc:.2f}%"
    )
    print(
        "Note: this is macro-balanced accuracy, a more honest number than "
        "raw accuracy on an imbalanced validation set. Compare it against "
        "accuracy on a small hand-labelled set of REAL camera photos before "
        "trusting it for deployment — validation accuracy on the same "
        "dataset's own split will always look better than real-world "
        "performance."
    )

    model.load_state_dict(
        torch.load(
            MODEL_DIR /
            "disease_model_best.pth",
            map_location=device
        )
    )

    torch.save(
        model.state_dict(),
        MODEL_DIR /
        "disease_model.pth"
    )

    with open(
        MODEL_DIR /
        "disease_classes.txt",
        "w"
    ) as f:

        for cls in class_names:
            f.write(cls + "\n")

    with open(
        MODEL_DIR /
        "disease_model_accuracy.txt",
        "w"
    ) as f:

        f.write(
            str(
                round(best_acc, 2)
            )
        )

    print("\nModel Saved Successfully")
    print("Classes:", len(class_names))
    print("Best Accuracy:", best_acc)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data",
        default="dataset",
        help="Path to your dataset folder (e.g. the 'dataset' folder next to "
             "agriculture/, ml_models/, etc. — can point straight at the "
             "extracted Kaggle zip, nested folders are auto-detected)"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=30
    )

    parser.add_argument(
        "--batch",
        type=int,
        default=32
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4
    )

    args = parser.parse_args()

    train(
        args.data,
        args.epochs,
        args.batch,
        args.lr
    )