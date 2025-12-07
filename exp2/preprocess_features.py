import os
import argparse
import tarfile
import requests
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms, datasets
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from PIL import Image


# Helper to avoid re-downloading
def download_file(url, dest_path):
    if os.path.exists(dest_path):
        print(f"File {os.path.basename(dest_path)} already exists. Skipping download.")
        return
    print(f"Downloading {url} to {dest_path}...")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(dest_path, "wb") as f:
            for chunk in tqdm(r.iter_content(chunk_size=8192)):
                f.write(chunk)


# ======================================================================================
# Custom Dataset Classes
# ======================================================================================


class StanfordDogsCustom(Dataset):
    """Custom Dataset for Stanford Dogs, as it's not in torchvision."""

    def __init__(
        self, root: str, split: str = "train", transform=None, download: bool = False
    ):
        from scipy.io import loadmat

        self.root = os.path.expanduser(root)
        self.split = split
        self.transform = transform
        self.base_folder = os.path.join(self.root, "StanfordDogs")
        self.images_folder = os.path.join(self.base_folder, "Images")

        urls = {
            "images": "http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar",
            "annotations": "http://vision.stanford.edu/aditya86/ImageNetDogs/annotation.tar",
            "lists": "http://vision.stanford.edu/aditya86/ImageNetDogs/lists.tar",
        }

        if download:
            os.makedirs(self.base_folder, exist_ok=True)
            for key, url in urls.items():
                filename = url.split("/")[-1]
                filepath = os.path.join(self.base_folder, filename)
                download_file(url, filepath)
                print(f"Extracting {filename}...")
                with tarfile.open(filepath, "r") as tar:
                    tar.extractall(path=self.base_folder)
                print("Extraction complete.")

        if not os.path.isdir(self.images_folder):
            raise RuntimeError(
                "Dataset not found. You can use download=True to download it."
            )

        # Load .mat files
        train_list_mat = loadmat(os.path.join(self.base_folder, "train_list.mat"))
        test_list_mat = loadmat(os.path.join(self.base_folder, "test_list.mat"))

        full_train_paths = [f[0][0] for f in train_list_mat["file_list"]]
        full_train_labels = [l[0] - 1 for l in train_list_mat["labels"]]  # 0-indexed  # noqa: E741

        self.test_paths = [f[0][0] for f in test_list_mat["file_list"]]
        self.test_labels = [l[0] - 1 for l in test_list_mat["labels"]]  # noqa: E741

        # Create a stratified val split from the full training data
        train_indices, val_indices = train_test_split(
            list(range(len(full_train_paths))),
            test_size=0.2,
            random_state=42,
            stratify=full_train_labels,
        )

        if self.split == "train":
            self.paths = [full_train_paths[i] for i in train_indices]
            self.labels = [full_train_labels[i] for i in train_indices]
        elif self.split == "val":
            self.paths = [full_train_paths[i] for i in val_indices]
            self.labels = [full_train_labels[i] for i in val_indices]
        else:  # test
            self.paths = self.test_paths
            self.labels = self.test_labels

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img_path = os.path.join(self.images_folder, self.paths[idx])
        label = self.labels[idx]
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, label


# ======================================================================================
# Main Script Logic
# ======================================================================================


def compute_dataset_stats(dataset: Dataset) -> tuple[torch.Tensor, torch.Tensor]:
    """Computes mean and std for a given dataset."""
    print("Calculating dataset statistics (mean and std)...")
    # Use a simple ToTensor transform to get pixel values in [0, 1]
    loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=4)

    mean = 0.0
    std = 0.0
    num_samples = 0.0

    for images, _ in tqdm(loader, desc="Stat Calc"):
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        num_samples += batch_samples

    mean /= num_samples
    std /= num_samples

    print(f"Computed Mean: {mean}")
    print(f"Computed Std: {std}")
    return mean, std


def get_backbone(device: torch.device) -> nn.Module:
    """Loads a pre-trained ResNet-50 model and removes the final fully-connected layer."""
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    backbone = nn.Sequential(*list(model.children())[:-1])
    backbone.to(device)
    backbone.eval()
    print("ResNet-50 backbone loaded.")
    return backbone


class DatasetTransformer(Dataset):
    """A wrapper to apply a specific transform to a dataset."""

    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, index):
        # For custom datasets that handle transforms internally during __getitem__
        if hasattr(self.dataset, "transform"):
            self.dataset.transform = self.transform
            return self.dataset[index]

        # For torchvision datasets
        image, label = self.dataset[index]
        if self.transform:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.dataset)


def get_datasets(dataset_name: str, data_path: str) -> tuple[Dataset, Dataset, Dataset]:
    """Factory function to get train, val, and test datasets."""
    download = True  # Always enable download for convenience
    if dataset_name == "aircraft":
        print("Loading FGVC-Aircraft dataset...")
        train_dataset = datasets.FGVCAircraft(
            root=data_path, split="train", download=download
        )
        val_dataset = datasets.FGVCAircraft(
            root=data_path, split="val", download=download
        )
        test_dataset = datasets.FGVCAircraft(
            root=data_path, split="test", download=download
        )
    elif dataset_name == "flowers":
        print("Loading Oxford Flowers-102 dataset...")
        train_dataset = datasets.Flowers102(
            root=data_path, split="train", download=download
        )
        val_dataset = datasets.Flowers102(
            root=data_path, split="val", download=download
        )
        test_dataset = datasets.Flowers102(
            root=data_path, split="test", download=download
        )
    elif dataset_name == "dogs-custom":
        print("Loading custom Stanford Dogs dataset...")
        train_dataset = StanfordDogsCustom(
            root=data_path, split="train", download=download
        )
        val_dataset = StanfordDogsCustom(root=data_path, split="val", download=download)
        test_dataset = StanfordDogsCustom(
            root=data_path, split="test", download=download
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    return train_dataset, val_dataset, test_dataset


def extract_and_save_features(model, loader, device, feature_path, label_path):
    all_features, all_labels = [], []
    with torch.no_grad():
        for images, labels in tqdm(
            loader, desc=f"Processing {os.path.basename(feature_path)}"
        ):
            features = model(images.to(device))
            all_features.append(torch.flatten(features, 1).cpu())
            all_labels.append(labels.cpu())
    torch.save(torch.cat(all_features), feature_path)
    torch.save(torch.cat(all_labels), label_path)
    print(f"Saved features to {feature_path}\nSaved labels to {label_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess image datasets and save ResNet-50 features."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["aircraft", "flowers", "dogs-custom"],
        help="Name of the dataset to process.",
    )
    parser.add_argument(
        "--data_path", type=str, default="./data", help="Path to download raw datasets."
    )
    parser.add_argument(
        "--features_path",
        type=str,
        default="./features",
        help="Path to save extracted features.",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=1,
        help="Epochs for augmented training set generation.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=64, help="Batch size for feature extraction."
    )
    args = parser.parse_args()

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_path = os.path.join(args.features_path, args.dataset)
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(args.data_path, exist_ok=True)
    print(f"Using device: {device}. Output will be in {output_path}")

    # Get datasets (without final transforms)
    train_data, val_data, test_data = get_datasets(args.dataset, args.data_path)

    # Compute normalization stats on the training set
    # The transform here is temporary, just for calculation
    stats_calc_dataset = DatasetTransformer(
        train_data,
        transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()]),
    )
    mean, std = compute_dataset_stats(stats_calc_dataset)

    # Define final transforms with the computed stats
    normalize_transform = transforms.Normalize(mean=mean, std=std)
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(224, scale=(0.4, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize_transform,
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize_transform,
        ]
    )

    # Get backbone model
    backbone = get_backbone(device)

    # --- Process Training Set ---
    if args.num_epochs > 1:
        print(
            f"\nProcessing training set {args.num_epochs} times with augmentations..."
        )
        aug_path = os.path.join(output_path, "augmented_train")
        os.makedirs(aug_path, exist_ok=True)
        train_dataset_transformed = DatasetTransformer(train_data, train_transform)
        train_loader = DataLoader(
            train_dataset_transformed,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )
        for epoch in range(args.num_epochs):
            print(f"--- Epoch {epoch + 1}/{args.num_epochs} ---")
            extract_and_save_features(
                backbone,
                train_loader,
                device,
                os.path.join(aug_path, f"epoch_{epoch}_features.pt"),
                os.path.join(aug_path, f"epoch_{epoch}_labels.pt"),
            )
    else:
        print("\nProcessing training set once without augmentation...")
        train_dataset_transformed = DatasetTransformer(train_data, eval_transform)
        train_loader = DataLoader(
            train_dataset_transformed,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )
        extract_and_save_features(
            backbone,
            train_loader,
            device,
            os.path.join(output_path, "train_features.pt"),
            os.path.join(output_path, "train_labels.pt"),
        )

    # --- Process Validation and Test Sets ---
    for split_name, split_data in [("validation", val_data), ("test", test_data)]:
        print(f"\nProcessing {split_name} set...")
        dataset_transformed = DatasetTransformer(split_data, eval_transform)
        loader = DataLoader(
            dataset_transformed,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )
        extract_and_save_features(
            backbone,
            loader,
            device,
            os.path.join(output_path, f"{split_name.replace('ation', '')}_features.pt"),
            os.path.join(output_path, f"{split_name.replace('ation', '')}_labels.pt"),
        )

    print("\n✅ All datasets processed successfully!")


if __name__ == "__main__":
    main()
