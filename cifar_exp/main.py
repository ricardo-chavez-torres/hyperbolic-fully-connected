from pathlib import Path
import sys
import time
import math
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import Subset, DataLoader
import wandb
from geoopt import ManifoldParameter
from geoopt.optim import RiemannianSGD

parent_dir = Path(__file__).parent
sys.path.insert(0, str(parent_dir.parent))
from layers import lorentz_resnet18, Lorentz


def get_param_groups(model, lr_manifold, weight_decay_manifold, verbose=False):
    no_decay = ["scale"]
    k_params = ["manifold.k"]

    # Group 0: standard params with weight decay
    group0_params = []
    group0_names = []
    # Group 1: ManifoldParameters
    group1_params = []
    group1_names = []
    # Group 2: k parameters (no weight decay)
    group2_params = []
    group2_names = []

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if any(nd in n for nd in k_params):
            group2_params.append(p)
            group2_names.append(n)
        elif isinstance(p, ManifoldParameter):
            group1_params.append(p)
            group1_names.append(n)
        elif not any(nd in n for nd in no_decay):
            group0_params.append(p)
            group0_names.append(n)
        else:
            # This would be params matching no_decay but not other conditions
            if verbose:
                print(f"  WARNING: param {n} excluded from all groups (no weight decay)")

    if verbose:
        print(f"\n--- Parameter Groups ---")
        print(f"Group 0 (standard, with WD): {len(group0_params)} params")
        gamma_params = [n for n in group0_names if 'gamma' in n]
        print(f"  gamma params in group 0: {gamma_params}")
        print(f"Group 1 (ManifoldParam, reduced LR): {len(group1_params)} params")
        print(f"Group 2 (k params, no WD): {len(group2_params)} params")

    parameters = [
        {"params": group0_params},
        {"params": group1_params, 'lr': lr_manifold, "weight_decay": weight_decay_manifold},
        {"params": group2_params, "weight_decay": 0, "lr": 1e-4}
    ]

    return parameters

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None, device='cuda'):
    """
    Load a checkpoint and restore model, optimizer, and scheduler states.

    Returns:
        start_epoch: The epoch to resume from
        checkpoint: The full checkpoint dict for inspection
    """
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Handle compiled models (state dict keys may have '_orig_mod.' prefix)
    state_dict = checkpoint['model_state_dict']

    # Try loading directly first
    try:
        model.load_state_dict(state_dict)
    except RuntimeError:
        # If model is compiled, keys might have _orig_mod prefix
        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = k.replace('_orig_mod.', '')
            new_state_dict[new_key] = v
        model.load_state_dict(new_state_dict)

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    start_epoch = checkpoint.get('epoch', 0)
    print(f"Loaded checkpoint from epoch {start_epoch}")
    print(f"  Val loss: {checkpoint.get('val_loss', 'N/A')}")
    print(f"  Val acc: {checkpoint.get('val_acc', 'N/A')}")

    return start_epoch, checkpoint


def get_dataloaders(
    batch_size,
    data_dir,
    val_fraction=0.1,
    train_subset_fraction=1.0,
    seed=42,
):
    """
    Create CIFAR-10 train/val/test dataloaders.

    Args:
        batch_size: Batch size for all loaders
        data_dir: Directory to store/load CIFAR data
        val_fraction: Fraction of training set to use for validation (default 10%)
        train_subset_fraction: Fraction of training set to use (after val split) for faster sweeps
        seed: Random seed for reproducible splits

    Returns:
        train_loader, val_loader, test_loader
    """
    mean = (0.5074, 0.4867, 0.4411)
    std = (0.267, 0.256, 0.276)

    train_transform = torchvision.transforms.Compose([
        torchvision.transforms.RandomCrop(32, padding=4),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean, std)
    ])

    val_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean, std)
    ])

    # Load full training set (will be split into train/val)
    full_trainset = torchvision.datasets.CIFAR10(
        data_dir, train=True, download=True, transform=train_transform
    )

    # For validation, we need the same data but without augmentation
    full_trainset_val = torchvision.datasets.CIFAR10(
        data_dir, train=True, download=True, transform=val_transform
    )

    # Test set is completely separate
    testset = torchvision.datasets.CIFAR10(
        data_dir, train=False, download=True, transform=val_transform
    )

    # Create reproducible train/val split
    num_train_full = len(full_trainset)
    indices = list(range(num_train_full))

    rng = np.random.RandomState(seed)
    rng.shuffle(indices)

    val_size = int(num_train_full * val_fraction)
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    # Optionally use only a subset of training data (for faster sweeps)
    if train_subset_fraction < 1.0:
        num_train_subset = int(len(train_indices) * train_subset_fraction)
        train_indices = train_indices[:num_train_subset]

    train_subset = Subset(full_trainset, train_indices)
    val_subset = Subset(full_trainset_val, val_indices)

    train_loader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=True,
        num_workers=2, pin_memory=True
    )
    val_loader = DataLoader(
        val_subset, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=True
    )
    test_loader = DataLoader(
        testset, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=True
    )

    return train_loader, val_loader, test_loader


def train_epoch(model, train_loader, optimizer, device='cuda'):
    """Train for one epoch, return avg loss and accuracy."""
    model.train()
    running_loss, total_correct, total_samples = 0.0, 0, 0

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        logits = model(x).squeeze()
        loss = F.cross_entropy(logits, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * x.size(0)
        total_correct += (logits.argmax(dim=1) == y).sum().item()
        total_samples += x.size(0)

    return running_loss / total_samples, total_correct / total_samples


def evaluate(model, loader, device='cuda'):
    """Evaluate on a dataset, return avg loss and accuracy."""
    model.eval()
    running_loss, total_correct, total_samples = 0.0, 0, 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)

            logits = model(x).squeeze()
            loss = F.cross_entropy(logits, y, reduction='sum')

            running_loss += loss.item()
            total_correct += (logits.argmax(dim=1) == y).sum().item()
            total_samples += x.size(0)

    return running_loss / total_samples, total_correct / total_samples


class EarlyStopping:
    """Early stopping based on validation loss."""

    def __init__(self, patience=10, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0
        self.should_stop = False

    def step(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


def train(config=None):
    """Main training function - callable by W&B sweeps."""
    if config is None:
        config = wandb.config

    def get_config(key, default=None):
        if hasattr(config, key):
            return getattr(config, key)
        elif isinstance(config, dict):
            return config.get(key, default)
        else:
            return default

    # Reproducibility
    seed_everything(get_config('seed', 0))
    device = 'cuda'

    # Data
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=get_config('batch_size', 128),
        data_dir="./data/cifar",
        val_fraction=get_config('val_fraction', 0.1),
        train_subset_fraction=get_config('train_subset_fraction', 1.0),
        seed=get_config('data_split_seed', 42),
    )

    # Model
    manifold = Lorentz(k_value=get_config('curvature', 1.0))

    # Handle coupled norm_config parameter (for sweeps)
    norm_config = get_config('norm_config', None)
    if norm_config == "centering_weightnorm":
        normalisation_mode = "centering_only"
        use_weight_norm = True
    elif norm_config == "normal_noweightnorm":
        normalisation_mode = "normal"
        use_weight_norm = False
    else:
        # Fall back to individual parameters
        normalisation_mode = get_config('normalisation_mode', get_config('bn_mode', 'normal'))
        use_weight_norm = get_config('use_weight_norm', False)

    # Optional coupled Lorentz method config
    lorentz_method = get_config('lorentz_method', None)
    if lorentz_method == "ours":
        fc_variant = "ours"
        mlr_type = "fc_mlr"
    elif lorentz_method == "theirs":
        fc_variant = "theirs"
        mlr_type = "lorentz_mlr"
    elif lorentz_method == "ilnn":
        fc_variant = "ilnn"
        mlr_type = "fc_mlr"
    else:
        fc_variant = get_config('fc_variant', 'ours')
        mlr_type = get_config('mlr_type', get_config('classifier_type', 'lorentz_mlr'))

    if get_config("manifold", "lorentz") == "euclidean":
        model = torchvision.models.resnet18(num_classes=10)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
        model = model.to(device)
    else:
        base_dim = get_config('hidden_dim', 64)
        embedding_dim = get_config('embedding_dim', None)
        model = lorentz_resnet18(
            num_classes=10,
            base_dim=base_dim,
            manifold=manifold,
            init_method=get_config('init_method', 'lorentz_kaiming'),
            input_proj_type=get_config('input_proj_type', 'conv_bn_relu'),
            mlr_init=get_config('mlr_init', 'mlr'),
            normalisation_mode=normalisation_mode,  # "normal", "fix_gamma", "skip_final_bn2", "clamp_scale", "mean_only", or "centering_only"
            mlr_type=mlr_type,  # "lorentz_mlr" or "fc_mlr"
            use_weight_norm=use_weight_norm,
            fc_variant=fc_variant,
            embedding_dim=embedding_dim,
        ).to(device)

    # Log model size
    total_params = sum(p.numel() for p in model.parameters())
    wandb.config.update({"total_params": total_params}, allow_val_change=True)

    # Optimizer
    optimizer_name = get_config('optimizer', 'adam').lower()
    lr = get_config('learning_rate', 1e-3)
    weight_decay = get_config('weight_decay', 0.0)
    start_epoch = 0

    if get_config('compile', True):
        model = torch.compile(model)

    # Use param groups: manifold params get 0.2x learning rate
    model_parameters = get_param_groups(model, lr * 0.2, weight_decay)

    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(
            model_parameters,
            lr=lr,
            weight_decay=weight_decay
        )
    elif optimizer_name == "sgd":
        momentum = get_config('momentum', 0.9)
        optimizer = RiemannianSGD(params=model_parameters, lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=True, stabilize=1)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    # Learning rate scheduler
    scheduler_type = get_config('scheduler', 'none').lower()
    num_epochs = get_config('num_epochs', 100)
    warmup_epochs = get_config('warmup_epochs', 0)
    scheduler = None

    # For StepLR: track first milestone to skip decay for manifold params
    steplr_first_milestone = None
    steplr_gamma = None

    if scheduler_type == 'steplr':
        from torch.optim.lr_scheduler import SequentialLR, MultiStepLR, LinearLR

        # Milestones at ~30%, 60%, 80% of training
        milestones = get_config('milestones', [int(num_epochs * 0.3), int(num_epochs * 0.6), int(num_epochs * 0.8)])
        gamma = get_config('lr_decay', 0.2)
        steplr_first_milestone = milestones[0]
        steplr_gamma = gamma

        if warmup_epochs > 0:
            warmup_scheduler = LinearLR(
                optimizer,
                start_factor=0.01,
                end_factor=1.0,
                total_iters=warmup_epochs
            )
            step_scheduler = MultiStepLR(
                optimizer,
                milestones=[m - warmup_epochs for m in milestones if m > warmup_epochs],
                gamma=gamma
            )
            scheduler = SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, step_scheduler],
                milestones=[warmup_epochs]
            )
        else:
            scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=gamma)

    elif scheduler_type == 'cosine':
        from torch.optim.lr_scheduler import SequentialLR, CosineAnnealingLR, LinearLR

        if warmup_epochs > 0:
            warmup_scheduler = LinearLR(
                optimizer,
                start_factor=0.01,
                end_factor=1.0,
                total_iters=warmup_epochs
            )
            cosine_scheduler = CosineAnnealingLR(
                optimizer,
                T_max=num_epochs - warmup_epochs,
                eta_min=lr * 0.01
            )
            scheduler = SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[warmup_epochs]
            )
        else:
            scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=lr * 0.01)

    # Load checkpoint if specified
    checkpoint_path_load = get_config("resume_checkpoint", None)
    if checkpoint_path_load:
        start_epoch, _ = load_checkpoint(
            checkpoint_path_load, model, optimizer, scheduler, device
        )

    # Early stopping
    early_stopping = None
    if get_config('early_stopping', False):
        early_stopping = EarlyStopping(
            patience=get_config('early_stopping_patience', 10),
            min_delta=get_config('early_stopping_min_delta', 0.0)
        )

    # Create checkpoint directory
    checkpoint_dir = Path(get_config('checkpoint_dir', './checkpoints'))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path_acc = checkpoint_dir / f"best_model_acc_{wandb.run.id}.pt"
    checkpoint_path_loss = checkpoint_dir / f"best_model_loss_{wandb.run.id}.pt"

    # Training loop
    best_val_acc = 0.0
    best_val_loss = float('inf')

    for epoch in range(start_epoch, num_epochs):
        start = time.time()

        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, device,
        )
        val_loss, val_acc = evaluate(model, val_loader, device)

        if not all(map(math.isfinite, [train_loss, train_acc, val_loss, val_acc])):
            msg = (
                f"NaN/Inf detected at epoch {epoch + 1}: "
                f"train_loss={train_loss}, train_acc={train_acc}, "
                f"val_loss={val_loss}, val_acc={val_acc}"
            )
            print(msg)
            wandb.run.summary["nan_detected"] = True
            wandb.run.summary["nan_epoch"] = epoch + 1
            wandb.log({"nan_detected": 1, "nan_epoch": epoch + 1})
            wandb.finish(exit_code=1)
            raise RuntimeError(msg)

        if scheduler:
            scheduler.step()

        # Skip first LR decay for manifold parameters (StepLR only)
        # Manifold params start at 0.2x LR; after first milestone they sync with standard params
        if steplr_first_milestone is not None and (epoch + 1) == steplr_first_milestone:
            optimizer.param_groups[1]['lr'] *= (1 / steplr_gamma)
            print(f"  Skipped lr drop for manifold parameters (restored to {optimizer.param_groups[1]['lr']:.6f})")

        epoch_time = time.time() - start

        # Log metrics
        metrics = {
            "epoch": epoch + 1,
            "train/loss": train_loss,
            "train/acc": train_acc,
            "val/loss": val_loss,
            "val/acc": val_acc,
            "epoch_time": epoch_time,
            "learning_rate": optimizer.param_groups[0]['lr']
        }
        wandb.log(metrics)

        # Track best validation metrics
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            wandb.run.summary["best_val_acc"] = best_val_acc
            # Save model checkpoint
            # Convert wandb config to dict safely
            if isinstance(config, dict):
                config_dict = config
            elif hasattr(config, '_items'):
                config_dict = dict(config._items)
            else:
                config_dict = dict(config)

            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
                'config': config_dict
            }
            if scheduler is not None:
                checkpoint['scheduler_state_dict'] = scheduler.state_dict()

            torch.save(checkpoint, checkpoint_path_acc)
            print(f"  → Saved checkpoint (best val_acc: {val_acc:.4f})")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            wandb.run.summary["best_val_loss"] = best_val_loss

            # Save model checkpoint
            # Convert wandb config to dict safely
            if isinstance(config, dict):
                config_dict = config
            elif hasattr(config, '_items'):
                config_dict = dict(config._items)
            else:
                config_dict = dict(config)

            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
                'config': config_dict
            }
            if scheduler is not None:
                checkpoint['scheduler_state_dict'] = scheduler.state_dict()

            torch.save(checkpoint, checkpoint_path_loss)
            print(f"  → Saved checkpoint (best val_loss: {val_loss:.4f})")

        print(f"Epoch {epoch+1}/{num_epochs} ({epoch_time:.1f}s)")
        print(f"  Train: loss={train_loss:.4f}, acc={train_acc:.4f}")
        print(f"  Val:   loss={val_loss:.4f}, acc={val_acc:.4f}")

        # Early stopping check
        if early_stopping is not None:
            if early_stopping.step(val_loss):
                print(f"Early stopping triggered at epoch {epoch+1}")
                wandb.run.summary["early_stopped_epoch"] = epoch + 1
                break

    # Final test evaluation (only if not a sweep or if explicitly requested)
    if get_config('evaluate_test', False):
        test_loss, test_acc = evaluate(model, test_loader, device)
        wandb.run.summary["test_loss"] = test_loss
        wandb.run.summary["test_acc"] = test_acc
        print(f"Test: loss={test_loss:.4f}, acc={test_acc:.4f}")

    return best_val_acc


def main():
    """
    Entry point for both standalone runs and W&B sweeps.

    For sweeps: wandb.init() connects to the sweep and populates wandb.config
    For standalone: wandb.init() uses the default config below
    """

    default_config = {
        # Model
        "hidden_dim": 64,
        "embedding_dim": None,  # Optional final embedding dimension before classifier
        "curvature": 1.0,
        "init_method": "xavier",
        "input_proj_type": "conv_bn_relu",
        "mlr_init": "mlr",
        "normalisation_mode": "centering_only",  # "normal", "fix_gamma", "skip_final_bn2", "clamp_scale", "mean_only", or "centering_only"
        "mlr_type": "fc_mlr",  # "lorentz_mlr" or "fc_mlr"
        "manifold": "lorentz",
        "fc_variant": "ours",  # "ours", "theirs", or "ilnn"
        "lorentz_method": "theirs",  # None, "ours", or "theirs"
        "norm_config": "normal_noweightnorm",

        # Optimization
        "optimizer": "sgd",
        "learning_rate": 1e-1,
        "weight_decay": 5e-4,
        "momentum": 0.9,
        "batch_size": 128,
        "num_epochs": 200,

        # Scheduler
        "scheduler": "steplr",
        "warmup_epochs": 0,
        "lr_decay": 0.2,

        # Data
        "val_fraction": 0.1,
        "train_subset_fraction": 1.0,
        "data_split_seed": 42,

        # Early stopping
        "early_stopping": False,
        "early_stopping_patience": 10,

        # Misc
        "seed": 0,
        "compile": False,
        "evaluate_test": False,

        # Checkpointing
        "checkpoint_dir": "./checkpoints",
        "resume_checkpoint": None,  # Path to checkpoint to resume from
        "use_weight_norm": True,
    } 

    # wandb.init() will use sweep config if run by wandb agent,
    # otherwise uses default_config
    wandb.init(
        project="ICML_Hyperbolic",
        config=default_config
    )

    train(wandb.config)
    wandb.finish()


if __name__ == "__main__":
    main()
