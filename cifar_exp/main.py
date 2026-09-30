from pathlib import Path
import sys
import time
import math
import json
import os
import platform
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

# Dataset name -> (torchvision class, number of classes).
DATASETS = {
    "cifar10": (torchvision.datasets.CIFAR10, 10),
    "cifar100": (torchvision.datasets.CIFAR100, 100),
}


def synchronize(device):
    """Waits for queued CUDA work, so that wall-clock timings are accurate."""
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def save_runtime(runtime, path):
    """Writes the runtime record to disk (rewritten after every epoch, so the
    measurements survive a job that is killed before the last epoch).

    The schema deliberately matches baselines/ilnn/experiments/vision/train.py, so
    that runtime.json files from both codebases can be compared field by field
    (see runtime_exp/summarize_training_benchmark.py).
    """
    if path is None:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(runtime, f, indent=2)


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
    dataset="cifar10",
    val_fraction=0.1,
    train_subset_fraction=1.0,
    seed=42,
    num_workers=2,
):
    """
    Create CIFAR-10 / CIFAR-100 train/val/test dataloaders.

    Args:
        batch_size: Batch size for all loaders
        data_dir: Directory to store/load CIFAR data
        dataset: "cifar10" or "cifar100"
        val_fraction: Fraction of training set to use for validation (default 10%).
            Set to 0 to train on the full training set and use the *test* set for
            validation — this is the protocol of the ILNN baseline (its
            select_dataset() returns val_loader = test_loader), and matching it is
            what makes the two per-epoch training times comparable (same number of
            iterations per epoch).
        train_subset_fraction: Fraction of training set to use (after val split) for faster sweeps
        seed: Random seed for reproducible splits
        num_workers: DataLoader worker processes (ILNN uses 8)

    Returns:
        train_loader, val_loader, test_loader
    """
    try:
        dataset_cls, _ = DATASETS[dataset]
    except KeyError:
        raise ValueError(f"Unknown dataset {dataset!r}, expected one of {sorted(DATASETS)}")

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
    full_trainset = dataset_cls(
        data_dir, train=True, download=True, transform=train_transform
    )

    # For validation, we need the same data but without augmentation
    full_trainset_val = dataset_cls(
        data_dir, train=True, download=True, transform=val_transform
    )

    # Test set is completely separate
    testset = dataset_cls(
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
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        testset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    if val_size == 0:
        # ILNN protocol: no held-out split, model selection happens on the test set.
        val_loader = test_loader
    else:
        val_loader = DataLoader(
            val_subset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True
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
    """Evaluate on a dataset, return avg loss, top-1 accuracy and top-5 accuracy.

    Accuracies are fractions in [0, 1]; ILNN reports them as percentages.
    """
    model.eval()
    running_loss, total_correct, total_correct5, total_samples = 0.0, 0, 0, 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)

            logits = model(x).squeeze()
            loss = F.cross_entropy(logits, y, reduction='sum')

            top5 = logits.topk(min(5, logits.size(1)), dim=1).indices

            running_loss += loss.item()
            total_correct += (logits.argmax(dim=1) == y).sum().item()
            total_correct5 += (top5 == y.unsqueeze(1)).any(dim=1).sum().item()
            total_samples += x.size(0)

    return (
        running_loss / total_samples,
        total_correct / total_samples,
        total_correct5 / total_samples,
    )


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
    script_start = time.perf_counter()

    dataset = get_config('dataset', 'cifar10')
    num_classes = DATASETS[dataset][1]
    val_fraction = get_config('val_fraction', 0.1)

    # Data
    t0 = time.perf_counter()
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=get_config('batch_size', 128),
        data_dir="./data/cifar",
        dataset=dataset,
        val_fraction=val_fraction,
        train_subset_fraction=get_config('train_subset_fraction', 1.0),
        seed=get_config('data_split_seed', 42),
        num_workers=get_config('num_workers', 2),
    )
    dataset_setup_time = time.perf_counter() - t0

    # Model
    t0 = time.perf_counter()
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
        model = torchvision.models.resnet18(num_classes=num_classes)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
        model = model.to(device)
    else:
        base_dim = get_config('hidden_dim', 64)
        embedding_dim = get_config('embedding_dim', None)
        model = lorentz_resnet18(
            num_classes=num_classes,
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

    synchronize(device)
    model_setup_time = time.perf_counter() - t0

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

    # Runtime record: cost of training this model. Same schema as
    # baselines/ilnn/experiments/vision/train.py so the two can be compared directly.
    runtime_path = get_config('runtime_json', None)
    runtime = {
        "exp_name": get_config('exp_name', f"{get_config('manifold', 'lorentz')}-{dataset}"),
        "dataset": dataset,
        "seed": get_config('seed', 0),
        "num_epochs": num_epochs,
        "batch_size": get_config('batch_size', 128),
        "encoder_manifold": get_config('manifold', 'lorentz'),
        "decoder_manifold": get_config('manifold', 'lorentz'),
        "fc_variant": fc_variant if get_config('manifold', 'lorentz') != 'euclidean' else None,
        "norm_config": norm_config,
        "compile": bool(get_config('compile', False)),
        "device": device,
        "gpu_name": torch.cuda.get_device_name(device) if torch.cuda.is_available() else None,
        "hostname": platform.node(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "wandb_run_id": wandb.run.id,
        "best_checkpoint": str(checkpoint_path_acc.resolve()),
        # ILNN has no held-out val split (val_loader = test_loader); val_fraction=0
        # reproduces that here, anything else means honest model selection.
        "val_protocol": "test_set_as_val" if val_fraction == 0 else f"heldout_{val_fraction}",
        "train_set_size": len(train_loader.dataset),
        "val_set_size": len(val_loader.dataset),
        "test_set_size": len(test_loader.dataset),
        "dataset_setup_time_s": dataset_setup_time,
        "model_setup_time_s": model_setup_time,
        "num_params": total_params,
        "num_trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "iterations_per_epoch": len(train_loader),
        "epochs": [],
    }
    save_runtime(runtime, runtime_path)

    # Training loop
    best_val_acc = 0.0
    best_val_loss = float('inf')
    best_epoch = 0

    train_time = 0.0  # pure optimisation time, without validation/checkpointing
    val_time = 0.0
    train_time_to_best = 0.0  # training seconds spent to reach the best val epoch
    loop_start = time.perf_counter()

    for epoch in range(start_epoch, num_epochs):
        start = time.time()
        epoch_start = time.perf_counter()

        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, device,
        )

        synchronize(device)
        epoch_train_time = time.perf_counter() - epoch_start
        train_time += epoch_train_time
        val_start = time.perf_counter()

        val_loss, val_acc, val_acc5 = evaluate(model, val_loader, device)

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
            "val/acc5": val_acc5,
            "epoch_time": epoch_time,
            "train_time": epoch_train_time,
            "learning_rate": optimizer.param_groups[0]['lr']
        }
        wandb.log(metrics)

        # Track best validation metrics
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            train_time_to_best = train_time
            wandb.run.summary["best_val_acc"] = best_val_acc
            wandb.run.summary["best_epoch"] = best_epoch
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

        synchronize(device)
        epoch_val_time = time.perf_counter() - val_start
        val_time += epoch_val_time

        runtime["epochs"].append({
            "epoch": epoch + 1,
            "train_time_s": epoch_train_time,
            "val_time_s": epoch_val_time,
            "val_acc1": 100.0 * val_acc,
        })
        runtime["epochs_completed"] = len(runtime["epochs"])
        runtime["train_time_s"] = train_time
        runtime["val_time_s"] = val_time
        runtime["elapsed_since_loop_start_s"] = time.perf_counter() - loop_start
        runtime["mean_epoch_train_time_s"] = train_time / len(runtime["epochs"])
        save_runtime(runtime, runtime_path)

        print(f"Epoch {epoch+1}/{num_epochs} ({epoch_time:.1f}s)")
        print(f"  Train: loss={train_loss:.4f}, acc={train_acc:.4f}")
        print(f"  Val:   loss={val_loss:.4f}, acc={val_acc:.4f}, acc5={val_acc5:.4f}")
        print("[timing] epoch {}/{}: train={:.1f}s, val={:.1f}s, elapsed={:.1f}s, "
              "projected_total_train={:.1f}s".format(
                  epoch + 1, num_epochs, epoch_train_time, epoch_val_time,
                  runtime["elapsed_since_loop_start_s"],
                  runtime["mean_epoch_train_time_s"] * num_epochs))

        # Early stopping check
        if early_stopping is not None:
            if early_stopping.step(val_loss):
                print(f"Early stopping triggered at epoch {epoch+1}")
                wandb.run.summary["early_stopped_epoch"] = epoch + 1
                break

    total_loop_time = time.perf_counter() - loop_start
    runtime["total_loop_time_s"] = total_loop_time
    runtime["best_epoch"] = best_epoch
    runtime["best_val_acc1"] = 100.0 * best_val_acc
    runtime["train_time_to_best_epoch_s"] = train_time_to_best

    print("-----------------\nTraining finished\n-----------------")
    print("Best epoch = {}, with val Acc@1={:.4f}".format(best_epoch, best_val_acc))

    epoch_train_times = [e["train_time_s"] for e in runtime["epochs"]]
    if epoch_train_times:
        # Steady-state estimate: the first epoch pays for CUDA/cuDNN warm-up.
        steady = epoch_train_times[1:] if len(epoch_train_times) > 1 else epoch_train_times
        runtime["median_epoch_train_time_s"] = float(np.median(steady))
        runtime["first_epoch_train_time_s"] = epoch_train_times[0]
        runtime["train_time_per_1k_iters_s"] = (
            1000.0 * float(np.median(steady)) / max(len(train_loader), 1)
        )
        print("[timing] RUNTIME SUMMARY for {} ({} epochs on {}):".format(
            runtime["exp_name"], len(epoch_train_times), runtime["gpu_name"] or device))
        print("[timing]   training only        = {:.1f}s ({:.2f} h)".format(train_time, train_time / 3600))
        print("[timing]   training+validation  = {:.1f}s ({:.2f} h)".format(total_loop_time, total_loop_time / 3600))
        print("[timing]   mean epoch (train)   = {:.2f}s".format(train_time / len(epoch_train_times)))
        print("[timing]   median epoch (train, excl. first) = {:.2f}s".format(runtime["median_epoch_train_time_s"]))
        print("[timing]   training to best epoch ({}) = {:.1f}s ({:.2f} h)".format(
            best_epoch, train_time_to_best, train_time_to_best / 3600))

    # Test evaluation. The final model is reported for reference, but the number
    # that matters is the one from the best-validation-epoch checkpoint.
    if get_config('evaluate_test', False):
        t0 = time.perf_counter()
        test_loss, test_acc, test_acc5 = evaluate(model, test_loader, device)
        synchronize(device)
        runtime["test_time_s"] = time.perf_counter() - t0
        wandb.run.summary["test_loss"] = test_loss
        wandb.run.summary["test_acc"] = test_acc
        runtime["final_test_acc1"] = 100.0 * test_acc
        runtime["final_test_acc5"] = 100.0 * test_acc5
        print(f"Test (final model): loss={test_loss:.4f}, acc={test_acc:.4f}, acc5={test_acc5:.4f}")

        print("Testing best model (epoch {})...".format(best_epoch))
        if checkpoint_path_acc.exists():
            # weights_only=False: the checkpoint also stores the run config.
            checkpoint = torch.load(checkpoint_path_acc, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            test_loss, test_acc, test_acc5 = evaluate(model, test_loader, device)
            wandb.run.summary["best_test_loss"] = test_loss
            wandb.run.summary["best_test_acc"] = test_acc
            runtime["best_test_acc1"] = 100.0 * test_acc
            runtime["best_test_acc5"] = 100.0 * test_acc5
            print(f"Test (best val epoch {best_epoch}): loss={test_loss:.4f}, "
                  f"acc={test_acc:.4f}, acc5={test_acc5:.4f}")
        else:
            print(f"No best-val checkpoint at {checkpoint_path_acc} — skipping.")

    runtime["total_script_time_s"] = time.perf_counter() - script_start
    save_runtime(runtime, runtime_path)
    if runtime_path:
        print("[timing] runtime record written to " + str(runtime_path))

    return best_val_acc


def main():
    """
    Entry point for both standalone runs and W&B sweeps.

    For sweeps: wandb.init() connects to the sweep and populates wandb.config
    For standalone: wandb.init() uses the default config below, optionally
    overridden by CLI flags (ignored by the wandb sweep agent).
    """
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--lorentz_method", type=str, default=None, choices=["ours", "theirs", "ilnn"])
    parser.add_argument("--fc_variant", type=str, default=None, choices=["ours", "theirs", "ilnn"])
    parser.add_argument("--norm_config", type=str, default=None,
                        choices=["centering_weightnorm", "normal_noweightnorm"],
                        help="FGG-LNN uses centering_weightnorm (centering-only BatchNorm + "
                             "WeightNorm), HCNN normal_noweightnorm")
    parser.add_argument("--manifold", type=str, default=None, choices=["lorentz", "euclidean"])
    parser.add_argument("--dataset", type=str, default=None, choices=sorted(DATASETS))
    parser.add_argument("--compile", action="store_true",
                        help="torch.compile the model, as in the paper's CIFAR runs")
    parser.add_argument("--num_epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--val_fraction", type=float, default=None,
                        help="0 = train on the full training set and validate on the test "
                             "set (the ILNN baseline's protocol)")
    parser.add_argument("--num_workers", type=int, default=None,
                        help="DataLoader workers (ILNN uses 8)")
    parser.add_argument("--evaluate_test", action="store_true",
                        help="Evaluate the test set with the final and best-val-epoch models")
    parser.add_argument("--exp_name", type=str, default=None)
    parser.add_argument("--runtime_json", type=str, default=None,
                        help="Write the per-epoch runtime record here (ILNN's runtime.json schema)")
    args, _ = parser.parse_known_args()

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
        "dataset": "cifar10",  # "cifar10" or "cifar100"
        "val_fraction": 0.1,
        "train_subset_fraction": 1.0,
        "data_split_seed": 42,
        "num_workers": 2,

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

    if args.fc_variant is not None:
        default_config["fc_variant"] = args.fc_variant
        default_config["lorentz_method"] = None
    elif args.lorentz_method is not None:
        default_config["lorentz_method"] = args.lorentz_method
    if args.norm_config is not None:
        default_config["norm_config"] = args.norm_config
    if args.manifold is not None:
        default_config["manifold"] = args.manifold
    if args.compile:
        default_config["compile"] = True
    if args.dataset is not None:
        default_config["dataset"] = args.dataset
    if args.num_epochs is not None:
        default_config["num_epochs"] = args.num_epochs
    if args.seed is not None:
        default_config["seed"] = args.seed
    if args.val_fraction is not None:
        default_config["val_fraction"] = args.val_fraction
    if args.num_workers is not None:
        default_config["num_workers"] = args.num_workers
    if args.evaluate_test:
        default_config["evaluate_test"] = True
    if args.exp_name is not None:
        default_config["exp_name"] = args.exp_name
    if args.runtime_json is not None:
        default_config["runtime_json"] = args.runtime_json

    # wandb.init() will use sweep config if run by wandb agent,
    # otherwise uses default_config
    wandb.init(
        project="FGG-LNN",
        config=default_config
    )

    train(wandb.config)
    wandb.finish()


if __name__ == "__main__":
    main()
