import os
import glob
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import wandb
import optuna

# Your custom models (assuming they are in a file named mlr.py in the same directory)
from mlr_layers import (
    Lorentz_fully_connected_forward,
    Lorentz_fully_connected_mlr_angle,
    Lorentz_fully_connected_mlr_dist,
    BdeirLorentzMLR_model,
    Baseline_Euclidean,
    Poincare_MLR_Shimizu_van_Spengler,
)

# Model registry for easy selection
MODEL_REGISTRY = {
    "euclidean": Baseline_Euclidean,
    "lorentz_forward": Lorentz_fully_connected_forward,
    "lorentz_angle": Lorentz_fully_connected_mlr_angle,
    "lorentz_dist": Lorentz_fully_connected_mlr_dist,
    "bdeir": BdeirLorentzMLR_model,
    "poincare": Poincare_MLR_Shimizu_van_Spengler,
}

# --- Re-used Helper Functions from the Training Script ---


def get_dataloaders(
    features_path: str, dataset_name: str, batch_size: int
) -> tuple[list, DataLoader, DataLoader, int]:
    base_path = os.path.join(features_path, dataset_name)
    train_data_paths = []
    augmented_path = os.path.join(base_path, "augmented_train")
    if os.path.exists(augmented_path):
        feature_files = sorted(
            glob.glob(os.path.join(augmented_path, "epoch_*_features.pt"))
        )
        label_files = sorted(
            glob.glob(os.path.join(augmented_path, "epoch_*_labels.pt"))
        )
        train_data_paths = list(zip(feature_files, label_files))
    else:
        train_data_paths = [
            (
                os.path.join(base_path, "train_features.pt"),
                os.path.join(base_path, "train_labels.pt"),
            )
        ]
    val_features = torch.load(
        os.path.join(base_path, "valid_features.pt"), map_location="cpu"
    )
    val_labels = torch.load(
        os.path.join(base_path, "valid_labels.pt"), map_location="cpu"
    )
    test_features = torch.load(
        os.path.join(base_path, "test_features.pt"), map_location="cpu"
    )
    test_labels = torch.load(
        os.path.join(base_path, "test_labels.pt"), map_location="cpu"
    )
    val_dataset = TensorDataset(val_features, val_labels)
    test_dataset = TensorDataset(test_features, test_labels)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=2,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=2,
    )
    num_classes = len(torch.unique(val_labels))
    return train_data_paths, val_loader, test_loader, num_classes


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    for features, labels in loader:
        features, labels = features.to(device), labels.to(device)
        outputs = model(features)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def evaluate(model, loader, criterion, device):
    model.eval()
    correct_top1, total_samples = 0, 0
    with torch.no_grad():
        for features, labels in loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_top1 += (predicted == labels).sum().item()
    return (correct_top1 / total_samples) * 100


# --- Optuna Objective Function ---


def objective(trial: optuna.Trial, args: argparse.Namespace):
    """
    This function is called by Optuna for each hyperparameter combination (a "trial").
    It trains a model with the suggested hyperparameters and returns the validation accuracy.
    """
    # 1. Suggest hyperparameters from the search space
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    if args.model == "euclidean":
        curvature = 0.0
    else:
        curvature = trial.suggest_float("curvature", 1e-5, 3.0, log=True)
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, 512])
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)

    # W&B setup for this specific trial
    run = wandb.init(
        project=args.wandb_project,
        config={**vars(args), **trial.params},  # Combine args and trial params
        group=f"HPSearch_{args.dataset}_{args.model}",
        job_type="hparam-search",
        reinit=True,  # Allows wandb.init to be called in a loop
    )
    wandb.run.name = f"trial_{trial.number}_bs{batch_size}_lr{lr:.1e}_k{curvature:.1e}"

    # 2. Setup training components
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_paths, val_loader, _, num_classes = get_dataloaders(
        args.features_path, args.dataset, batch_size
    )

    model_class = MODEL_REGISTRY[args.model]
    model = model_class(in_features=2048, out_features=num_classes, k=curvature).to(
        device
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # 3. Training and Evaluation Loop
    best_val_acc = 0.0
    for epoch in range(args.epochs_per_trial):
        # Load data for the current epoch
        feature_path, label_path = train_paths[epoch % len(train_paths)]
        train_features = torch.load(feature_path, map_location="cpu")
        train_labels = torch.load(label_path, map_location="cpu")
        train_dataset = TensorDataset(train_features, train_labels)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=2,
        )

        train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_acc = evaluate(model, val_loader, criterion, device)

        if val_acc > best_val_acc:
            best_val_acc = val_acc

        # Log metrics to wandb
        wandb.log({"epoch": epoch, "val_accuracy": val_acc})

        # Pruning: Stop unpromising trials early
        trial.report(val_acc, epoch)
        if trial.should_prune():
            wandb.run.summary["state"] = "pruned"
            run.finish()
            raise optuna.exceptions.TrialPruned()

    wandb.run.summary["best_val_accuracy"] = best_val_acc
    run.finish()

    # 4. Return the metric for Optuna to optimize
    return best_val_acc


# --- Main Script Execution ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Hyperparameter search for classification models using Optuna."
    )
    parser.add_argument(
        "--features_path",
        type=str,
        required=True,
        help="Path to the directory of saved features.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["aircraft", "flowers", "dogs-custom"],
        help="Name of the dataset to use.",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=MODEL_REGISTRY.keys(),
        help="Name of the model head to use.",
    )
    parser.add_argument(
        "--n_trials", type=int, default=50, help="Number of Optuna trials to run."
    )
    parser.add_argument(
        "--epochs_per_trial",
        type=int,
        default=50,
        help="Number of epochs to train each trial.",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="Hyperparam-Search",
        help="Weights & Biases project name.",
    )
    args = parser.parse_args()

    # Create a wrapper for the objective function to pass static arguments
    def objective_wrapper(trial):
        return objective(trial, args)

    # Create an Optuna study
    # The pruner stops unpromising trials early
    pruner = optuna.pruners.MedianPruner()
    study = optuna.create_study(direction="maximize", pruner=pruner)

    print(
        f"🚀 Starting hyperparameter search for {args.model} on {args.dataset} for {args.n_trials} trials."
    )

    # Start the optimization
    study.optimize(objective_wrapper, n_trials=args.n_trials)

    # Print results
    print("\n--- Hyperparameter Search Complete ---")
    print(f"Number of finished trials: {len(study.trials)}")

    best_trial = study.best_trial
    print("🏆 Best trial:")
    print(f"  Value (Val Accuracy): {best_trial.value:.4f}%")
    print("  Params: ")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")

    # Save best trial results to a file exp2/hyperparameters.txt
    # Write best parameters as command-line arguments in a single line
    cmd_args = f"--dataset {args.dataset} --model {args.model}"
    for key, value in best_trial.params.items():
        cmd_args += f" --{key} {value}"
    with open("exp2/hyperparameters.txt", "a") as f:
        f.write(cmd_args + "\n")
    print("Results saved to exp2/hyperparameters.txt")
    print("🚀 Hyperparameter search completed successfully!")
