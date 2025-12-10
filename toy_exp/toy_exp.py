import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import argparse
import matplotlib.pyplot as plt
import json
import sys
from typing import List, Dict, Union
from pathlib import Path

# --- Path Setup ---
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

# Attempt imports, assuming the directory structure is correct
try:
    from layers import Lorentz, Lorentz_fully_connected, ChenLinear, Poincare, Poincare_linear, project
except ImportError:
    print("⚠️  Could not import 'layers'. Ensure you are running this from the correct directory.")
    sys.exit(1)


SEEDS = [0]
# Testing distances: 1 to 50, step 3
HYPERBOLIC_DISTANCES = range(1, 50)
LEARNING_RATE = 0.001
MAX_ITERATIONS = 50000
LOSS_THRESHOLD = 0.01
BATCH_SIZE = 64
DIM = 4

# Output Files
OUTPUT_TXT = "./hyperplane_convergence_results.txt"
OUTPUT_JSON = "./hyperplane_convergence_results.json"
OUTPUT_PLOT = "./hyperplane_convergence_plot.png"


def set_seed(seed: int):
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def tangent_space_mse_loss(
    y_pred: torch.Tensor, y_target: torch.Tensor, manifold: Union[Lorentz, Poincare]
) -> torch.Tensor:
    """Calculates MSE in the tangent space at the origin."""
    tangent_pred = manifold.logmap0(y_pred)
    tangent_target = manifold.logmap0(y_target)
    return nn.functional.mse_loss(tangent_pred, tangent_target)


def create_target_point(
    hyperbolic_distance: float, manifold: Union[Lorentz, Poincare], dim: int
) -> torch.Tensor:
    """Creates a target point at a specified hyperbolic distance from the origin."""
    if isinstance(manifold, Lorentz):
        d = float(hyperbolic_distance)
        sqrt_k = manifold.k().sqrt().double()

        time = (1.0 / sqrt_k) * torch.cosh(sqrt_k * d)
        space_first = (1.0 / sqrt_k) * torch.sinh(sqrt_k * d)

        point = torch.zeros(dim + 1, dtype=torch.float64)
        point[0] = time
        point[1] = space_first
        return point
    
    else: # Poincare
        d = float(hyperbolic_distance)
        sqrt_c = manifold.c().sqrt().double()

        # ||x|| = (1 / sqrt(c)) * tanh(sqrt(c) * d / 2)
        euclidean_norm = (1.0 / sqrt_c) * torch.tanh(sqrt_c * d / 2.0)

        point = torch.zeros(dim, dtype=torch.float64)
        point[0] = euclidean_norm.item()
        
        # Project for numerical stability
        point = project(point.unsqueeze(0), manifold.c().double()).squeeze(0)
        return point


def create_input_point(manifold: Union[Lorentz, Poincare], dim: int) -> torch.Tensor:
    """Creates an input point at hyperbolic distance sqrt(2) from the origin."""
    dist_input = np.sqrt(2)

    if isinstance(manifold, Lorentz):
        sqrt_k = manifold.k().sqrt()
        time_input = (1.0 / sqrt_k) * torch.cosh(sqrt_k * dist_input)
        space_input = (1.0 / sqrt_k) * torch.sinh(sqrt_k * dist_input)

        point = torch.tensor(
            [time_input.item(), space_input.item()] + [0.0] * (dim - 1), dtype=torch.float64
        )
        point = point.unsqueeze(0).repeat(BATCH_SIZE, 1)
        return point
    
    else: # Poincare
        sqrt_c = manifold.c().sqrt().double()
        euclidean_norm = (1.0 / sqrt_c) * torch.tanh(sqrt_c * dist_input / 2.0)

        point = torch.zeros(dim, dtype=torch.float64)
        point[0] = euclidean_norm.item()
        point = project(point.unsqueeze(0), manifold.c().double()).squeeze(0)
        point = point.unsqueeze(0).repeat(BATCH_SIZE, 1)
        return point


def run_experiment(
    hyperbolic_distance: int,
    seed: int,
    manifold: Union[Lorentz, Poincare],
    model_type: str,
    dim: int,
) -> int:
    """Runs a single training instance and returns iterations to converge."""
    set_seed(seed)

    # Initialize Model
    if model_type == "ours":
        model = Lorentz_fully_connected(
            in_features=dim,
            out_features=dim,
            manifold=manifold,
        ).double()
        optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)
    elif model_type == "chen":
        model = ChenLinear(
            in_features=dim + 1,
            out_features=dim + 1,
            manifold=manifold,
        ).double()
        optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE * 100)
    else: # Poincare
        model = Poincare_linear(
            in_features=dim,
            out_features=dim,
            manifold=manifold,
        ).double()
        optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)

    x = create_input_point(manifold, dim)
    y = create_target_point(hyperbolic_distance, manifold, dim)
    y = y.unsqueeze(0).repeat(BATCH_SIZE, 1)

    

    # Tracking
    grad_norms = []

    for i in range(MAX_ITERATIONS):
        optimizer.zero_grad()
        y_pred = model(x)
        loss = tangent_space_mse_loss(y_pred, y, manifold)

        if loss.item() < LOSS_THRESHOLD:
            # Calculate gradient norm for display
            total_grad_norm = sum(p.grad.norm().item() ** 2 for p in model.parameters() if p.grad is not None)
            grad_norms.append(np.sqrt(total_grad_norm))
            
            # Calculate predicted distance correctly based on manifold type
            if isinstance(manifold, Lorentz):
                # Distance from origin in Hyperboloid is acosh(time_component * sqrt_k) / sqrt_k
                # Assuming origin is (1/k, 0...0)
                time_val = y_pred[0, 0].item()
                sqrt_k = manifold.k().sqrt().item()
                # clamp for safety
                val = max(1.0, time_val * sqrt_k)
                pred_hyp_dist = (1.0 / sqrt_k) * np.arccosh(val)
            else:
                pred_norm = y_pred[0].norm().item()
                sqrt_c = manifold.c().sqrt().item()
                pred_hyp_dist = (2.0 / sqrt_c) * np.arctanh(min(sqrt_c * pred_norm, 0.9999))

            print(
                f"  Step {i}: dist={hyperbolic_distance}, loss={loss.item():.6f}, "
                f"grad_norm={grad_norms[-1]:.6e}, pred_hyp_dist={pred_hyp_dist:.4f}"
            )
            return i + 1

        loss.backward()

        # Gradient Scaling (Heuristic to help hyperbolic optimization)
        total_param_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_param_norm += p.norm().item() ** 2
        total_param_norm = np.sqrt(total_param_norm)

        scale_factor = max(1.0, total_param_norm)
        for p in model.parameters():
            if p.grad is not None:
                p.grad.mul_(scale_factor)
        
        # Diagnostic Logging
        if i % 1000 == 0:
            # Calculate gradient norm for display
            total_grad_norm = sum(p.grad.norm().item() ** 2 for p in model.parameters() if p.grad is not None)
            grad_norms.append(np.sqrt(total_grad_norm))
            
            # Calculate predicted distance correctly based on manifold type
            if isinstance(manifold, Lorentz):
                # Distance from origin in Hyperboloid is acosh(time_component * sqrt_k) / sqrt_k
                # Assuming origin is (1/k, 0...0)
                time_val = y_pred[0, 0].item()
                sqrt_k = manifold.k().sqrt().item()
                # clamp for safety
                val = max(1.0, time_val * sqrt_k)
                pred_hyp_dist = (1.0 / sqrt_k) * np.arccosh(val)
            else:
                pred_norm = y_pred[0].norm().item()
                sqrt_c = manifold.c().sqrt().item()
                pred_hyp_dist = (2.0 / sqrt_c) * np.arctanh(min(sqrt_c * pred_norm, 0.9999))

            print(
                f"  Step {i}: dist={hyperbolic_distance}, loss={loss.item():.6f}, "
                f"grad_norm={grad_norms[-1]:.6e}, pred_hyp_dist={pred_hyp_dist:.4f}"
            )

        optimizer.step()

    return MAX_ITERATIONS


def save_results_to_file(results: Dict[str, Dict[int, Dict]], filename: str):
    """Saves results to a readable text file."""
    with open(filename, "w", encoding="utf-8") as f:
        f.write("📊 Hyperplane Distance Convergence Results\n")
        f.write("========================================\n")
        
        for model_name, data in results.items():
            f.write(f"\nModel: {model_name.upper()}\n")
            f.write(f"{'Distance':<10} | {'Mean Iters':<12} | {'Std Dev':<10} | {'Status':<10}\n")
            f.write("-" * 50 + "\n")
            
            for dist, stats in data.items():
                mean = stats['mean']
                std = stats['std']
                status = "✅ OK"
                if mean >= MAX_ITERATIONS: status = "❌ FAIL"
                elif mean > MAX_ITERATIONS * 0.8: status = "⚠️ STRUGGLE"
                
                f.write(f"{dist:<10} | {mean:<12.1f} | {std:<10.1f} | {status:<10}\n")
    print(f"\n✅ Text results saved to '{filename}'")


def plot_results(results: Dict[str, Dict[int, Dict]], filename: str):
    """Plots the convergence results."""
    plt.figure(figsize=(10, 6))
    
    colors = {'ours': 'green', 'chen': 'blue', 'poincare': 'red'}
    markers = {'ours': 'o', 'chen': 's', 'poincare': '^'}

    for model_name, data in results.items():
        if not data: continue
        
        distances = sorted(data.keys())
        means = [data[d]['mean'] for d in distances]
        stds = [data[d]['std'] for d in distances]
        
        plt.errorbar(
            distances, means, yerr=stds, 
            label=model_name, color=colors.get(model_name, 'black'),
            fmt=f'-{markers.get(model_name, "o")}', capsize=3, alpha=0.8
        )

    plt.axhline(y=MAX_ITERATIONS, color='gray', linestyle='--', alpha=0.5, label="Max Iterations")
    plt.xlabel("Hyperbolic Distance")
    plt.ylabel("Iterations to Convergence")
    plt.title(f"Convergence vs Hyperbolic Distance (Batch {BATCH_SIZE})")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.savefig(filename, dpi=300)
    print(f"✅ Plot saved to '{filename}'")


def main():
    parser = argparse.ArgumentParser(description="Hyperbolic Hyperplane Convergence Test")
    parser.add_argument("--models", nargs="+", choices=["ours", "chen", "poincare"], 
                        default=["ours", "chen", "poincare"], help="Models to run")
    args = parser.parse_args()

    print("🚀 Starting Hyperplane Distance Test")
    print(f"Models: {args.models}")
    print(f"Distance Range: {min(HYPERBOLIC_DISTANCES)} - {max(HYPERBOLIC_DISTANCES)}")
    print("-" * 80)

    # Manifolds (Instantiate once)
    lorentz_manifold = Lorentz(k=0.1)
    poincare_manifold = Poincare(c=0.1)

    # Results Structure: results[model][distance] = {mean, std, raw}
    results = {m: {} for m in args.models}
    
    # Track which models have failed to stop running them on larger distances
    failed_models = set()

    for dist in HYPERBOLIC_DISTANCES:
        print(f"\n📏 Testing distance: {dist}")

        # If all requested models have failed, we can stop the experiment entirely
        if len(failed_models) == len(args.models):
            print("🛑 All models failed. Stopping experiment.")
            break

        for model_name in args.models:
            if model_name in failed_models:
                continue

            # Select Manifold
            if model_name in ["ours", "chen"]:
                manifold = lorentz_manifold
            else:
                manifold = poincare_manifold

            print(f"   ▶️  Running {model_name}...", end="", flush=True)

            current_iterations = []
            for seed in SEEDS:
                iters = run_experiment(dist, seed, manifold, model_name, DIM)
                current_iterations.append(iters)

            # Analyze Results
            mean_iters = np.mean(current_iterations)
            std_iters = np.std(current_iterations)
            
            results[model_name][dist] = {
                "mean": mean_iters,
                "std": std_iters,
                "raw": current_iterations
            }

            # Check Status
            if mean_iters >= MAX_ITERATIONS:
                print(f" ❌ FAILED (Avg {mean_iters:.0f}) -> Skipping future distances for {model_name}")
                failed_models.add(model_name)
            elif mean_iters > MAX_ITERATIONS * 0.8:
                print(f" ⚠️  STRUGGLING (Avg {mean_iters:.0f})")
            else:
                print(f" ✅ (Avg {mean_iters:.0f})")

    # --- Save & Plot ---
    save_results_to_file(results, OUTPUT_TXT)
    
    # Save JSON for programmatic access
    with open(OUTPUT_JSON, "w") as f:
        # Convert np types to python native for JSON serialization
        json_ready = {
            m: {d: {"mean": float(v["mean"]), "std": float(v["std"]), "raw": [int(x) for x in v["raw"]]} 
                for d, v in dists.items()}
            for m, dists in results.items()
        }
        json.dump(json_ready, f, indent=4)
        
    plot_results(results, OUTPUT_PLOT)
    print("\nExperiment Complete.")

if __name__ == "__main__":
    main()