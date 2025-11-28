"""
Test script for Poincaré model - Testing hyperplane distance convergence
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from typing import List, Dict
from pathlib import Path
import sys

# Add layers directory to path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from layers.poincare import Poincare, Poincare_linear, project


# --- Constants for the Experiment ---
SEEDS = range(3)
HYPERBOLIC_DISTANCES = range(1, 45, 3)  # Start from 1, Poincaré may be more sensitive
LEARNING_RATE = 0.001
MAX_ITERATIONS = 10000
LOSS_THRESHOLD = 0.01
BATCH_SIZE = 64
DIM = 4
OUTPUT_FILENAME = "./poincare_hyperplane_distance_results.txt"


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


def tangent_space_mse_loss(y_pred: torch.Tensor, y_target: torch.Tensor, manifold: Poincare) -> torch.Tensor:
    """Calculates MSE in the tangent space at the origin."""
    tangent_pred = manifold.logmap0(y_pred)
    tangent_target = manifold.logmap0(y_target)
    return nn.functional.mse_loss(tangent_pred, tangent_target)


def create_target_point_poincare(hyperbolic_distance: float, manifold: Poincare, dim: int) -> torch.Tensor:
    """
    Creates a target point at a specified hyperbolic distance from the origin in the Poincaré ball.
    
    In the Poincaré ball with curvature c, the hyperbolic distance from origin to a point x is:
        d(0, x) = (2 / sqrt(c)) * arctanh(sqrt(c) * ||x||)
    
    Inverting: ||x|| = (1 / sqrt(c)) * tanh(sqrt(c) * d / 2)
    """
    d = float(hyperbolic_distance)
    sqrt_c = manifold.c().sqrt().double()
    
    # Compute the Euclidean norm needed to achieve hyperbolic distance d
    # ||x|| = (1 / sqrt(c)) * tanh(sqrt(c) * d / 2)
    euclidean_norm = (1.0 / sqrt_c) * torch.tanh(sqrt_c * d / 2.0)
    
    # Create point along first spatial dimension
    point = torch.zeros(dim, dtype=torch.float64)
    point[0] = euclidean_norm.item()
    
    # Project to ensure it's inside the ball (numerical safety)
    point = project(point.unsqueeze(0), manifold.c().double()).squeeze(0)
    
    return point


def create_input_point_poincare(manifold: Poincare, dim: int) -> torch.Tensor:
    """
    Creates an input point at hyperbolic distance sqrt(2) from the origin in Poincaré ball.
    """
    dist_input = np.sqrt(2)
    sqrt_c = manifold.c().sqrt().double()
    
    # Compute Euclidean norm for hyperbolic distance sqrt(2)
    euclidean_norm = (1.0 / sqrt_c) * torch.tanh(sqrt_c * dist_input / 2.0)
    
    # Create point along first spatial dimension
    point = torch.zeros(dim, dtype=torch.float64)
    point[0] = euclidean_norm.item()
    
    # Project to ensure it's inside the ball
    point = project(point.unsqueeze(0), manifold.c().double()).squeeze(0)
    
    return point


def run_experiment(
    hyperbolic_distance: int,
    seed: int,
    manifold: Poincare,
    dim: int,
) -> int:
    """Runs a single training instance and returns the number of iterations to converge."""
    set_seed(seed)
    
    # Initialize Poincaré linear model
    model = Poincare_linear(
        in_features=dim,
        out_features=dim,
        manifold=manifold,
    ).double()
    
    # Create input point at hyperbolic distance sqrt(2)
    x = create_input_point_poincare(manifold, dim)
    x = x.unsqueeze(0).repeat(BATCH_SIZE, 1)
    
    # Create target point at specified hyperbolic distance
    y = create_target_point_poincare(hyperbolic_distance, manifold, dim)
    y = y.unsqueeze(0).repeat(BATCH_SIZE, 1)
    
    # Setup optimizer
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)
    
    # 🔍 DIAGNOSTIC: Track gradient magnitudes
    grad_norms = []
    param_norms = []
    
    for i in range(MAX_ITERATIONS):
        optimizer.zero_grad()
        y_pred = model(x)
        loss = tangent_space_mse_loss(y_pred, y, manifold)
        
        if loss.item() < LOSS_THRESHOLD:
            return i + 1
        
        loss.backward()
        
        # 🔧 Scale gradients by parameter norm to counteract vanishing gradients
        total_param_norm = 0.0
        for name, p in model.named_parameters():
            if p.grad is not None:
                total_param_norm += p.norm().item() ** 2
        total_param_norm = np.sqrt(total_param_norm)
        
        # Scale factor: larger params need larger gradient steps
        scale_factor = max(1.0, total_param_norm)
        for p in model.parameters():
            if p.grad is not None:
                p.grad.mul_(scale_factor)
        
        # 🔍 Log gradient and parameter norms (after scaling)
        total_grad_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_grad_norm += p.grad.norm().item() ** 2
        grad_norms.append(np.sqrt(total_grad_norm))
        param_norms.append(total_param_norm)
        
        # 🔍 Print every 1000 steps
        if i % 1000 == 0:
            # Compute predicted hyperbolic distance
            pred_norm = y_pred[0].norm().item()
            sqrt_c = manifold.c().sqrt().item()
            pred_hyp_dist = (2.0 / sqrt_c) * np.arctanh(min(sqrt_c * pred_norm, 0.9999))
            
            print(f"  Step {i}: dist={hyperbolic_distance}, loss={loss.item():.6f}, "
                  f"grad_norm={grad_norms[-1]:.6e}, pred_hyp_dist={pred_hyp_dist:.4f}")
        
        optimizer.step()
    
    return MAX_ITERATIONS


def save_results_to_txt(results: Dict[int, List[int]], filename: str):
    """Saves the aggregated results to a plain text file."""
    header1 = f"{'Distance':<12} | {'Euclidean Norm':<20} | {'Mean ± Std (Iters)':<25} | {'Status':<15}"
    header2 = f"{'':-<12} | {'':-<20} | {'':-<25} | {'':-<15}"
    separator = "-" * len(header1)
    
    # Compute Euclidean norm for each distance (with c=0.1)
    sqrt_c = np.sqrt(0.1)
    
    with open(filename, 'w') as f:
        f.write("=" * len(header1) + "\n")
        f.write("📊 Poincaré Hyperplane Distance Convergence Results\n")
        f.write("=" * len(header1) + "\n")
        f.write("Testing Poincaré linear layer convergence\n")
        f.write(f"Batch Size: {BATCH_SIZE}, Seeds: {len(SEEDS)}, Max Iterations: {MAX_ITERATIONS}\n")
        f.write("=" * len(header1) + "\n")
        f.write(header1 + "\n")
        f.write(header2 + "\n")
        f.write(separator + "\n")
        
        for dist, iters in sorted(results.items()):
            res = np.array(iters)
            mean_iters = np.mean(res)
            std_iters = np.std(res)
            
            # Euclidean norm in Poincaré ball for this hyperbolic distance
            euclidean_norm = (1.0 / sqrt_c) * np.tanh(sqrt_c * dist / 2.0)
            euclidean_info = f"||x|| ≈ {euclidean_norm:.6f}"
            
            if mean_iters >= MAX_ITERATIONS:
                status = "❌ FAILED"
            elif mean_iters > MAX_ITERATIONS * 0.8:
                status = "⚠️  STRUGGLING"
            else:
                status = "✅ OK"
            
            f.write(f"{dist:<12} | {euclidean_info:<20} | {mean_iters:7.1f} ± {std_iters:5.1f} | {status:<15}\n")
        
        f.write(separator + "\n")
    
    print(f"\n✅ Results successfully saved to '{filename}'")


def main():
    """Main function to run the full Poincaré experiment."""
    print("🚀 Starting Poincaré Hyperplane Distance Test")
    print(f"Testing hyperbolic distances: {min(HYPERBOLIC_DISTANCES)} to {max(HYPERBOLIC_DISTANCES)}")
    print(f"Seeds: {len(SEEDS)}, Batch Size: {BATCH_SIZE}, Dimension: {DIM}")
    print("-" * 80)
    
    # Use curvature c=0.1 (matching default in Poincare class)
    manifold = Poincare(c=0.1)
    print(f"Manifold curvature c = {manifold.c().item():.4f}")
    print(f"Ball radius = {manifold.radius().item():.4f}")
    print("-" * 80)
    
    results: Dict[int, List[int]] = {d: [] for d in HYPERBOLIC_DISTANCES}
    
    convergence_failed = False
    first_failure_distance = None
    
    for dist in HYPERBOLIC_DISTANCES:
        sqrt_c = manifold.c().sqrt().item()
        euclidean_norm = (1.0 / sqrt_c) * np.tanh(sqrt_c * dist / 2.0)
        print(f"\n📏 Testing hyperbolic distance: {dist} (Euclidean norm ≈ {euclidean_norm:.6f})")
        
        if convergence_failed:
            print(f"   Skipping... (convergence failed at distance {first_failure_distance})")
            results[dist] = [MAX_ITERATIONS] * len(SEEDS)
            continue
        
        current_iterations = []
        for seed in SEEDS:
            iterations = run_experiment(
                hyperbolic_distance=dist,
                seed=seed,
                manifold=manifold,
                dim=DIM,
            )
            current_iterations.append(iterations)
        
        results[dist] = current_iterations
        mean_iters = np.mean(current_iterations)
        
        if mean_iters >= MAX_ITERATIONS:
            convergence_failed = True
            first_failure_distance = dist
            print(f"   ❌ FAILED to converge (mean iterations: {mean_iters:.1f})")
            print(f"   📊 Will skip remaining distances")
        elif mean_iters > MAX_ITERATIONS * 0.8:
            print(f"   ⚠️  STRUGGLING (mean iterations: {mean_iters:.1f})")
        else:
            print(f"   ✅ Converged (mean iterations: {mean_iters:.1f})")
    
    # --- Print Summary ---
    print("\n" + "=" * 80)
    print("📊 Poincaré Convergence Results Summary")
    print("=" * 80)
    
    header1 = f"{'Distance':<12} | {'Euclidean Norm':<20} | {'Mean ± Std (Iters)':<25} | {'Status':<15}"
    print(header1)
    print("-" * len(header1))
    
    sqrt_c = manifold.c().sqrt().item()
    for dist, iters in sorted(results.items()):
        res = np.array(iters)
        mean_iters = np.mean(res)
        std_iters = np.std(res)
        euclidean_norm = (1.0 / sqrt_c) * np.tanh(sqrt_c * dist / 2.0)
        euclidean_info = f"||x|| ≈ {euclidean_norm:.6f}"
        
        if mean_iters >= MAX_ITERATIONS:
            status = "❌ FAILED"
        elif mean_iters > MAX_ITERATIONS * 0.8:
            status = "⚠️  STRUGGLING"
        else:
            status = "✅ OK"
        
        print(f"{dist:<12} | {euclidean_info:<20} | {mean_iters:7.1f} ± {std_iters:5.1f} | {status:<15}")
    
    print("-" * len(header1))
    
    save_results_to_txt(results, OUTPUT_FILENAME)
    
    # --- Summary Statistics ---
    print("\n" + "=" * 80)
    print("📈 Summary Statistics")
    print("=" * 80)
    
    max_converged_distance = None
    for dist in sorted(results.keys()):
        if np.mean(results[dist]) < MAX_ITERATIONS:
            max_converged_distance = dist
        else:
            break
    
    if max_converged_distance:
        print(f"✅ Maximum hyperbolic distance with consistent convergence: {max_converged_distance}")
        euclidean_norm = (1.0 / sqrt_c) * np.tanh(sqrt_c * max_converged_distance / 2.0)
        print(f"   (Euclidean norm ≈ {euclidean_norm:.6f})")
    else:
        print("❌ Failed to converge even at the smallest distance tested")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()