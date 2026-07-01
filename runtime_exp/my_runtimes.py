from pathlib import Path
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import gc  # Import the garbage collector module at the top of your script
import csv

parent_dir = Path(__file__).parent
sys.path.insert(0, str(parent_dir.parent))
from layers import Lorentz, LorentzFullyConnected, Poincare, project, Poincare_linear, ChenLinear, ILNNLinear, PoincareActivation

# # Examples of input shapes and outputs shapes:
# Lorentz_fully_connected(10, 5, manifold=Lorentz(0.1)) # accepts input of shape [batch_size, 10+1] and outputs [batch_size, 5+1]
# BdeirLorentzMLR(num_features=10, num_classes=5, manifold=Lorentz(0.1)) # accepts input of shape [batch_size, 10] and outputs [batch_size, 5]
# ChenLinear(manifold=Lorentz(0.1), in_features=10, out_features=5) # accepts input of shape [batch_size, 10] and outputs [batch_size, 5]
# Poincare_linear(10, 5, manifold=Poincare(0.1)) # accepts input of shape [batch_size, 10] and outputs [batch_size, 5]

# =============================================================================
# Timing Function (with minor cleanup)
# =============================================================================
class EuclideanLinear(nn.Module):
      def __init__(self, in_features, out_features):
          super().__init__()
          self.weight = nn.Parameter(torch.randn(in_features, out_features))
          self.bias = nn.Parameter(torch.zeros(out_features))

      def forward(self, x):
          return F.relu(x @ self.weight + self.bias)
      

def time_operation_cuda(model, input_tensor, method_name="forward"):
    """
    Measures the forward and backward pass time for a given model and input
    using CUDA events for accurate GPU timing. Removes outliers and calculates
    mean and standard deviation.

    Args:
        model (torch.nn.Module): The model to benchmark. Must be on a CUDA device.
        input_tensor (torch.Tensor): The input data. Must be on a CUDA device.
        method_name (str): The name of the method to call on the model.

    Returns:
        tuple: (mean_forward, std_forward, mean_backward, std_backward) in milliseconds.
    """
    if not next(model.parameters()).is_cuda or not input_tensor.is_cuda:
        raise ValueError(
            "Model and input_tensor must be on a CUDA device for accurate timing."
        )

    # --- Warm-up runs ---
    for _ in range(10):
        output = getattr(model, method_name)(input_tensor)
        if output.requires_grad:
            grad_tensor = torch.randn_like(output)
            output.backward(grad_tensor, retain_graph=False)
            del grad_tensor
        del output
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    # --- Timing runs ---
    forward_times = []
    backward_times = []
    num_runs = 50
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    for _ in range(num_runs):
        # --- Forward Pass Timing ---
        start_event.record()
        output = getattr(model, method_name)(input_tensor)
        end_event.record()
        torch.cuda.synchronize()
        forward_times.append(start_event.elapsed_time(end_event))

        # --- Backward Pass Timing (if applicable) ---
        if output.requires_grad:
            grad_tensor = torch.randn_like(output)
            start_event.record()
            output.backward(grad_tensor, retain_graph=False)
            end_event.record()
            torch.cuda.synchronize()
            backward_times.append(start_event.elapsed_time(end_event))
            del grad_tensor

        del output  # Clean up the output tensor in the loop

    def get_stats_without_outliers(times):
        if not times:
            return 0.0, 0.0
        times_np = np.array(times)
        q1, q3 = np.percentile(times_np, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        filtered_times = times_np[(times_np >= lower_bound) & (times_np <= upper_bound)]
        if len(filtered_times) == 0:
            filtered_times = times_np
        return np.mean(filtered_times), np.std(filtered_times)

    fw_mean, fw_std = get_stats_without_outliers(forward_times)
    bw_mean, bw_std = get_stats_without_outliers(backward_times)

    return fw_mean, fw_std, bw_mean, bw_std


# =============================================================================
# Main Experiment Function (Corrected)
# =============================================================================


def main():
    """
    Main function to run the benchmarking experiment with robust memory
    management and dynamic results saving.
    """
    if not torch.cuda.is_available():
        print("CUDA is not available. This script requires a GPU.")
        return

    device = torch.device("cuda")
    print(f"Using device: {device}")

    # --- Experiment Configurations ---
    configs = []

    dims = [16, 64, 256, 1024, 4096]
    configs = [(i, o, 512) for i, o in zip(dims, dims)]

    # --- Setup for Dynamic CSV Writing ---
    output_filename = "runtime_results_mine.csv"
    fieldnames = [
        "Model",
        "In",
        "Out",
        "Batch",
        "Fwd Mean (ms)",
        "Fwd Std (ms)",
        "Bwd Mean (ms)",
        "Bwd Std (ms)",
    ]

    # Write header. 'w' mode overwrites the file if it exists.
    with open(output_filename, "w", newline="") as f_output:
        writer = csv.DictWriter(f_output, fieldnames=fieldnames)
        writer.writeheader()

    lorentz_manifold = Lorentz().to(device)
    poincare_manifold = Poincare().to(device)
    len_configs = len(configs)

    for index_run, (in_dim, out_dim, batch_size) in enumerate(configs):
        print(
            f"\n--- Running Config: In={in_dim}, Out={out_dim}, Batch={batch_size} ---"
        )
        print(
            f"  Index: {index_run + 1}/{len_configs} | {100 * (index_run + 1) / len_configs:.2f}% complete"
        )

        try:
            # --- Prepare Synthetic Data ---
            base_tangent_data = torch.randn(batch_size, in_dim, device=device)
            poincare_input = poincare_manifold.expmap0(
                base_tangent_data.clone()
            ).detach()
            poincare_input.requires_grad = True
            lorentz_input = lorentz_manifold.expmap0(base_tangent_data.clone()).detach()
            lorentz_input.requires_grad = True
            euclidean_input = base_tangent_data.clone().detach()
            euclidean_input.requires_grad = True
            del base_tangent_data

        except torch.cuda.OutOfMemoryError as e:
            print(
                f"  ERROR preparing data for this config: {e}. Skipping to next config."
            )
            # Clean up any partial allocations
            gc.collect()
            torch.cuda.empty_cache()
            continue

        # --- Models to Test (defined inside the loop to be fresh for each config) ---
        models_to_test = {
            "FGG-LNN (cached)": (
                LorentzFullyConnected(
                    in_dim + 1, out_dim, manifold=lorentz_manifold, activation=F.relu
                ),
                lorentz_input,
                "forward_cache",
            ),
            "FGG-LNN": (
                LorentzFullyConnected(
                    in_dim + 1, out_dim, manifold=lorentz_manifold, activation=F.relu
                ),
                lorentz_input,
                "forward",
            ),
            "Chen": (
                ChenLinear(
                    manifold=lorentz_manifold,
                    in_features=in_dim + 1,
                    out_features=out_dim + 1,
                    bias=True,
                    normalize=True,
                    learn_scale=True,
                ),
                lorentz_input,
                "forward",
            ),
            "ILNN": (
                ILNNLinear(
                    manifold=lorentz_manifold,
                    in_features=in_dim + 1,
                    out_features=out_dim + 1,
                ),
                lorentz_input,
                "forward",
            ),
            "Poincare": (nn.Sequential(Poincare_linear(in_dim, out_dim, manifold=poincare_manifold), PoincareActivation(activation=nn.ReLU(), manifold=poincare_manifold)),
                                poincare_input,
                                "forward"),
            "Euclidean": (
                EuclideanLinear(in_dim, out_dim),
                euclidean_input,
                "forward",
            ),
        }

        for name, (model_instance, input_data, method) in models_to_test.items():
            for compiled in [False]:
                print(f"  Testing: {name} | Compiled: {compiled}")

                model = None  # Ensure model is defined for the finally block
                result_row = {
                    "Model": name,
                    "In": in_dim,
                    "Out": out_dim,
                    "Batch": batch_size,
                }

                try:
                    model = model_instance.to(device)

                    if compiled:
                        try:
                            if method != "forward":
                                setattr(
                                    model, method, torch.compile(getattr(model, method))
                                )
                            else:
                                model = torch.compile(model)
                        except Exception as e:
                            print(f"    Could not compile {name}: {e}")
                            # Skip this iteration if compilation fails
                            continue

                    current_input = input_data.clone().detach()
                    current_input.requires_grad = True

                    fw_mean, fw_std, bw_mean, bw_std = time_operation_cuda(
                        model, current_input, method_name=method
                    )

                    result_row.update(
                        {
                            "Fwd Mean (ms)": fw_mean,
                            "Fwd Std (ms)": fw_std,
                            "Bwd Mean (ms)": bw_mean,
                            "Bwd Std (ms)": bw_std,
                        }
                    )

                except Exception as e:
                    print(f"    ERROR running {name} (Compiled: {compiled}): {e}")
                    result_row.update(
                        {
                            "Fwd Mean (ms)": "ERROR",
                            "Fwd Std (ms)": "ERROR",
                            "Bwd Mean (ms)": "ERROR",
                            "Bwd Std (ms)": "ERROR",
                        }
                    )

                finally:
                    # 💾 DYNAMICALLY SAVE RESULT: Append the result for this run to the CSV.
                    with open(output_filename, "a", newline="") as f_output:
                        writer = csv.DictWriter(f_output, fieldnames=fieldnames)
                        writer.writerow(result_row)

                    # 🧹 MEMORY CLEANUP: Crucial for preventing OOM errors.
                    del model
                    if "current_input" in locals():
                        del current_input
                    gc.collect()
                    torch.cuda.empty_cache()

        # --- Clean up data tensors at the end of the config loop ---
        del poincare_input
        del lorentz_input
        del euclidean_input
        gc.collect()
        torch.cuda.empty_cache()
        torch._dynamo.reset()

    print("\n\n--- Benchmark Finished ---")
    print(f"Results have been saved to {output_filename}")


if __name__ == "__main__":
    main()
