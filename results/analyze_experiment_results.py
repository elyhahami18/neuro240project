#!/usr/bin/env python
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os

# Set working directory to script location
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Load the results
results_path = Path("experiment_results.pt")
print(f"Loading results from {results_path}")
data = torch.load(results_path)

# Check available keys
print("Available data in results:")
for key in data.keys():
    print(f"- {key}")

# Extract data components
method_results = data.get("results", {})
forgetting_data = data.get("task_forgetting", {})
initial_accuracies = data.get("initial_accuracies", {})
forward_transfer = data.get("forward_transfer", {})
backward_transfer = data.get("backward_transfer", {})

# Get methods
methods = list(method_results.keys())
print(f"\nMethods in this experiment: {methods}")

# Create results directory
output_dir = Path("visualizations")
output_dir.mkdir(exist_ok=True)

# Set up consistent colors for methods
colors = plt.cm.tab10(np.linspace(0, 1, len(methods)))
method_colors = {method: color for method, color in zip(methods, colors)}

# 1. Plot accuracy after each task
plt.figure(figsize=(12, 8))

for i, method in enumerate(methods):
    results = method_results[method]
    if not results:
        continue
    
    # Get final accuracies after completing all tasks
    final_accuracies = results[-1]
    tasks = list(range(1, len(final_accuracies) + 1))
    
    plt.plot(tasks, final_accuracies, marker='o', label=method, 
             color=method_colors[method], linewidth=2)

plt.title('Final Accuracy on Each Task', fontsize=16)
plt.xlabel('Task', fontsize=14)
plt.ylabel('Accuracy (%)', fontsize=14)
plt.xticks(tasks)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig(output_dir / "final_accuracies.png", dpi=300)
print("Saved final accuracies plot")

# 2. Plot forgetting curves for Task 1
plt.figure(figsize=(12, 8))

for method in methods:
    results = method_results[method]
    if not results or len(results) <= 1:
        continue
    
    # Extract Task 1 accuracy after each task
    task1_acc = [result[0] for result in results]
    x_values = list(range(1, len(task1_acc) + 1))
    
    plt.plot(x_values, task1_acc, marker='o', label=method, 
             color=method_colors[method], linewidth=2)

plt.title('Task 1 Accuracy Throughout Training (Forgetting Curve)', fontsize=16)
plt.xlabel('After Training on Task #', fontsize=14)
plt.ylabel('Task 1 Accuracy (%)', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig(output_dir / "forgetting_curves.png", dpi=300)
print("Saved forgetting curves plot")

# 3. Compare forward transfer
if forward_transfer:
    plt.figure(figsize=(10, 6))
    
    methods_with_data = [m for m in methods if m in forward_transfer and forward_transfer[m]]
    if methods_with_data:
        ft_values = [forward_transfer[m][0] for m in methods_with_data]
        
        bars = plt.bar(methods_with_data, ft_values, color=[method_colors[m] for m in methods_with_data])
        plt.title('Forward Transfer (Task 1 → Task 2)', fontsize=16)
        plt.ylabel('Accuracy on Task 2 before Training (%)', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(output_dir / "forward_transfer.png", dpi=300)
        print("Saved forward transfer plot")

# 4. Compare backward transfer
if backward_transfer:
    plt.figure(figsize=(10, 6))
    
    methods_with_data = [m for m in methods if m in backward_transfer and backward_transfer[m]]
    if methods_with_data:
        bt_values = [backward_transfer[m][0] for m in methods_with_data]
        
        bars = plt.bar(methods_with_data, bt_values, color=[method_colors[m] for m in methods_with_data])
        plt.title('Backward Transfer (Last Task → Previous Tasks)', fontsize=16)
        plt.ylabel('Average Accuracy Change on Previous Tasks (%)', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.5 if height > 0 else height - 2,
                    f'{height:.1f}%', ha='center', va='bottom' if height > 0 else 'top', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(output_dir / "backward_transfer.png", dpi=300)
        print("Saved backward transfer plot")

# 5. Plot specific task forgetting
for task_idx in [1, 2, 3]:  # Tasks for which we have forgetting data
    plt.figure(figsize=(12, 8))
    
    for method in methods:
        if method in forgetting_data and task_idx in forgetting_data[method]:
            forget_values = forgetting_data[method][task_idx]
            if not forget_values:
                continue
                
            # x-values represent tasks after task_idx
            x_values = list(range(task_idx + 1, task_idx + 1 + len(forget_values)))
            
            plt.plot(x_values, forget_values, marker='o', label=method, 
                     color=method_colors[method], linewidth=2)
    
    plt.title(f'Task {task_idx} Accuracy After Learning Later Tasks', fontsize=16)
    plt.xlabel('After Training on Task #', fontsize=14)
    plt.ylabel('Task {task_idx} Accuracy (%)', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(output_dir / f"task{task_idx}_forgetting.png", dpi=300)
    print(f"Saved Task {task_idx} forgetting plot")

# 6. Calculate and plot average accuracy across all tasks
plt.figure(figsize=(10, 6))

avg_accuracies = []
for method in methods:
    results = method_results[method]
    if not results:
        avg_accuracies.append(0)
        continue
        
    # Final accuracies on all tasks
    final_results = results[-1]
    avg_acc = sum(final_results) / len(final_results)
    avg_accuracies.append(avg_acc)

bars = plt.bar(methods, avg_accuracies, color=[method_colors[m] for m in methods])
plt.title('Average Accuracy Across All Tasks', fontsize=16)
plt.ylabel('Average Accuracy (%)', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.3, axis='y')

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
            f'{height:.1f}%', ha='center', va='bottom', fontsize=12)

plt.tight_layout()
plt.savefig(output_dir / "average_accuracy.png", dpi=300)
print("Saved average accuracy plot")

# 7. Calculate catastrophic forgetting (CF) metrics
plt.figure(figsize=(10, 6))

cf_metrics = []
for method in methods:
    results = method_results[method]
    if not results or len(results) <= 1:
        cf_metrics.append(0)
        continue
    
    # Task 1 accuracy after first task vs after all tasks
    task1_initial = results[0][0]
    task1_final = results[-1][0]
    forgetting = task1_initial - task1_final
    cf_metrics.append(forgetting)

bars = plt.bar(methods, cf_metrics, color=[method_colors[m] for m in methods])
plt.title('Catastrophic Forgetting (Task 1)', fontsize=16)
plt.ylabel('Accuracy Decrease (%)', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.3, axis='y')

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
            f'{height:.1f}%', ha='center', va='bottom', fontsize=12)

plt.tight_layout()
plt.savefig(output_dir / "catastrophic_forgetting.png", dpi=300)
print("Saved catastrophic forgetting plot")

# 8. Generate comprehensive text report
report_path = output_dir / "experiment_report.txt"
with open(report_path, 'w') as f:
    f.write("CONTINUAL LEARNING EXPERIMENT RESULTS\n")
    f.write("===================================\n\n")
    
    f.write(f"Methods analyzed: {', '.join(methods)}\n\n")
    
    # Average accuracy
    f.write("AVERAGE ACCURACY ACROSS ALL TASKS:\n")
    for i, method in enumerate(methods):
        f.write(f"{method}: {avg_accuracies[i]:.2f}%\n")
    f.write("\n")
    
    # Catastrophic forgetting
    f.write("CATASTROPHIC FORGETTING (Task 1):\n")
    for i, method in enumerate(methods):
        f.write(f"{method}: {cf_metrics[i]:.2f}%\n")
    f.write("\n")
    
    # Forward transfer
    if forward_transfer:
        f.write("FORWARD TRANSFER (Task 1 → Task 2):\n")
        for method in methods:
            if method in forward_transfer and forward_transfer[method]:
                f.write(f"{method}: {forward_transfer[method][0]:.2f}%\n")
        f.write("\n")
    
    # Backward transfer
    if backward_transfer:
        f.write("BACKWARD TRANSFER (Last Task → Previous Tasks):\n")
        for method in methods:
            if method in backward_transfer and backward_transfer[method]:
                f.write(f"{method}: {backward_transfer[method][0]:.2f}%\n")
        f.write("\n")
    
    # Final accuracies per task
    f.write("FINAL ACCURACIES PER TASK:\n")
    for method in methods:
        results = method_results[method]
        if not results:
            continue
        final_accs = results[-1]
        f.write(f"{method}: {[f'{acc:.2f}%' for acc in final_accs]}\n")
    
print(f"Saved experiment report to {report_path}")
print("Analysis complete!") 
