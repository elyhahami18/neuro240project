import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Load the results file
results_path = Path("/n/holylabs/LABS/sompolinsky_lab/Everyone/xupan/ely/gated-LoRA/experiments/exploration/results/experiment_results.pt")
data = torch.load(results_path)

# Print available data keys
print("Available data in the results file:")
for key in data.keys():
    print(f"- {key}")

# Extract the data
accuracy_results = data.get('results', {})
forgetting = data.get('task_forgetting', {})
initial_accuracies = data.get('initial_accuracies', {})
forward_transfer = data.get('forward_transfer', {})
backward_transfer = data.get('backward_transfer', {})

# Analyze accuracy results
print("\n--- Accuracy Results ---")
for method, results in accuracy_results.items():
    print(f"\nMethod: {method}")
    for task_id, task_accs in enumerate(results):
        print(f"  After Task {task_id+1}: {task_accs}")
    
    # Calculate average accuracy
    if results:
        final_accs = results[-1]
        avg_acc = sum(final_accs) / len(final_accs)
        print(f"  Final average accuracy: {avg_acc:.2f}%")

# Analyze forgetting
print("\n--- Forgetting Metrics ---")
for method, task_forgetting in forgetting.items():
    print(f"\nMethod: {method}")
    for task, values in task_forgetting.items():
        if values:
            print(f"  Task {task} forgetting: {values}")
            
            # Calculate forgetting as difference between max and final
            if len(values) > 1:
                max_acc = max(values)
                final_acc = values[-1]
                forget = max_acc - final_acc
                print(f"  Task {task} forgetting amount: {forget:.2f}%")

# Analyze transfer
print("\n--- Transfer Metrics ---")
print("Forward Transfer:")
for method, values in forward_transfer.items():
    if values:
        print(f"  {method}: {values[0]:.2f}%")

print("\nBackward Transfer:")
for method, values in backward_transfer.items():
    if values:
        print(f"  {method}: {values[0]:.2f}%")

# Create a plot of accuracy over tasks
plt.figure(figsize=(12, 6))
for method, results in accuracy_results.items():
    if not results:
        continue
    
    # Get final accuracies for each task
    final_result = results[-1]
    
    # Create x-axis labels for tasks
    tasks = [f"Task {i+1}" for i in range(len(final_result))]
    
    plt.plot(tasks, final_result, marker='o', label=method)

plt.title('Final Accuracy per Task')
plt.xlabel('Task')
plt.ylabel('Accuracy (%)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.savefig('task_accuracy.png')
print("\nPlot saved as 'task_accuracy.png'")

# If there's more than one task, create a forgetting visualization
if len(accuracy_results) > 0 and any(len(results) > 1 for method, results in accuracy_results.items()):
    plt.figure(figsize=(10, 6))
    
    for method, results in accuracy_results.items():
        if len(results) <= 1:
            continue
            
        # Focus on first task accuracy across training
        task1_acc = [result[0] for result in results]
        x_points = list(range(1, len(task1_acc) + 1))
        
        plt.plot(x_points, task1_acc, marker='x', label=f"{method} - Task 1")
    
    plt.title('Task 1 Accuracy Throughout Training')
    plt.xlabel('After Training on Task #')
    plt.ylabel('Accuracy (%)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig('forgetting_visualization.png')
    print("Plot saved as 'forgetting_visualization.png'")

print("\nAnalysis complete!") 
