import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import timm
from collections import defaultdict
import random
from pathlib import Path
import os
import sys

# Force CUDA to be available
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Configuration and setup
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")

# Verify CUDA is available
if not torch.cuda.is_available():
    print("ERROR: CUDA is not available. This script requires GPU.")
    print("Please ensure:")
    print("1. You have NVIDIA drivers installed")
    print("2. You have PyTorch with CUDA support installed")
    print("3. You're running this on a node with GPUs")
    print("\nYou can install PyTorch with CUDA support using:")
    print("pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121")
    sys.exit(1)

# Set device explicitly to first GPU
device = torch.device("cuda:0")
print(f"Using device: {device}")
print(f"Device name: {torch.cuda.get_device_name(0)}")

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
torch.cuda.manual_seed_all(42)

# Setup cache directory
cache_dir = Path("./model_cache")
cache_dir.mkdir(exist_ok=True)
print(f"Cache directory: {cache_dir.absolute()}")

# Hyperparameters
CONFIG = {
    "num_epochs": 2,
    "batch_size": 32,
    "learning_rate": 0.0001,
    "samples_per_class": 1000
}

# Replay buffer sizes to test
REPLAY_BUFFER_SIZES = [10, 100, 250, 500]

# Data transformation pipeline
data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load and split MNIST dataset into tasks
def load_mnist_tasks():
    train_data = datasets.MNIST(root='./data', train=True, transform=data_transform, download=True)
    test_data = datasets.MNIST(root='./data', train=False, transform=data_transform, download=True)
    
    task_classes = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]
    train_subsets, test_subsets = [], []
    
    for classes in task_classes:
        train_indices = []
        for cls in classes:
            cls_indices = [i for i, lbl in enumerate(train_data.targets) if lbl == cls]
            train_indices.extend(random.sample(cls_indices, min(CONFIG["samples_per_class"], len(cls_indices))))
        test_indices = [i for i, lbl in enumerate(test_data.targets) if lbl in classes]
        train_subsets.append(Subset(train_data, train_indices))
        test_subsets.append(Subset(test_data, test_indices))
    
    return train_subsets, test_subsets, task_classes

# Model with task-specific heads
class ViTWithTaskHeads(nn.Module):
    def __init__(self, backbone, num_tasks, classes_per_task=1):
        super().__init__()
        self.backbone = backbone
        self.backbone.head = nn.Identity()
        self.task_heads = nn.ModuleList([
            nn.Linear(backbone.num_features, classes_per_task).to(device)
            for _ in range(num_tasks)
        ])
    
    def forward(self, x, task_id):
        features = self.backbone(x)
        output = self.task_heads[task_id](features)
        return output, features

# Training function
def train_task(model, loader, optimizer, loss_fn, task_id, task_classes, replay_loader=None):
    model.train()
    for epoch in range(CONFIG["num_epochs"]):
        total_loss, batches = 0, 0
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            target_binary = (target == task_classes[task_id][1]).float()
            
            if replay_loader:
                try:
                    r_data, r_target = next(replay_iter)
                except (StopIteration, NameError):
                    replay_iter = iter(replay_loader)
                    r_data, r_target = next(replay_iter)
                
                r_data, r_target = r_data.to(device), r_target.to(device)
                r_binary = torch.zeros_like(r_target, device=device).float()
                for t, (cls0, cls1) in enumerate(task_classes[:task_id]):
                    r_binary[r_target == cls1] = 1.0
                data = torch.cat([data, r_data])
                target_binary = torch.cat([target_binary, r_binary])
            
            optimizer.zero_grad()
            output, _ = model(data, task_id)
            
            # Reshape target to match output dimensions for BCE loss
            target_binary = target_binary.view(-1, 1)
            
            loss = loss_fn(output, target_binary)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            batches += 1
        
        print(f"Epoch {epoch + 1}, Task {task_id + 1}, Loss: {total_loss / batches:.6f}")

# Evaluation function
def evaluate_task(model, loader, task_id, task_classes):
    model.eval()
    correct, total = 0, 0
    
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            target_binary = (target == task_classes[task_id][1]).float()
            output, _ = model(data, task_id)
            
            # For binary classification with BCEWithLogitsLoss
            pred = (torch.sigmoid(output) > 0.5).float()
            
            # Make sure pred and target have compatible shapes for comparison
            if pred.dim() > 1:
                pred = pred.squeeze()
            if target_binary.dim() > 1:
                target_binary = target_binary.squeeze()
            
            correct += (pred == target_binary).sum().item()
            total += target_binary.size(0)
    
    return 100 * correct / total

# Main experiment
def run_ablation():
    train_datasets, test_datasets, task_classes = load_mnist_tasks()
    num_tasks = len(task_classes)
    
    # Results dictionary to store forgetting curves for each buffer size
    results = {size: [] for size in REPLAY_BUFFER_SIZES}
    task1_accuracy = {size: [] for size in REPLAY_BUFFER_SIZES}
    
    for buffer_size in REPLAY_BUFFER_SIZES:
        print(f"\n--- Running ablation with replay buffer size: {buffer_size} ---")
        
        backbone = timm.create_model("vit_tiny_patch16_224", pretrained=True, cache_dir=str(cache_dir), num_classes=0).to(device)
        model = ViTWithTaskHeads(backbone, num_tasks).to(device)
        
        replay_samples = []
        
        for task_id, (train_data, test_data) in enumerate(zip(train_datasets, test_datasets)):
            print(f"Training on task {task_id + 1}")
            train_loader = DataLoader(train_data, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=0)
            test_loader = DataLoader(test_data, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=0)
            
            optimizer = optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])
            loss_fn = nn.BCEWithLogitsLoss()
            
            replay_loader = None
            if task_id > 0 and replay_samples:
                replay_dataset = TensorDataset(torch.stack([x[0] for x in replay_samples]), torch.tensor([x[1] for x in replay_samples]))
                replay_loader = DataLoader(replay_dataset, batch_size=min(CONFIG["batch_size"], len(replay_samples)), shuffle=True, num_workers=0)
            
            train_task(model, train_loader, optimizer, loss_fn, task_id, task_classes, replay_loader)
            
            # Store samples for replay
            if task_id < num_tasks - 1:  # Don't need to store samples after the last task
                task_indices = train_data.indices
                samples_per_class = buffer_size // (2 * (task_id + 1))  # Evenly distribute across all seen classes
                
                new_samples = []
                for cls in task_classes[task_id]:
                    cls_indices = [i for i in task_indices if train_data.dataset.targets[i] == cls]
                    cls_indices = random.sample(cls_indices, min(samples_per_class, len(cls_indices)))
                    new_samples.extend([(train_data.dataset[i][0], train_data.dataset[i][1]) for i in cls_indices])
                
                # Add new samples to replay buffer
                replay_samples.extend(new_samples)
                
                # If buffer is too large, randomly subsample
                if len(replay_samples) > buffer_size:
                    replay_samples = random.sample(replay_samples, buffer_size)
            
            # Freeze current task head
            model.task_heads[task_id].requires_grad_(False)
            
            # Evaluate performance on all tasks
            task_accs = []
            for t in range(task_id + 1):
                t_loader = DataLoader(test_datasets[t], batch_size=CONFIG["batch_size"], shuffle=False, num_workers=0)
                acc = evaluate_task(model, t_loader, t, task_classes)
                task_accs.append(acc)
                
                # Track accuracy of Task 1 for forgetting curve
                if t == 0:
                    task1_accuracy[buffer_size].append(acc)
            
            print(f"Task accuracies after training on Task {task_id + 1}: {task_accs}")
            results[buffer_size].append(task_accs)
    
    # Plot forgetting curves for Task 1
    plt.figure(figsize=(10, 6))
    for size in REPLAY_BUFFER_SIZES:
        x_values = list(range(1, num_tasks + 1))
        plt.plot(x_values, task1_accuracy[size], marker='o', label=f"Buffer Size = {size}")
    
    plt.title('Forgetting Curve for Task 1 with Different Replay Buffer Sizes')
    plt.xlabel('After Training on Task #')
    plt.ylabel('Task 1 Accuracy (%)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig('replay_buffer_forgetting_curve.png')
    print("\nForgetting curve plot saved as 'replay_buffer_forgetting_curve.png'")
    
    # Save results
    result_dir = Path("./results")
    result_dir.mkdir(exist_ok=True)
    torch.save({
        "task1_accuracy": task1_accuracy,
        "all_results": results,
    }, result_dir / "replay_buffer_ablation_results.pt")
    
    print("Ablation study completed and results saved!")

if __name__ == "__main__":
    run_ablation() 
