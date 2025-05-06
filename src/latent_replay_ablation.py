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
    "samples_per_class": 1000,
    "lora_rank": 32,
    "lora_reg_lambda": 0.01
}

# Latent replay buffer sizes to test
LATENT_BUFFER_SIZES = [10, 100, 250, 500]

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

# LoRA layer implementation
class LoRALayer(nn.Module):
    def __init__(self, input_dim, output_dim, rank):
        super().__init__()
        self.A = nn.Parameter(torch.zeros(input_dim, rank))
        self.B = nn.Parameter(torch.zeros(rank, output_dim))
        self.weight = nn.Parameter(torch.randn(output_dim, input_dim))
        self.bias = nn.Parameter(torch.zeros(output_dim))
        self.dropout = nn.Dropout(0.1)
        nn.init.orthogonal_(self.A)
        nn.init.orthogonal_(self.B)
    
    def forward(self, x):
        delta_weight = self.dropout(self.A @ self.B)
        return x @ (self.weight.T + delta_weight) + self.bias
    
    def lora_regularization(self):
        return torch.norm(self.A, p=2) + torch.norm(self.B, p=2)

# Apply LoRA to Vision Transformer layers
def apply_lora_to_vit(model, rank):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and ("mlp" in name.lower() or "attn" in name.lower()) and any(f"blocks.{i}" in name for i in range(6, 12)):
            lora = LoRALayer(module.in_features, module.out_features, rank).to(device)
            lora.weight.data = module.weight.data.clone()
            lora.bias.data = module.bias.data.clone()
            parent_name, child_name = name.rsplit(".", 1) if "." in name else ("", name)
            parent = model if not parent_name else dict(model.named_modules())[parent_name]
            setattr(parent, child_name, lora)
    return model

# Compute total LoRA regularization
def compute_total_lora_reg(model):
    total_reg = 0.0
    for module in model.modules():
        if isinstance(module, LoRALayer):
            total_reg += module.lora_regularization()
    return total_reg

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
def train_task(model, loader, optimizer, loss_fn, task_id, task_classes, latent_buffer=None):
    model.train()
    for epoch in range(CONFIG["num_epochs"]):
        total_loss, batches = 0, 0
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            target_binary = (target == task_classes[task_id][1]).float()
                        
            optimizer.zero_grad()
            loss = 0
            
            # Forward pass and compute loss for current task data
            output, features = model(data, task_id)
            target_binary = target_binary.view(-1, 1)
            loss += loss_fn(output, target_binary)
            
            # If using latent replay, add loss for replayed features
            if latent_buffer and len(latent_buffer) > 0:
                # Sample from latent buffer
                num_replay = min(CONFIG["batch_size"], len(latent_buffer))
                indices = random.sample(range(len(latent_buffer)), num_replay)
                
                # Process each latent feature individually
                for i in indices:
                    feature, prev_task_id, binary_label = latent_buffer[i]
                    feature = feature.to(device).unsqueeze(0)  # Add batch dimension
                    
                    # Forward through the head of the previous task
                    latent_output = model.task_heads[prev_task_id](feature)
                    
                    # Create label tensor properly formatted for BCE loss
                    latent_label = torch.tensor([binary_label], device=device).view(-1, 1)
                    
                    # Add to loss
                    loss += loss_fn(latent_output, latent_label)
            
            # Add LoRA regularization if using LoRA
            loss += CONFIG["lora_reg_lambda"] * compute_total_lora_reg(model.backbone)
            
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
    results = {size: [] for size in LATENT_BUFFER_SIZES}
    task1_accuracy = {size: [] for size in LATENT_BUFFER_SIZES}
    
    for buffer_size in LATENT_BUFFER_SIZES:
        print(f"\n--- Running ablation with latent replay buffer size: {buffer_size} ---")
        
        backbone = timm.create_model("vit_tiny_patch16_224", pretrained=True, cache_dir=str(cache_dir), num_classes=0).to(device)
        backbone = apply_lora_to_vit(backbone, CONFIG["lora_rank"])
        model = ViTWithTaskHeads(backbone, num_tasks).to(device)
        
        latent_buffer = []
        
        for task_id, (train_data, test_data) in enumerate(zip(train_datasets, test_datasets)):
            print(f"Training on task {task_id + 1}")
            train_loader = DataLoader(train_data, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=0)
            test_loader = DataLoader(test_data, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=0)
            
            optimizer = optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])
            loss_fn = nn.BCEWithLogitsLoss()
            
            # Train on current task
            train_task(model, train_loader, optimizer, loss_fn, task_id, task_classes, latent_buffer)
            
            # Store latent representations for future replay
            if task_id < num_tasks - 1:  # Don't need to store after the last task
                print(f"Storing latent representations for task {task_id + 1}")
                
                # Determine samples per class to store
                samples_per_class = buffer_size // (2 * (task_id + 1))
                
                # Store latent representations
                task_indices = train_data.indices
                for cls in task_classes[task_id]:
                    cls_indices = [i for i in task_indices if train_data.dataset.targets[i] == cls]
                    cls_indices = random.sample(cls_indices, min(samples_per_class, len(cls_indices)))
                    
                    for idx in cls_indices:
                        img, lbl = train_data.dataset[idx]
                        img = img.unsqueeze(0).to(device)  # Add batch dimension
                        
                        # Generate and store feature representation
                        with torch.no_grad():
                            _, feat = model(img, task_id)
                        
                        # Store as (feature, task_id, binary_label)
                        binary_lbl = 1.0 if lbl == task_classes[task_id][1] else 0.0
                        latent_buffer.append((feat.squeeze(0).cpu(), task_id, binary_lbl))
                
                # If buffer is too large, randomly subsample
                if len(latent_buffer) > buffer_size:
                    latent_buffer = random.sample(latent_buffer, buffer_size)
                
                print(f"Latent buffer size: {len(latent_buffer)}")
            
            # Freeze current task head to prevent forgetting
            model.task_heads[task_id].requires_grad_(False)
            
            # Evaluate performance on all tasks seen so far
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
    for size in LATENT_BUFFER_SIZES:
        x_values = list(range(1, num_tasks + 1))
        plt.plot(x_values, task1_accuracy[size], marker='o', label=f"Buffer Size = {size}")
    
    plt.title('Forgetting Curve for Task 1 with Different Latent Replay Buffer Sizes')
    plt.xlabel('After Training on Task #')
    plt.ylabel('Task 1 Accuracy (%)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig('latent_replay_buffer_forgetting_curve.png')
    print("\nForgetting curve plot saved as 'latent_replay_buffer_forgetting_curve.png'")
    
    # Save results
    result_dir = Path("./results")
    result_dir.mkdir(exist_ok=True)
    torch.save({
        "task1_accuracy": task1_accuracy,
        "all_results": results,
    }, result_dir / "latent_replay_buffer_ablation_results.pt")
    
    print("Latent replay buffer ablation study completed and results saved!")

if __name__ == "__main__":
    run_ablation() 
