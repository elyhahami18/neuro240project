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
    "l2_lambda": 0.01,
    "replay_size": 100,
    "lora_rank": 32,
    "lora_reg_lambda": 0.01,
    "samples_per_class": 1000,
    "ewc_lambda": 1,
    "latent_buffer_size": 200
}

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

# Fisher Information Matrix for EWC
def compute_fisher_matrix(model, loader, task_id, loss_fn, task_classes):
    model.eval()
    fisher = defaultdict(float)
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        target_binary = (target == task_classes[task_id][1]).float()
        
        model.zero_grad()
        output, _ = model(data, task_id)
        
        # Reshape target to match output dimensions for BCE loss
        if output.dim() > target_binary.dim():
            target_binary = target_binary.view(-1, 1)
            
        loss = loss_fn(output, target_binary)
        loss.backward()
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                fisher[name] += (param.grad ** 2) * len(data)
    
    for name in fisher:
        fisher[name] /= len(loader.dataset)
    return fisher

# EWC penalty calculation
def compute_ewc_penalty(model, fishers, prev_params):
    penalty = 0.0
    for fisher, params in zip(fishers, prev_params):
        for name, param in model.named_parameters():
            if name in fisher:
                penalty += (fisher[name] * (param - params[name]) ** 2).sum()
    return penalty

# Training function
def train_task(model, loader, optimizer, loss_fn, method, task_id, task_classes, prev_params=None, replay_loader=None, latent_buffer=None, fishers=None, saved_params=None):
    model.train()
    for epoch in range(CONFIG["num_epochs"]):
        total_loss, batches = 0, 0
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            target_binary = (target == task_classes[task_id][1]).float()
            
            if method == "replay" and replay_loader:
                r_data, r_target = next(iter(replay_loader))
                r_data, r_target = r_data.to(device), r_target.to(device)
                r_binary = torch.zeros_like(r_target, device=device).float()
                for t, (cls0, cls1) in enumerate(task_classes):
                    r_binary[r_target == cls1] = 1.0
                data = torch.cat([data, r_data])
                target_binary = torch.cat([target_binary, r_binary])
            
            elif method == "latent_replay" and latent_buffer:
                if len(latent_buffer) > 0:
                    indices = random.sample(range(len(latent_buffer)), min(CONFIG["batch_size"], len(latent_buffer)))
                    latent_data = torch.stack([latent_buffer[i][0] for i in indices]).to(device)
                    latent_tasks = torch.tensor([latent_buffer[i][1] for i in indices]).to(device)
                    latent_labels = torch.tensor([latent_buffer[i][2] for i in indices]).to(device)
            
            optimizer.zero_grad()
            loss = 0
            if method == "latent_replay":
                output, features = model(data, task_id)
                
                # Reshape target to match output dimensions for BCE loss
                target_binary = target_binary.view(-1, 1)
                
                loss += loss_fn(output, target_binary)
                if latent_buffer and len(latent_buffer) > 0:
                    for i in range(len(latent_data)):
                        latent_output = model.task_heads[latent_tasks[i]](latent_data[i:i+1])
                        # Create a properly formatted tensor for BCEWithLogitsLoss
                        latent_label = torch.tensor([latent_labels[i]], device=device).view(-1, 1)
                        # Reshape output if needed
                        if latent_output.shape != latent_label.shape:
                            latent_output = latent_output.view_as(latent_label)
                        loss += loss_fn(latent_output, latent_label)
            else:
                output, features = model(data, task_id)
                # Reshape target to match output dimensions for BCE loss
                target_binary = target_binary.view(-1, 1)
                loss = loss_fn(output, target_binary)
            
            if "lora" in method.lower():
                loss += CONFIG["lora_reg_lambda"] * compute_total_lora_reg(model.backbone)
            
            if "l2" in method.lower():
                l2_loss = sum(torch.norm(p - pp, p=2) for p, pp in zip(model.parameters(), prev_params))
                loss += CONFIG["l2_lambda"] * l2_loss
            
            if method == "ewc" and fishers and saved_params:
                loss += CONFIG["ewc_lambda"] * compute_ewc_penalty(model, fishers, saved_params)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            batches += 1
        
        print(f"Epoch {epoch + 1}, Method: {method}, Task Loss: {total_loss / batches:.6f}")

# Evaluation function
def evaluate_task(model, loader, task_id, task_classes, debug_logits=False):
    model.eval()
    correct, total = 0, 0
    logits, targets = [], []
    
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
            
            if debug_logits:
                logits.append(output)
                targets.append(target_binary)
    
    if debug_logits and logits:
        logits = torch.cat(logits, dim=0)
        targets = torch.cat(targets, dim=0)
        print(f"\nTask {task_id + 1} Logits Debug:")
        print(f"Mean logits for class 0 (digit {task_classes[task_id][0]}): {logits[targets == 0].mean()}")
        print(f"Mean logits for class 1 (digit {task_classes[task_id][1]}): {logits[targets == 1].mean()}")
    
    return 100 * correct / total

# Main experiment
def run_experiment():
    methods = ["fft", "replay", "ewc", "fft_l2", "lora", "lora_l2", "latent_replay"]
    train_datasets, test_datasets, task_classes = load_mnist_tasks()
    num_tasks = len(task_classes)
    
    results = {m: [] for m in methods}
    forgetting = {m: {t: [] for t in [1, 2, 3]} for m in methods}
    initial_accs = {m: {} for m in methods}
    forward_trans = {m: [] for m in methods}
    backward_trans = {m: [] for m in methods}
    
    for method in methods:
        print(f"\nRunning method: {method}")
        backbone = timm.create_model("vit_tiny_patch16_224", pretrained=True, cache_dir=str(cache_dir), num_classes=0).to(device)
        if "lora" in method.lower():
            backbone = apply_lora_to_vit(backbone, CONFIG["lora_rank"])
        model = ViTWithTaskHeads(backbone, num_tasks).to(device)
        
        prev_params = [p.clone().detach() for p in model.parameters()]
        fishers, saved_params = [], []
        replay_loader, latent_buffer = None, []
        
        for task_id, (train_data, test_data) in enumerate(zip(train_datasets, test_datasets)):
            print(f"Training on task {task_id + 1}")
            train_loader = DataLoader(train_data, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=0)
            test_loader = DataLoader(test_data, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=0)
            
            optimizer = optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])
            loss_fn = nn.BCEWithLogitsLoss()
            
            if method == "replay":
                task_indices = train_data.indices
                replay_samples = []
                for cls in task_classes[task_id]:
                    cls_indices = [i for i in task_indices if train_data.dataset.targets[i] == cls]
                    cls_indices = random.sample(cls_indices, min(CONFIG["replay_size"], len(cls_indices)))
                    replay_samples.extend([(train_data.dataset[i][0], train_data.dataset[i][1]) for i in cls_indices])
                replay_dataset = TensorDataset(torch.stack([x[0] for x in replay_samples]), torch.tensor([x[1] for x in replay_samples]))
                replay_loader = DataLoader(replay_dataset, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=0)
            
            elif method == "latent_replay":
                train_task(model, train_loader, optimizer, loss_fn, method, task_id, task_classes, prev_params, replay_loader, latent_buffer, fishers, saved_params)
                task_indices = train_data.indices
                for cls in task_classes[task_id]:
                    cls_indices = [i for i in task_indices if train_data.dataset.targets[i] == cls]
                    cls_indices = random.sample(cls_indices, min(CONFIG["replay_size"], len(cls_indices)))
                    for idx in cls_indices:
                        img, lbl = train_data.dataset[idx]
                        img = img.unsqueeze(0).to(device)
                        with torch.no_grad():
                            _, feat = model(img, task_id)
                        binary_lbl = 1.0 if lbl == task_classes[task_id][1] else 0.0
                        latent_buffer.append((feat.squeeze(0).cpu(), task_id, binary_lbl))
                if len(latent_buffer) > CONFIG["latent_buffer_size"] * (task_id + 1):
                    latent_buffer = random.sample(latent_buffer, CONFIG["latent_buffer_size"] * (task_id + 1))
            else:
                train_task(model, train_loader, optimizer, loss_fn, method, task_id, task_classes, prev_params, replay_loader, latent_buffer, fishers, saved_params)
            
            model.task_heads[task_id].requires_grad_(False)
            
            if task_id == 0:
                t2_loader = DataLoader(test_datasets[1], batch_size=CONFIG["batch_size"], shuffle=False, num_workers=0)
                acc_t2 = evaluate_task(model, t2_loader, 1, task_classes)
                forward_trans[method].append(acc_t2)
                print(f"Forward Transfer (Task 1 -> Task 2): {acc_t2:.2f}%")
            
            if method == "ewc":
                fisher = compute_fisher_matrix(model, train_loader, task_id, loss_fn, task_classes)
                fishers.append(fisher)
                saved_params.append({n: p.clone().detach() for n, p in model.named_parameters()})
            
            if "l2" in method.lower():
                prev_params = [p.clone().detach() for p in model.parameters()]
            
            task_accs = []
            for t, test_data_t in enumerate(test_datasets):
                t_loader = DataLoader(test_data_t, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=0)
                debug = (t == 0 and task_id == num_tasks - 1)
                acc = evaluate_task(model, t_loader, t, task_classes, debug)
                task_accs.append(acc)
                
                if t in [1, 2, 3]:
                    forgetting[method][t].append(acc)
                
                if task_id == t:
                    initial_accs[method][t] = acc
            
            results[method].append(task_accs)
        
        if task_id == num_tasks - 1:
            accs_before = results[method][-2][:4]
            accs_after = results[method][-1][:4]
            bwt = np.mean([a - b for b, a in zip(accs_before, accs_after)])
            backward_trans[method].append(bwt)
            print(f"Backward Transfer (Task 5 -> Tasks 1-4): {bwt:.2f}%")
        
        latent_buffer = []
    
    result_dir = Path("./results")
    result_dir.mkdir(exist_ok=True)
    torch.save({
        "results": results,
        "task_forgetting": forgetting,
        "initial_accuracies": initial_accs,
        "forward_transfer": forward_trans,
        "backward_transfer": backward_trans
    }, result_dir / "experiment_results.pt")
    
    print("Experiment completed successfully!")

if __name__ == "__main__":
    run_experiment()
    
