# Continual Learning in Vision Transformers


## Overview

The project implements several techniques for continual learning on the MNIST dataset:
- **Fine-tuning**: Standard sequential training without memory
- **L2 Regularization & Elastic Weight Consolidation (EWC)**: Regularization-based approach (simple L2) and more advanced techniques using Fisher information matrix
- **Exemplar Replay**: Traditional memory replay with stored samples
- **Latent Replay**: Approach storing feature representations instead of raw images

We use a pre-trained Vision Transformer and fine-tuning to adapt efficiently to each task.

## Requirements

```
torch>=2.1.2
torchvision>=0.16.2
timm>=0.9.10
matplotlib>=3.7.0
numpy>=1.24.0
```

## Project Structure

- `src/main.py`: Main training script supporting multiple methods
- `src/latent_replay_ablation.py`: Ablation study for different latent buffer sizes
- `src/naive_replay_ablation.py`: Ablation study for different replay buffer sizes
- `results/`: Directory containing analysis scripts and result visualizations

## How to Run

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/neuro240project.git
   cd neuro240project
   ```

2. Install dependencies:
   ```
   pip install torch==2.1.2 torchvision==0.16.2 timm==0.9.10 matplotlib numpy
   ```

3. Run the main experiment:
   ```
   python src/main.py
   ```

4. Run ablation studies:
   ```
   python src/latent_replay_ablation.py
   python src/naive_replay_ablation.py
   ```

5. Analyze results:
   ```
   python results/analyze_results.py
   ```

## Reproducing Results

The key experimental results can be reproduced with the following steps:

1. **Main Experiment**: Run `python src/main.py` which compares all methods (fine-tuning, EWC, L2 regularization, exemplar replay, and latent replay) across 5 sequential MNIST binary classification tasks. Results are saved to the `results/` directory.

2. **Ablation Studies**: 
   - For latent replay buffer sizes: Run `python src/latent_replay_ablation.py`
   - For traditional replay buffer sizes: Run `python src/naive_replay_ablation.py`

3. **Visualize Results**:
   - The forgetting curves comparing different methods are automatically generated in `results/`
   - Additional analysis can be performed using `python results/analyze_results.py`

## Methodology

1. We split MNIST into 5 binary classification tasks: (0,1), (2,3), (4,5), (6,7), (8,9). This means task 1 entails classifying digit 0 versus digit 1, task 2 entails classifying digit 2 versus 3, and so on and so forth. 
2. A pre-trained ViT model with task-specific heads is trained sequentially
3. LoRA adapters are used to efficiently fine-tune the model
4. Memory strategies (EWC, replay, latent replay) are applied to prevent forgetting
5. Metrics tracked: accuracy per task, forgetting amount, forward/backward transfer

## Key Results

The plots in the `results/` directory show:
- `forgetting_visualization.png`: Comparison of Task 1 accuracy as training progresses through tasks
- `replay_buffer_forgetting_curve.png`: Impact of buffer size on forgetting in exemplar replay
- `latent_replay_buffer_forgetting_curve.png`: Impact of buffer size on forgetting in latent replay

Enjoy!
