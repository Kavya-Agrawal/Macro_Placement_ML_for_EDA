# RL-Based Macro Placement Ordering for MaskPlace

This implementation integrates reinforcement learning-based macro ordering into MaskPlace, replacing the heuristic `get_node_id_to_name_topology` function with a learned policy using actor-critic RL.

## 🎯 Overview

The RL agent learns to order macros for placement by:
- **Actor Network**: Scores each macro at each step
- **Critic Network**: Estimates the value of the current state
- **Training**: Policy gradient with advantage (actor-critic)
- **Reward**: Based on HPWL (Half-Perimeter Wire Length) and placement cost

## 📁 File Structure

```
rl_ordering.py              # Core RL implementation (Actor, Critic, Agent)
place_db_integrated.py      # Modified PlaceDB with RL support
train_rl_ordering.py        # Training and evaluation scripts
README.md                   # This file
rl_checkpoints/            # Directory for saved models
```

## 🚀 Quick Start

### 1. Training a Model

```bash
python train_rl_ordering.py --mode train --benchmark ariane --num_episodes 1000
```

**Training Parameters:**
- `--num_episodes`: Number of training episodes (default: 1000)
- `--alpha`: HPWL weight in reward (default: 1.0)
- `--beta`: Cost weight in reward (default: 0.1)
- `--hidden_dim`: Neural network hidden dimension (default: 256)
- `--lr`: Learning rate (default: 1e-4)

### 2. Using Trained Model

```python
from place_db_integrated import PlaceDB

# Load with RL ordering
placedb = PlaceDB(
    benchmark="ariane",
    ordering_method="rl",
    rl_checkpoint="rl_checkpoints/best_model.pt"
)

# The node_id_to_name list now contains the RL-learned ordering
ordering = placedb.node_id_to_name
```

### 3. Comparing Methods

```bash
python train_rl_ordering.py --mode compare --benchmark ariane
```

This compares:
- Heuristic (topology-based) ordering
- RL-learned ordering

## 🔧 Integration with Existing MaskPlace Code

### Option 1: Direct Replacement

Replace your existing PlaceDB initialization:

```python
# OLD CODE
from place_db import PlaceDB
placedb = PlaceDB("ariane")

# NEW CODE
from place_db_integrated import PlaceDB
placedb = PlaceDB(
    benchmark="ariane",
    ordering_method="rl",  # or "topology" for heuristic
    rl_checkpoint="rl_checkpoints/best_model.pt"
)
```

### Option 2: Programmatic Training

```python
from place_db_integrated import PlaceDB
from rl_ordering import train_rl_ordering

# Load benchmark with heuristic ordering first
placedb = PlaceDB("ariane", ordering_method="topology")

# Train RL agent
agent = train_rl_ordering(
    placedb,
    num_episodes=1000,
    alpha=1.0,
    beta=0.1,
    checkpoint_dir='my_checkpoints'
)

# Use trained agent to get ordering
ordering_indices = agent.generate_ordering(training=False)
ordering_names = [placedb.node_id_to_name[idx] for idx in ordering_indices]
```

### Option 3: Custom Training Loop

```python
from place_db_integrated import PlaceDB
from rl_ordering import RLOrderingAgent

placedb = PlaceDB("ariane", ordering_method="topology")
agent = RLOrderingAgent(placedb, hidden_dim=256, lr=1e-4)

# Custom training loop
for episode in range(1000):
    ordering, reward, loss, hpwl, cost = agent.train_episode(alpha=1.0, beta=0.1)
    
    if episode % 100 == 0:
        print(f"Episode {episode}: Reward={reward:.4f}, HPWL={hpwl:.4f}")
    
    # Your custom logic here (e.g., early stopping, curriculum learning)

# Save best model
agent.save_checkpoint('my_best_model.pt')
```

## 🏗️ Architecture Details

### Actor Network
- **Input**: Global state features + per-macro features
- **Architecture**: 
  - State encoder (MLP with layer norm)
  - Macro feature encoder
  - Multi-head attention for macro scoring
  - Final scoring layer
- **Output**: Logits for each macro (masked for already placed)

### Critic Network
- **Input**: Global state features
- **Architecture**: MLP with layer norm
- **Output**: Estimated state value

### State Features (20-dimensional)
**Global Features (10):**
- Normalized number of placed macros
- Current partial HPWL
- Canvas utilization
- Average connectivity
- Placement progress
- (5 additional extensible features)

**Aggregate Features (10):**
- Average macro area
- (9 additional extensible features)

### Macro Features (8-dimensional per macro)
- Normalized width
- Normalized height
- Normalized area
- Aspect ratio
- Connectivity (number of nets)
- Original ID
- (2 extensible features)

## 📊 Training Process

1. **Episode Generation**:
   ```python
   for step in range(num_macros):
       - Extract state features
       - Actor scores all unplaced macros
       - Sample action from categorical distribution
       - Update state with placed macro
   ```

2. **Reward Calculation**:
   ```python
   reward = -(alpha * hpwl + beta * cost)
   ```

3. **Policy Update**:
   ```python
   advantage = reward - critic_value
   actor_loss = -log_prob * advantage
   critic_loss = advantage^2
   total_loss = actor_loss + 0.5 * critic_loss
   ```

## 🎛️ Hyperparameter Tuning

### Key Parameters

| Parameter | Default | Description | Tuning Tips |
|-----------|---------|-------------|-------------|
| `alpha` | 1.0 | HPWL weight | Increase for better wirelength |
| `beta` | 0.1 | Cost weight | Increase to penalize congestion |
| `lr` | 1e-4 | Learning rate | Decrease if unstable |
| `hidden_dim` | 256 | Network size | Increase for complex benchmarks |
| `num_episodes` | 1000 | Training length | Increase until convergence |

### Recommended Settings by Benchmark Size

**Small benchmarks (<100 macros):**
```python
hidden_dim=128, lr=1e-4, num_episodes=500
```

**Medium benchmarks (100-500 macros):**
```python
hidden_dim=256, lr=1e-4, num_episodes=1000
```

**Large benchmarks (>500 macros):**
```python
hidden_dim=512, lr=5e-5, num_episodes=2000
```

## 📈 Monitoring Training

The training script automatically generates:
1. **Training curves**: `rl_checkpoints/training_progress.png`
2. **Periodic checkpoints**: `rl_checkpoints/checkpoint_N.pt`
3. **Best model**: `rl_checkpoints/best_model.pt`

### Interpreting Results

Good training shows:
- Increasing average reward (less negative)
- Decreasing HPWL over episodes
- Stabilizing loss

```python
# Access training history
agent = RLOrderingAgent(placedb)
agent.load_checkpoint('rl_checkpoints/best_model.pt')

print(f"Best reward: {max(agent.episode_rewards):.4f}")
print(f"Final avg reward: {np.mean(agent.episode_rewards[-100:]):.4f}")
```

## 🔍 Evaluation & Testing

### Evaluate Trained Model
```bash
python train_rl_ordering.py --mode eval --benchmark ariane
```

### Test Inference Speed
```bash
python train_rl_ordering.py --mode test --benchmark ariane
```

### Compare with Baseline
```python
from train_rl_ordering import compare_orderings

compare_orderings(
    benchmark="ariane",
    rl_checkpoint="rl_checkpoints/best_model.pt"
)
```

## 🛠️ Advanced Usage

### Custom Reward Function

Modify `_evaluate_ordering` in `RLOrderingAgent`:

```python
def _evaluate_ordering(self, ordering):
    # Your custom placement evaluation
    hpwl = self._calculate_full_hpwl(ordered_names)
    
    # Add custom metrics
    congestion = self._calculate_congestion(ordered_names)
    timing = self._calculate_timing(ordered_names)
    
    # Custom cost
    cost = 0.5 * congestion + 0.3 * timing
    
    return hpwl / 1e6, cost
```

### Custom State Features

Extend `_extract_state_features` in `RLOrderingAgent`:

```python
def _extract_state_features(self, ordering, placed_mask):
    # ... existing features ...
    
    # Add your custom features
    custom_feature_1 = self._compute_custom_metric_1(ordering)
    custom_feature_2 = self._compute_custom_metric_2(ordering)
    
    features.extend([custom_feature_1, custom_feature_2])
    
    return torch.FloatTensor(features).to(self.device)
```

### Transfer Learning

Train on one benchmark, fine-tune on another:

```python
# Train on base benchmark
agent = train_rl_ordering(placedb_base, num_episodes=1000)
agent.save_checkpoint('base_model.pt')

# Fine-tune on target benchmark
placedb_target = PlaceDB("target_benchmark")
agent_target = RLOrderingAgent(placedb_target)
agent_target.load_checkpoint('base_model.pt')

# Continue training
for episode in range(500):  # Fewer episodes for fine-tuning
    agent_target.train_episode()
```

## 🐛 Troubleshooting

### Common Issues

**1. "RL ordering not available"**
- Ensure `rl_ordering.py` is in the same directory
- Check PyTorch installation: `pip install torch`

**2. Training doesn't converge**
- Reduce learning rate: `--lr 5e-5`
- Increase training episodes: `--num_episodes 2000`
- Adjust reward weights: `--alpha 1.0 --beta 0.05`

**3. Out of memory**
- Reduce hidden dimension: `--hidden_dim 128`
- Use CPU: Set `device='cpu'` in `RLOrderingAgent.__init__`

**4. Poor HPWL results**
- Increase alpha: `--alpha 2.0`
- Train longer: `--num_episodes 2000`
- Check that placement coordinates are loaded correctly

## 📚 Code Structure

### Key Classes

**`ActorNetwork`** (`rl_ordering.py`)
- Scores macros for placement
- Uses attention mechanism for context

**`CriticNetwork`** (`rl_ordering.py`)
- Estimates state value
- Guides policy learning

**`RLOrderingAgent`** (`rl_ordering.py`)
- Manages training loop
- Handles feature extraction
- Implements reward calculation

**`PlaceDB`** (`place_db_integrated.py`)
- Loads benchmark data
- Supports multiple ordering methods
- Backward compatible with original code

## 🔬 Experimental Features

### Curriculum Learning
```python
# Start with simple sub-problems, gradually increase complexity
for stage in range(3):
    alpha = 1.0 * (stage + 1)  # Increase HPWL importance
    beta = 0.1 / (stage + 1)   # Decrease cost importance
    
    agent.train_episode(alpha=alpha, beta=beta)
```

### Multi-Objective Optimization
```python
# Pareto-optimal ordering exploration
alphas = [0.5, 1.0, 2.0]
betas = [0.05, 0.1, 0.2]

for alpha in alphas:
    for beta in betas:
        agent = train_rl_ordering(placedb, alpha=alpha, beta=beta)
        # Evaluate and compare
```

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@software{rl_macro_ordering,
  title={RL-Based Macro Placement Ordering for MaskPlace},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/rl-macro-ordering}
}
```

## 📄 License

This implementation is provided as-is for research and educational purposes.

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- Advanced reward shaping
- Multi-agent coordination for large designs
- Integration with commercial placers
- Benchmark-specific tuning
- Distributed training

## 📧 Contact

For questions or issues, please open a GitHub issue or contact [your email].

---

**Happy training! 🚀**
