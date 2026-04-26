# Macro Placement ML for EDA

Machine learning approaches for automated macro placement in chip design, implementing multiple state-of-the-art methods for Electronic Design Automation (EDA).

## Overview

This repository contains implementations of various ML-based macro placement techniques for VLSI chip design. Macro placement is a critical step in the physical design flow that determines the locations of large circuit blocks (macros) on a chip, directly impacting performance, power consumption, and routability.

**Primary Implementation:** MaskPlace - A reinforcement learning method using visual representation learning for fast chip placement.

**Reference Paper:** [MaskPlace: Fast Chip Placement via Reinforced Visual Representation Learning](https://arxiv.org/pdf/2211.13382.pdf) (NeurIPS 2022, Spotlight)

## Quick Start

### Prerequisites

```bash
Python >= 3.9
PyTorch >= 1.10
gym >= 0.21.0
matplotlib >= 3.7.1
tqdm
protobuf  # Required for ariane benchmark
```

### Installation

```bash
# Clone the repository
git clone https://github.com/Kavya-Agrawal/Macro_Placement_ML_for_EDA.git
cd Macro_Placement_ML_for_EDA

# Install dependencies
pip install torch>=1.10 gym>=0.21.0 matplotlib>=3.7.1 tqdm protobuf
```

### Running MaskPlace

```bash
cd maskplace
python PPO2.py
```

## Directory Structure

```
Macro_Placement_ML_for_EDA/
├── maskplace/              # MaskPlace implementation
│   ├── PPO2.py            # Main training script (PPO algorithm)
│   ├── environment/        # Placement environment
│   ├── models/            # Neural network architectures
│   ├── benchmarks/        # Circuit benchmarks
│   │   ├── adaptec1/     # ISPD 2005 benchmark
│   │   └── ariane/       # RISC-V processor design
│   └── utils/            # Helper functions
├── imgs/                  # Visualization and results
└── README.md
```

## Key Configuration Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--gamma` | Discount factor for RL | 0.99 |
| `--lr` | Learning rate | 3e-4 |
| `--batch_size` | Training batch size | 128 |
| `--pnm` | Number of placement modules per trajectory | - |
| `--benchmark` | Circuit benchmark (adaptec1, ariane, etc.) | adaptec1 |
| `--soft_coefficient` | Constraint actions based on wiremask | True |
| `--is_test` | Testing mode using trained agent | False |
| `--save_fig` | Save placement visualizations | False |
| `--seed` | Random seed for reproducibility | 42 |
| `--disable_tqdm` | Disable progress bar | False |
| `--log-interval` | Training status logging interval | 10 |

### Example Usage

```bash
# Training on adaptec1 benchmark
python PPO2.py --benchmark adaptec1 --lr 1e-4 --batch_size 256

# Testing with trained model
python PPO2.py --is_test --benchmark ariane --save_fig

# Custom configuration
python PPO2.py --benchmark adaptec2 --gamma 0.95 --pnm 50 --seed 123
```

## Benchmarks

### Included Benchmarks
- **adaptec1** - Small-scale ISPD 2005 benchmark
- **ariane** - RISC-V processor (requires protobuf)

### Additional Benchmarks
Download ISPD 2005 benchmarks from: http://www.cerc.utexas.edu/~zixuan/ispd2005dp.tar.xz

Supported designs: adaptec1, adaptec2, adaptec3, adaptec4, bigblue1, bigblue3

## Branch Architecture

This repository contains multiple branches, each implementing a different ML approach or variant for macro placement:

### Main Branches

#### `main`
- **Method:** MaskPlace with PPO (Proximal Policy Optimization)
- **Approach:** Reinforced visual representation learning using position and wire masks
- **Key Features:** 
  - Visual mask-based state representation
  - Fast convergence with sample efficiency
  - Zero-overlap constraint enforcement

#### Branch Variations (Different Models/Methods)

Each branch explores alternative approaches to macro placement:

**RL-Based Variants:**
- Different policy gradient methods (A2C, SAC, TD3)
- Alternative reward formulations
- Modified state representations
- Various neural network architectures

**Optimization-Based:**
- Analytical placement methods
- Gradient-based optimization
- Hybrid RL + analytical approaches

**Graph-Based:**
- Graph neural network (GNN) representations
- Attention mechanisms for netlist modeling
- Heterogeneous graph encodings

**Vision-Based:**
- CNN-based placement prediction
- Transformer architectures for layout
- Multi-scale visual feature extraction

**Baseline Comparisons:**
- DREAMPlace integration
- Simulated annealing variants
- Traditional EDA tool comparisons

### Switching Between Methods

```bash
# List all available branches
git branch -a

# Switch to a specific method
git checkout <branch-name>

# Example: Switch to graph-based method
git checkout graph-based-placement

# Return to main MaskPlace implementation
git checkout main
```

Each branch maintains its own setup instructions and may require different dependencies. Always check the branch-specific README after switching.

## Methodology

### MaskPlace Approach

1. **State Representation:** Chip canvas represented as position mask and wire mask
2. **Action Space:** Sequential placement of macros on grid locations
3. **Reward Function:** Weighted sum of:
   - Half-Perimeter Wire Length (HPWL)
   - Wire Length (Wirel)
   - Overlap penalty (enforces zero overlap)
4. **Training:** PPO with visual encoder-decoder architecture
5. **Post-processing:** Standard cell placement via force-directed method

### Key Metrics

- **HPWL (Half-Perimeter Wire Length):** Approximate wire length metric
- **Wirel:** Actual routed wire length
- **Overlap:** Percentage of macro overlap (0% target)
- **Congestion:** Routing congestion prediction
- **Density:** Cell placement density

## Results

MaskPlace achieves superior performance compared to baselines:

| Method | HPWL | Wirel | Overlap | Speed |
|--------|------|-------|---------|-------|
| DREAMPlace | Baseline | Baseline | 0-8% | ~1 min |
| Graph-based | 1.5-2x higher | 1.5-2x higher | 1-7% | Hours |
| DeepPR | 1.2-2.5x higher | 1.3-2.8x higher | 19-85% | Hours |
| **MaskPlace** | **Best** | **Best** | **0-1.9%** | **6-12 hrs** |

*Results averaged across ISPD 2005 benchmarks*

## Standard Cell Placement

After macro placement, use DREAMPlace or commercial tools for standard cell placement:

```bash
# Example with DREAMPlace (if installed)
dreamplace --config macro_placement_output.json
```

## Visualization

Generated placement figures show:
- **Placement:** Final macro locations (color-coded)
- **Position Mask:** Legal placement regions at timestep t
- **Wire Mask:** Connectivity-based attention mask
- **View Mask:** Combined visualization

Enable visualization:
```bash
python PPO2.py --save_fig
```

Outputs saved to `imgs/` directory.

## Extending the Framework

### Adding New Benchmarks

1. Place benchmark files in `maskplace/benchmarks/<benchmark_name>/`
2. Update benchmark loader in environment code
3. Configure grid dimensions and macro specifications

### Implementing New Methods

Different branches contain alternative implementations. To create a new method:

1. Create a new branch: `git checkout -b my-new-method`
2. Modify the RL algorithm in `PPO2.py` or create new trainer
3. Update state representation in environment
4. Adjust reward function as needed
5. Document changes in branch-specific README

## Common Issues

**Import Errors:** Ensure all dependencies installed with correct versions
**CUDA Errors:** PyTorch GPU support requires CUDA toolkit
**Memory Issues:** Reduce batch_size or use smaller benchmarks
**Protobuf Errors:** Install protobuf for ariane: `pip install protobuf`

## Citation

If you use this code, please cite:

```bibtex
@article{lai2022maskplace,
  title={MaskPlace: Fast Chip Placement via Reinforced Visual Representation Learning},
  author={Lai, Yao and Mu, Yao and Luo, Ping},
  journal={Advances in Neural Information Processing Systems},
  volume={35},
  pages={24019--24030},
  year={2022}
}
```

## License

See LICENSE file for details.

## Contributing

Contributions welcome! Each branch represents a different research direction. Please:
1. Fork the repository
2. Create feature branch from appropriate base branch
3. Submit pull request with clear description

## Acknowledgments

- Original MaskPlace paper authors
- ISPD benchmark providers
- PyTorch and OpenAI Gym communities

## Contact

For questions about specific implementations, check the branch you're working with or open an issue.
