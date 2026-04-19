# RL-Based Macro Ordering Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        MaskPlace Integration                     │
│                                                                   │
│  ┌───────────────┐      ┌──────────────┐      ┌──────────────┐ │
│  │   PlaceDB     │─────▶│ RL Ordering  │─────▶│  Placement   │ │
│  │  (Benchmark)  │      │    Agent     │      │  Algorithm   │ │
│  └───────────────┘      └──────────────┘      └──────────────┘ │
│         │                       │                      │         │
│         │                       │                      │         │
│    Node/Net Info          Macro Ordering           Final        │
│                                                   Placement      │
└─────────────────────────────────────────────────────────────────┘
```

## Training Workflow

```
┌──────────────────────────────────────────────────────────────────┐
│                         Training Loop                             │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
        ┌──────────────────────────────────────┐
        │  Initialize Episode                   │
        │  - Empty ordering: []                 │
        │  - Placed mask: all zeros             │
        └──────────────────────────────────────┘
                              │
                              ▼
        ┌──────────────────────────────────────┐
        │  For each macro in design:            │
        │                                        │
        │  1. Extract State Features            │
        │     - Global: HPWL, utilization, etc. │
        │     - Per-macro: area, connectivity   │
        │                                        │
        │  2. Actor Network Forward             │
        │     - Encode state                    │
        │     - Score each unplaced macro       │
        │     - Mask already placed             │
        │                                        │
        │  3. Sample Action                     │
        │     - Create categorical distribution │
        │     - Sample next macro to place      │
        │     - Log probability                 │
        │                                        │
        │  4. Critic Network Forward            │
        │     - Estimate state value            │
        │                                        │
        │  5. Update State                      │
        │     - Add macro to ordering           │
        │     - Mark as placed                  │
        └──────────────────────────────────────┘
                              │
                              ▼
        ┌──────────────────────────────────────┐
        │  Evaluate Complete Ordering           │
        │  - Calculate HPWL                     │
        │  - Calculate placement cost           │
        │  - Compute reward                     │
        │    reward = -(α*HPWL + β*cost)       │
        └──────────────────────────────────────┘
                              │
                              ▼
        ┌──────────────────────────────────────┐
        │  Policy Gradient Update               │
        │                                        │
        │  For each step:                       │
        │    advantage = reward - value         │
        │    actor_loss = -log_prob * advantage │
        │    critic_loss = advantage²           │
        │                                        │
        │  total_loss = actor_loss + 0.5*critic │
        │                                        │
        │  Backprop & optimizer step            │
        └──────────────────────────────────────┘
                              │
                              ▼
        ┌──────────────────────────────────────┐
        │  Save Checkpoint if Best              │
        └──────────────────────────────────────┘
```

## Neural Network Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                          Actor Network                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  State Features (20-dim)           Macro Features (N x 8)        │
│         │                                  │                      │
│         ▼                                  ▼                      │
│  ┌──────────────┐                  ┌──────────────┐             │
│  │ State Encoder│                  │Macro Encoder │             │
│  │   (MLP)      │                  │   (MLP)      │             │
│  │  3 layers    │                  │  1 layer     │             │
│  │  hidden=256  │                  │  hidden=128  │             │
│  │  LayerNorm   │                  │  LayerNorm   │             │
│  └──────────────┘                  └──────────────┘             │
│         │                                  │                      │
│         └────────────┬─────────────────────┘                     │
│                      ▼                                            │
│            ┌──────────────────┐                                  │
│            │  Concatenate     │                                  │
│            │  & Expand        │                                  │
│            └──────────────────┘                                  │
│                      │                                            │
│                      ▼                                            │
│            ┌──────────────────┐                                  │
│            │ Multi-Head       │                                  │
│            │ Attention        │                                  │
│            │ (4 heads)        │                                  │
│            └──────────────────┘                                  │
│                      │                                            │
│                      ▼                                            │
│            ┌──────────────────┐                                  │
│            │  Score Layer     │                                  │
│            │  (MLP 2 layers)  │                                  │
│            └──────────────────┘                                  │
│                      │                                            │
│                      ▼                                            │
│              Logits (N macros)                                   │
│                      │                                            │
│                      ▼                                            │
│            ┌──────────────────┐                                  │
│            │  Apply Mask      │                                  │
│            │  (placed = -∞)   │                                  │
│            └──────────────────┘                                  │
│                      │                                            │
│                      ▼                                            │
│         Masked Logits for Sampling                               │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                         Critic Network                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  State Features (20-dim)                                         │
│         │                                                         │
│         ▼                                                         │
│  ┌──────────────┐                                                │
│  │     MLP      │                                                │
│  │  3 layers    │                                                │
│  │  hidden=256  │                                                │
│  │  LayerNorm   │                                                │
│  └──────────────┘                                                │
│         │                                                         │
│         ▼                                                         │
│  ┌──────────────┐                                                │
│  │ Output Layer │                                                │
│  │  (Linear)    │                                                │
│  └──────────────┘                                                │
│         │                                                         │
│         ▼                                                         │
│    State Value (scalar)                                          │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

## Feature Extraction Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                    State Feature Extraction                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  Current Placement State                                         │
│         │                                                         │
│         ├─────────────┬──────────────┬────────────┐             │
│         ▼             ▼              ▼            ▼              │
│   ┌─────────┐  ┌──────────┐  ┌──────────┐  ┌─────────┐        │
│   │ # Placed│  │ Partial  │  │  Canvas  │  │  Avg    │        │
│   │ Macros  │  │  HPWL    │  │  Util.   │  │ Connect.│  ...   │
│   └─────────┘  └──────────┘  └──────────┘  └─────────┘        │
│         │             │              │            │              │
│         └─────────────┴──────────────┴────────────┘             │
│                              │                                    │
│                              ▼                                    │
│                   Global Features (10-dim)                       │
│                                                                   │
│                              +                                    │
│                                                                   │
│                   Aggregate Features (10-dim)                    │
│                              │                                    │
│                              ▼                                    │
│                    State Vector (20-dim)                         │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                   Macro Feature Extraction                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  For Each Macro:                                                 │
│         │                                                         │
│         ├────┬────┬────┬────┬────┬─────┬─────┬─────┐           │
│         ▼    ▼    ▼    ▼    ▼    ▼     ▼     ▼     ▼            │
│      Width Height Area Aspect #Nets  ID   Ext1  Ext2           │
│         │    │    │    │    │    │     │     │     │            │
│         └────┴────┴────┴────┴────┴─────┴─────┴─────┘           │
│                              │                                    │
│                              ▼                                    │
│                   Per-Macro Vector (8-dim)                       │
│                                                                   │
│         Stack for all N macros                                   │
│                              │                                    │
│                              ▼                                    │
│                  Macro Matrix (N x 8)                            │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

## Inference Flow

```
┌──────────────────────────────────────────────────────────────────┐
│                      Inference (Deployment)                       │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
        ┌──────────────────────────────────────┐
        │  Load Trained Model                   │
        │  - Actor weights                      │
        │  - Critic weights (not used)          │
        └──────────────────────────────────────┘
                              │
                              ▼
        ┌──────────────────────────────────────┐
        │  Initialize Inference                 │
        │  - ordering = []                      │
        │  - placed_mask = zeros(N)             │
        └──────────────────────────────────────┘
                              │
                              ▼
        ┌──────────────────────────────────────┐
        │  Greedy Decoding (no sampling)        │
        │                                        │
        │  For each step:                       │
        │    1. Extract features                │
        │    2. Actor forward pass              │
        │    3. Select argmax (deterministic)   │
        │    4. Update state                    │
        └──────────────────────────────────────┘
                              │
                              ▼
        ┌──────────────────────────────────────┐
        │  Return Ordering                      │
        │  - List of macro names/IDs            │
        │  - Ready for placement algorithm      │
        └──────────────────────────────────────┘
```

## Integration Points

```
┌─────────────────────────────────────────────────────────────────┐
│                    Integration with MaskPlace                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  Original PlaceDB:                                               │
│  ┌──────────────────────────────────────────────────┐           │
│  │ placedb = PlaceDB("benchmark")                    │           │
│  │ ordering = placedb.node_id_to_name  # Heuristic   │           │
│  └──────────────────────────────────────────────────┘           │
│                                                                   │
│  Modified PlaceDB (Backward Compatible):                        │
│  ┌──────────────────────────────────────────────────┐           │
│  │ # Option 1: Heuristic (default)                   │           │
│  │ placedb = PlaceDB("benchmark",                    │           │
│  │                   ordering_method="topology")     │           │
│  │                                                    │           │
│  │ # Option 2: RL-based                              │           │
│  │ placedb = PlaceDB("benchmark",                    │           │
│  │                   ordering_method="rl",           │           │
│  │                   rl_checkpoint="model.pt")       │           │
│  │                                                    │           │
│  │ # Ordering is in same format                      │           │
│  │ ordering = placedb.node_id_to_name                │           │
│  └──────────────────────────────────────────────────┘           │
│                                                                   │
│  Your Placement Algorithm (unchanged):                          │
│  ┌──────────────────────────────────────────────────┐           │
│  │ for node_name in placedb.node_id_to_name:        │           │
│  │     place_macro(node_name, ...)                   │           │
│  └──────────────────────────────────────────────────┘           │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

## File Dependencies

```
rl_ordering.py
    │
    ├─ Imports: torch, numpy
    ├─ Defines: ActorNetwork, CriticNetwork, RLOrderingAgent
    └─ Exports: train_rl_ordering()

place_db_integrated.py
    │
    ├─ Imports: rl_ordering (optional), original modules
    ├─ Defines: PlaceDB (extended)
    └─ Exports: PlaceDB class

train_rl_ordering.py
    │
    ├─ Imports: place_db_integrated, rl_ordering
    ├─ Defines: Training utilities, evaluation
    └─ Exports: Command-line interface

examples.py
    │
    ├─ Imports: place_db_integrated, rl_ordering
    └─ Defines: Usage examples
```

## Performance Considerations

```
┌─────────────────────────────────────────────────────────────────┐
│                     Computational Complexity                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  Training:                                                       │
│    Time per episode: O(N × (F + A))                             │
│      N = number of macros                                        │
│      F = feature extraction time                                │
│      A = network forward/backward time                           │
│                                                                   │
│  Inference:                                                      │
│    Time per ordering: O(N × F)                                  │
│      - No gradient computation                                   │
│      - No critic evaluation                                      │
│      - Deterministic (argmax)                                    │
│                                                                   │
│  Memory:                                                         │
│    Model size: ~10-50 MB (depends on hidden_dim)                │
│    Episode memory: O(N) for storing log_probs and values        │
│                                                                   │
│  Typical Performance:                                            │
│    - Small designs (<100 macros): ~10ms per ordering            │
│    - Medium designs (100-500 macros): ~50ms per ordering        │
│    - Large designs (>500 macros): ~200ms per ordering           │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```
