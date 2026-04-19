"""
Comprehensive training and usage script for RL-based macro placement ordering.

This script demonstrates:
1. Training an RL agent to learn macro ordering
2. Evaluating the learned policy
3. Comparing RL vs heuristic ordering
4. Integration with MaskPlace
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# Import PlaceDB and RL modules
from place_db_integrated import PlaceDB
from rl_ordering import RLOrderingAgent, train_rl_ordering


def train_model(benchmark, num_episodes=1000, alpha=1.0, beta=0.1, 
                checkpoint_dir='rl_checkpoints', hidden_dim=256, lr=1e-4):
    """Train an RL model for macro ordering"""
    
    print("=" * 80)
    print(f"Training RL Ordering Model for {benchmark}")
    print("=" * 80)
    
    # Load benchmark
    print(f"\nLoading benchmark: {benchmark}")
    placedb = PlaceDB(benchmark, ordering_method="topology")  # Use heuristic for initialization
    placedb.debug_str()
    
    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Train agent
    print(f"\nStarting training for {num_episodes} episodes...")
    agent = train_rl_ordering(
        placedb,
        num_episodes=num_episodes,
        alpha=alpha,
        beta=beta,
        checkpoint_dir=checkpoint_dir,
        log_interval=10
    )
    
    # Plot training progress
    plot_training_progress(agent, checkpoint_dir)
    
    return agent


def plot_training_progress(agent, save_dir='rl_checkpoints'):
    """Plot and save training metrics"""
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot rewards
    axes[0].plot(agent.episode_rewards, alpha=0.3, label='Episode Reward')
    # Moving average
    window = 50
    if len(agent.episode_rewards) >= window:
        moving_avg = np.convolve(agent.episode_rewards, 
                                 np.ones(window)/window, mode='valid')
        axes[0].plot(range(window-1, len(agent.episode_rewards)), 
                    moving_avg, linewidth=2, label=f'{window}-Episode Moving Avg')
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Reward')
    axes[0].set_title('Training Rewards Over Time')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot losses
    axes[1].plot(agent.episode_losses, alpha=0.3, label='Episode Loss')
    # Moving average
    if len(agent.episode_losses) >= window:
        moving_avg = np.convolve(agent.episode_losses, 
                                 np.ones(window)/window, mode='valid')
        axes[1].plot(range(window-1, len(agent.episode_losses)), 
                    moving_avg, linewidth=2, label=f'{window}-Episode Moving Avg')
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Training Loss Over Time')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'training_progress.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nTraining progress plot saved to: {save_path}")
    plt.close()


def evaluate_ordering(placedb, ordering, method_name="Unknown"):
    """Evaluate a macro ordering"""
    
    print(f"\n{method_name} Ordering Evaluation:")
    print("-" * 60)
    
    # Convert ordering to names if needed
    if isinstance(ordering[0], int):
        ordered_names = [placedb.node_id_to_name[idx] for idx in ordering]
    else:
        ordered_names = ordering
    
    # Calculate metrics
    metrics = {}
    
    # 1. HPWL
    total_hpwl = 0.0
    for net_name in placedb.net_info:
        net_nodes = list(placedb.net_info[net_name]['nodes'].keys())
        if len(net_nodes) >= 2:
            xs = [placedb.node_info[name].get('raw_x', 0) for name in net_nodes]
            ys = [placedb.node_info[name].get('raw_y', 0) for name in net_nodes]
            if xs and ys:
                hpwl = (max(xs) - min(xs)) + (max(ys) - min(ys))
                total_hpwl += hpwl
    metrics['hpwl'] = total_hpwl
    
    # 2. Connectivity clustering
    # Measure how well connected macros are placed near each other
    avg_distance = 0.0
    count = 0
    for i, name1 in enumerate(ordered_names[:-1]):
        name2 = ordered_names[i + 1]
        # Check if they share nets
        nets1 = placedb.node_to_net_dict[name1]
        nets2 = placedb.node_to_net_dict[name2]
        shared_nets = len(nets1 & nets2)
        if shared_nets > 0:
            avg_distance += 1  # They are adjacent
            count += 1
    metrics['connectivity_score'] = avg_distance / count if count > 0 else 0
    
    # 3. Area distribution
    total_area = sum([
        placedb.node_info[name]['x'] * placedb.node_info[name]['y']
        for name in ordered_names
    ])
    metrics['total_area'] = total_area
    metrics['utilization'] = total_area / (placedb.max_height * placedb.max_width)
    
    # Print results
    print(f"  HPWL: {metrics['hpwl']:,.0f}")
    print(f"  Connectivity Score: {metrics['connectivity_score']:.4f}")
    print(f"  Total Area: {metrics['total_area']:,.0f}")
    print(f"  Utilization: {metrics['utilization']:.2%}")
    print(f"  First 10 macros: {ordered_names[:10]}")
    
    return metrics


def compare_orderings(benchmark, rl_checkpoint=None):
    """Compare RL ordering vs heuristic ordering"""
    
    print("\n" + "=" * 80)
    print("Comparing Ordering Methods")
    print("=" * 80)
    
    # Load with heuristic ordering
    print("\n1. Loading with HEURISTIC ordering...")
    placedb_heuristic = PlaceDB(benchmark, ordering_method="topology")
    heuristic_ordering = placedb_heuristic.node_id_to_name
    heuristic_metrics = evaluate_ordering(placedb_heuristic, heuristic_ordering, "Heuristic")
    
    # Load with RL ordering
    if rl_checkpoint and os.path.exists(rl_checkpoint):
        print("\n2. Loading with RL ordering...")
        placedb_rl = PlaceDB(benchmark, ordering_method="rl", rl_checkpoint=rl_checkpoint)
        rl_ordering = placedb_rl.node_id_to_name
        rl_metrics = evaluate_ordering(placedb_rl, rl_ordering, "RL")
        
        # Compare
        print("\n" + "=" * 80)
        print("Comparison Summary")
        print("=" * 80)
        
        improvement = {}
        for key in heuristic_metrics:
            if key == 'hpwl':
                # Lower is better
                improvement[key] = ((heuristic_metrics[key] - rl_metrics[key]) / 
                                   heuristic_metrics[key] * 100)
                print(f"{key.upper()}: RL is {improvement[key]:.2f}% {'better' if improvement[key] > 0 else 'worse'}")
            else:
                print(f"{key}: Heuristic={heuristic_metrics[key]:.4f}, RL={rl_metrics[key]:.4f}")
    else:
        print("\nNo RL checkpoint found. Skipping RL comparison.")
        print("Train a model first using: python train_rl_ordering.py --train --benchmark ariane")


def test_inference(benchmark, checkpoint_path, num_tests=10):
    """Test inference speed and consistency"""
    
    print("\n" + "=" * 80)
    print("Testing RL Model Inference")
    print("=" * 80)
    
    # Load PlaceDB
    placedb = PlaceDB(benchmark, ordering_method="topology")
    
    # Load agent
    agent = RLOrderingAgent(placedb)
    if os.path.exists(checkpoint_path):
        agent.load_checkpoint(checkpoint_path)
    else:
        print(f"Warning: Checkpoint not found at {checkpoint_path}")
        return
    
    # Generate multiple orderings
    orderings = []
    import time
    
    print(f"\nGenerating {num_tests} orderings...")
    start_time = time.time()
    
    for i in range(num_tests):
        ordering = agent.generate_ordering(training=False)
        orderings.append(ordering)
        if i == 0:
            print(f"First ordering: {ordering[:10]}...")
    
    end_time = time.time()
    avg_time = (end_time - start_time) / num_tests
    
    print(f"\nInference Statistics:")
    print(f"  Average time per ordering: {avg_time:.4f} seconds")
    print(f"  Throughput: {1/avg_time:.2f} orderings/second")
    
    # Check consistency (deterministic inference should give same result)
    all_same = all(orderings[0] == o for o in orderings)
    print(f"  Deterministic: {'Yes' if all_same else 'No'}")


def visualize_ordering(placedb, ordering, save_path='ordering_visualization.png'):
    """Visualize the macro ordering"""
    
    # Convert to names if needed
    if isinstance(ordering[0], int):
        ordered_names = [placedb.node_id_to_name[idx] for idx in ordering]
    else:
        ordered_names = ordering
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Ordering sequence colored by connectivity
    connectivities = [len(placedb.node_to_net_dict[name]) for name in ordered_names]
    
    axes[0].scatter(range(len(ordered_names)), connectivities, 
                   c=range(len(ordered_names)), cmap='viridis', s=50, alpha=0.6)
    axes[0].set_xlabel('Position in Ordering')
    axes[0].set_ylabel('Number of Connected Nets')
    axes[0].set_title('Macro Connectivity vs Position in Ordering')
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Area distribution
    areas = [placedb.node_info[name]['x'] * placedb.node_info[name]['y'] 
             for name in ordered_names]
    
    axes[1].bar(range(len(ordered_names)), areas, alpha=0.6)
    axes[1].set_xlabel('Position in Ordering')
    axes[1].set_ylabel('Macro Area')
    axes[1].set_title('Macro Area Distribution in Ordering')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nOrdering visualization saved to: {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='RL-based Macro Placement Ordering')
    
    parser.add_argument('--benchmark', type=str, default='ariane',
                       help='Benchmark name (default: ariane)')
    parser.add_argument('--mode', type=str, choices=['train', 'eval', 'compare', 'test'],
                       default='compare', help='Operation mode')
    parser.add_argument('--num_episodes', type=int, default=1000,
                       help='Number of training episodes (default: 1000)')
    parser.add_argument('--checkpoint_dir', type=str, default='rl_checkpoints',
                       help='Checkpoint directory (default: rl_checkpoints)')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Specific checkpoint path to load')
    parser.add_argument('--alpha', type=float, default=1.0,
                       help='HPWL weight in reward (default: 1.0)')
    parser.add_argument('--beta', type=float, default=0.1,
                       help='Cost weight in reward (default: 0.1)')
    parser.add_argument('--hidden_dim', type=int, default=256,
                       help='Hidden dimension for networks (default: 256)')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate (default: 1e-4)')
    
    args = parser.parse_args()
    
    # Set default checkpoint if not specified
    if args.checkpoint is None:
        args.checkpoint = os.path.join(args.checkpoint_dir, 'best_model.pt')
    
    if args.mode == 'train':
        # Train a new model
        agent = train_model(
            args.benchmark,
            num_episodes=args.num_episodes,
            alpha=args.alpha,
            beta=args.beta,
            checkpoint_dir=args.checkpoint_dir,
            hidden_dim=args.hidden_dim,
            lr=args.lr
        )
        
        # Evaluate the trained model
        print("\nEvaluating trained model...")
        placedb = PlaceDB(args.benchmark, ordering_method="rl", 
                         rl_checkpoint=args.checkpoint)
        ordering = placedb.node_id_to_name
        evaluate_ordering(placedb, ordering, "Trained RL")
        visualize_ordering(placedb, ordering, 
                          os.path.join(args.checkpoint_dir, 'rl_ordering_visualization.png'))
        
    elif args.mode == 'eval':
        # Evaluate existing model
        if not os.path.exists(args.checkpoint):
            print(f"Error: Checkpoint not found at {args.checkpoint}")
            print("Train a model first using --mode train")
            return
        
        placedb = PlaceDB(args.benchmark, ordering_method="rl", 
                         rl_checkpoint=args.checkpoint)
        ordering = placedb.node_id_to_name
        evaluate_ordering(placedb, ordering, "RL")
        visualize_ordering(placedb, ordering, 'rl_ordering_visualization.png')
        
    elif args.mode == 'compare':
        # Compare RL vs heuristic
        compare_orderings(args.benchmark, args.checkpoint)
        
    elif args.mode == 'test':
        # Test inference
        if not os.path.exists(args.checkpoint):
            print(f"Error: Checkpoint not found at {args.checkpoint}")
            return
        test_inference(args.benchmark, args.checkpoint, num_tests=10)


if __name__ == "__main__":
    # Example usage without command line args
    if len(sys.argv) == 1:
        print("=" * 80)
        print("RL-based Macro Placement Ordering - Quick Start Guide")
        print("=" * 80)
        print("\nUsage examples:")
        print("\n1. Train a new model:")
        print("   python train_rl_ordering.py --mode train --benchmark ariane --num_episodes 1000")
        print("\n2. Evaluate trained model:")
        print("   python train_rl_ordering.py --mode eval --benchmark ariane")
        print("\n3. Compare RL vs heuristic:")
        print("   python train_rl_ordering.py --mode compare --benchmark ariane")
        print("\n4. Test inference speed:")
        print("   python train_rl_ordering.py --mode test --benchmark ariane")
        print("\nRunning default comparison mode...")
        print("=" * 80)
        
        # Run default comparison
        compare_orderings('ariane', 'rl_checkpoints/best_model.pt')
    else:
        main()
