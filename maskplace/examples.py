"""
Simple examples demonstrating how to use RL-based macro ordering with MaskPlace
"""

import os
import sys


def example1_basic_training():
    """Example 1: Basic training from scratch"""
    
    print("\n" + "="*80)
    print("EXAMPLE 1: Basic Training")
    print("="*80)
    
    from place_db_integrated import PlaceDB
    from rl_ordering import train_rl_ordering
    
    # Load benchmark
    print("\n1. Loading benchmark...")
    placedb = PlaceDB("ariane", ordering_method="topology")
    print(f"   Loaded {len(placedb.node_info)} macros")
    
    # Train RL agent
    print("\n2. Training RL agent...")
    agent = train_rl_ordering(
        placedb,
        num_episodes=100,  # Use more for real training (e.g., 1000)
        alpha=1.0,
        beta=0.1,
        checkpoint_dir='example_checkpoints',
        log_interval=10
    )
    
    print("\n3. Training complete!")
    print(f"   Best reward: {max(agent.episode_rewards):.4f}")
    print(f"   Model saved to: example_checkpoints/best_model.pt")


def example2_using_pretrained():
    """Example 2: Using a pre-trained model"""
    
    print("\n" + "="*80)
    print("EXAMPLE 2: Using Pre-trained Model")
    print("="*80)
    
    from place_db_integrated import PlaceDB
    
    checkpoint_path = "example_checkpoints/best_model.pt"
    
    if not os.path.exists(checkpoint_path):
        print(f"\nCheckpoint not found: {checkpoint_path}")
        print("Run example1_basic_training() first to train a model.")
        return
    
    # Load PlaceDB with RL ordering
    print("\n1. Loading PlaceDB with RL ordering...")
    placedb = PlaceDB(
        benchmark="ariane",
        ordering_method="rl",
        rl_checkpoint=checkpoint_path
    )
    
    # Get the learned ordering
    ordering = placedb.node_id_to_name
    
    print("\n2. RL-learned ordering (first 10 macros):")
    for i, node_name in enumerate(ordering[:10]):
        num_nets = len(placedb.node_to_net_dict[node_name])
        area = placedb.node_info[node_name]['x'] * placedb.node_info[node_name]['y']
        print(f"   {i}: {node_name:20s} | Nets: {num_nets:3d} | Area: {area:8d}")
    
    print("\n3. Ready to use this ordering with your placement algorithm!")


def example3_comparing_methods():
    """Example 3: Compare RL vs heuristic ordering"""
    
    print("\n" + "="*80)
    print("EXAMPLE 3: Comparing Ordering Methods")
    print("="*80)
    
    from place_db_integrated import PlaceDB
    
    # Load with heuristic ordering
    print("\n1. Heuristic ordering (topology-based):")
    placedb_heuristic = PlaceDB("ariane", ordering_method="topology")
    heuristic_order = placedb_heuristic.node_id_to_name[:10]
    print(f"   First 10: {heuristic_order}")
    
    # Load with RL ordering
    checkpoint_path = "example_checkpoints/best_model.pt"
    if os.path.exists(checkpoint_path):
        print("\n2. RL ordering:")
        placedb_rl = PlaceDB("ariane", ordering_method="rl", rl_checkpoint=checkpoint_path)
        rl_order = placedb_rl.node_id_to_name[:10]
        print(f"   First 10: {rl_order}")
        
        # Compare
        print("\n3. Comparison:")
        different = sum(1 for i in range(10) if heuristic_order[i] != rl_order[i])
        print(f"   Different macros in top 10: {different}/10")
    else:
        print("\n2. RL model not found. Train one first using example1_basic_training()")


def example4_custom_training():
    """Example 4: Custom training loop with monitoring"""
    
    print("\n" + "="*80)
    print("EXAMPLE 4: Custom Training Loop")
    print("="*80)
    
    from place_db_integrated import PlaceDB
    from rl_ordering import RLOrderingAgent
    import numpy as np
    
    # Setup
    print("\n1. Initializing...")
    placedb = PlaceDB("ariane", ordering_method="topology")
    agent = RLOrderingAgent(placedb, hidden_dim=256, lr=1e-4)
    
    # Training loop with custom monitoring
    print("\n2. Training with custom monitoring...")
    num_episodes = 50
    best_reward = float('-inf')
    best_ordering = None
    
    for episode in range(num_episodes):
        # Train one episode
        ordering, reward, loss, hpwl, cost = agent.train_episode(alpha=1.0, beta=0.1)
        
        # Track best
        if reward > best_reward:
            best_reward = reward
            best_ordering = ordering
            print(f"   Episode {episode:3d}: NEW BEST! Reward={reward:.4f}, HPWL={hpwl:.4f}")
        
        # Periodic logging
        if (episode + 1) % 10 == 0:
            recent_rewards = agent.episode_rewards[-10:]
            print(f"   Episode {episode+1:3d}: Avg Reward (10)={np.mean(recent_rewards):.4f}")
    
    print("\n3. Training complete!")
    print(f"   Best reward achieved: {best_reward:.4f}")
    print(f"   Best ordering (first 5): {[placedb.node_id_to_name[i] for i in best_ordering[:5]]}")
    
    # Save
    agent.save_checkpoint('example_checkpoints/custom_trained.pt')
    print("   Model saved to: example_checkpoints/custom_trained.pt")


def example5_inference_only():
    """Example 5: Fast inference without training"""
    
    print("\n" + "="*80)
    print("EXAMPLE 5: Fast Inference")
    print("="*80)
    
    from place_db_integrated import PlaceDB
    from rl_ordering import RLOrderingAgent
    import time
    
    checkpoint_path = "example_checkpoints/best_model.pt"
    
    if not os.path.exists(checkpoint_path):
        print(f"\nCheckpoint not found: {checkpoint_path}")
        print("Run example1_basic_training() first.")
        return
    
    # Load model
    print("\n1. Loading model...")
    placedb = PlaceDB("ariane", ordering_method="topology")
    agent = RLOrderingAgent(placedb)
    agent.load_checkpoint(checkpoint_path)
    
    # Generate orderings
    print("\n2. Generating orderings...")
    num_orderings = 5
    
    start_time = time.time()
    orderings = []
    for i in range(num_orderings):
        ordering = agent.generate_ordering(training=False)
        orderings.append(ordering)
    end_time = time.time()
    
    avg_time = (end_time - start_time) / num_orderings
    
    print(f"   Generated {num_orderings} orderings")
    print(f"   Average time: {avg_time:.4f} seconds")
    print(f"   Throughput: {1/avg_time:.2f} orderings/second")
    
    # Show first ordering
    print("\n3. First ordering (first 10 macros):")
    for i, idx in enumerate(orderings[0][:10]):
        print(f"   {i}: {placedb.node_id_to_name[idx]}")


def example6_integration_with_placement():
    """Example 6: Integration with actual placement algorithm"""
    
    print("\n" + "="*80)
    print("EXAMPLE 6: Integration with Placement Algorithm")
    print("="*80)
    
    from place_db_integrated import PlaceDB
    
    checkpoint_path = "example_checkpoints/best_model.pt"
    
    # This example shows how to integrate with your existing placement code
    print("\n1. Loading PlaceDB with RL ordering...")
    
    if os.path.exists(checkpoint_path):
        placedb = PlaceDB("ariane", ordering_method="rl", rl_checkpoint=checkpoint_path)
        print("   Using RL-learned ordering")
    else:
        placedb = PlaceDB("ariane", ordering_method="topology")
        print("   Using heuristic ordering (RL model not found)")
    
    # The ordering is now in placedb.node_id_to_name
    ordering = placedb.node_id_to_name
    
    print(f"\n2. Macro ordering ready ({len(ordering)} macros)")
    print("   Ordering:", ordering[:5], "...")
    
    print("\n3. Now use this ordering with your placement algorithm:")
    print("""
    # Pseudocode for integration:
    for node_name in placedb.node_id_to_name:
        macro = placedb.node_info[node_name]
        width = macro['x']
        height = macro['y']
        
        # Your placement logic here
        position = your_placement_function(macro, already_placed)
        
        # Update placement
        macro['placed_x'] = position[0]
        macro['placed_y'] = position[1]
    
    # Calculate final HPWL
    final_hpwl = calculate_hpwl(placedb)
    """)


def example7_hyperparameter_tuning():
    """Example 7: Hyperparameter tuning"""
    
    print("\n" + "="*80)
    print("EXAMPLE 7: Hyperparameter Tuning")
    print("="*80)
    
    from place_db_integrated import PlaceDB
    from rl_ordering import RLOrderingAgent
    import numpy as np
    
    placedb = PlaceDB("ariane", ordering_method="topology")
    
    # Try different hyperparameters
    configs = [
        {'alpha': 1.0, 'beta': 0.1, 'lr': 1e-4, 'hidden_dim': 128},
        {'alpha': 2.0, 'beta': 0.1, 'lr': 1e-4, 'hidden_dim': 256},
        {'alpha': 1.0, 'beta': 0.05, 'lr': 5e-5, 'hidden_dim': 256},
    ]
    
    print("\nTesting different hyperparameter configurations...")
    results = []
    
    for i, config in enumerate(configs):
        print(f"\n{i+1}. Config: alpha={config['alpha']}, beta={config['beta']}, "
              f"lr={config['lr']}, hidden_dim={config['hidden_dim']}")
        
        agent = RLOrderingAgent(placedb, hidden_dim=config['hidden_dim'], lr=config['lr'])
        
        # Train for a few episodes
        rewards = []
        for episode in range(20):  # Use more episodes for real tuning
            _, reward, _, _, _ = agent.train_episode(alpha=config['alpha'], beta=config['beta'])
            rewards.append(reward)
        
        avg_reward = np.mean(rewards[-10:])  # Average of last 10
        results.append({'config': config, 'avg_reward': avg_reward})
        print(f"   Average reward (last 10): {avg_reward:.4f}")
    
    # Find best config
    best = max(results, key=lambda x: x['avg_reward'])
    print(f"\nBest configuration:")
    print(f"   Config: {best['config']}")
    print(f"   Avg reward: {best['avg_reward']:.4f}")


def main():
    """Run examples based on user choice"""
    
    print("\n" + "="*80)
    print("RL-Based Macro Ordering Examples")
    print("="*80)
    print("\nAvailable examples:")
    print("  1. Basic training from scratch")
    print("  2. Using a pre-trained model")
    print("  3. Comparing ordering methods")
    print("  4. Custom training loop")
    print("  5. Fast inference")
    print("  6. Integration with placement")
    print("  7. Hyperparameter tuning")
    print("  0. Run all examples")
    
    if len(sys.argv) > 1:
        choice = sys.argv[1]
    else:
        choice = input("\nSelect example (0-7): ").strip()
    
    examples = {
        '1': example1_basic_training,
        '2': example2_using_pretrained,
        '3': example3_comparing_methods,
        '4': example4_custom_training,
        '5': example5_inference_only,
        '6': example6_integration_with_placement,
        '7': example7_hyperparameter_tuning,
    }
    
    if choice == '0':
        # Run all examples
        for key in sorted(examples.keys()):
            try:
                examples[key]()
            except Exception as e:
                print(f"\nExample {key} failed with error: {e}")
    elif choice in examples:
        examples[choice]()
    else:
        print(f"\nInvalid choice: {choice}")
        return
    
    print("\n" + "="*80)
    print("Example complete!")
    print("="*80)


if __name__ == "__main__":
    # Quick start
    if len(sys.argv) == 1:
        print("\nQuick start guide:")
        print("\n1. Train a model:")
        print("   python examples.py 1")
        print("\n2. Use the trained model:")
        print("   python examples.py 2")
        print("\n3. Compare with heuristic:")
        print("   python examples.py 3")
        print("\nOr run interactively:")
        print("   python examples.py")
    
    main()
