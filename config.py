import argparse

def get_args():
    parser = argparse.ArgumentParser()
    
    # Model architecture parameters
    parser.add_argument("--num-initial-weights", type=int, default=3)
    parser.add_argument("--hidden-dim-pre-weight", type=int, default=32)
    parser.add_argument("--num-lay-pre-weight", type=int, default=4)
    parser.add_argument("--in-channel-gl", type=int, default=32)
    parser.add_argument("--num-conv-lay-gl", type=int, default=1)
    parser.add_argument("--hidden-dim-conv-gl", type=int, default=32)
    parser.add_argument("--hidden-dim-gl", type=int, default=32)
    parser.add_argument("--in-channel-imp", type=int, default=32)
    parser.add_argument("--num-conv-lay-imp", type=int, default=1)
    parser.add_argument("--hidden-dim-conv-imp", type=int, default=32)
    parser.add_argument("--hidden-dim-readout-imp", type=int, default=32)
    parser.add_argument("--hidden-dim-updater", type=int, default=32)
    
    # Training parameters
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-epoch", type=int, default=50)
    parser.add_argument("--k-neigh", type=int, default=15)
    
    # Dropout rates
    parser.add_argument("--dropout-pre-weight", type=float, default=0.2)
    parser.add_argument("--dropout-gl", type=float, default=0.2)
    parser.add_argument("--dropout-imp", type=float, default=0.2)
    
    # Data parameters
    parser.add_argument("--dataset", type=str, default='PEMS-BAY')
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    return args
