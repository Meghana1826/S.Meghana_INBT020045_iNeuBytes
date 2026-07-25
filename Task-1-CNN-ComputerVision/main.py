import sys
import argparse
from train_baseline import run_part_a_baseline
from run_experiments import run_part_b_experiments
from train_final_model import run_part_c_final_model

def main():
    parser = argparse.ArgumentParser(description="Task 1: Computer Vision using CNN Models (CIFAR-10)")
    parser.add_argument('--part', type=str, choices=['A', 'B', 'C', 'all'], default='all',
                        help="Choose which part of Task 1 to execute (A: Baseline, B: Controlled Experiments, C: Final Model, all: Complete pipeline)")

    args = parser.parse_args()

    if args.part in ['A', 'all']:
        print("\n>>> Executing Part A: Traditional Baseline CNN...")
        run_part_a_baseline()

    if args.part in ['B', 'all']:
        print("\n>>> Executing Part B: Controlled Experiments...")
        run_part_b_experiments()

    if args.part in ['C', 'all']:
        print("\n>>> Executing Part C: Final Customized CNN...")
        run_part_c_final_model()

if __name__ == '__main__':
    main()
