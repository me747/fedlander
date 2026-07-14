import os
import argparse
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.run_fedrl import run_fedrl
from experiments.baseline_agent import run_single_agent

# experiment configs.                                         
# quick config, with small numbers to verify everything works
QUICK_CONFIG = {
    "n_rounds": 5,
    "local_steps": 500,
    "n_clients_per_round": 3,
    "n_envs": 2,
    "eval_every": 2,
    "n_eval_episodes": 3,
    "total_single_steps": 2000,
    "eval_every_single": 1000,
}

# full config, closer to what paper used
FULL_CONFIG = {
    "n_rounds": 125,
    "local_steps": 10000,
    "n_clients_per_round": 11,
    "n_envs": 8,        # paper had 64(had access to chameleon cloud during uni, so was kinda feasible to run training for longer) using 8 now to run on local machine
    "eval_every": 10,
    "n_eval_episodes": 10,
    "total_single_steps": 200000,
    "eval_every_single": 20000,
}

# run all experiments & save them to csv
def run_all_experiments(config, results_dir="results", seed=42):
    
    print(f"\n{'='*67}")
    print("FEDRL EXPERIMENT SUITE")
    print(f"config: {config}")
    print(f"{'='*67}\n")
    
    results = {}
    
    
    # experiment 1: standard federated learning (no DP)
    print("\n[Experiment 1/7] standard FL (no Differential Privacy)")
    print("-" * 50)
    results["fl_baseline"] = run_fedrl(
        n_rounds=config["n_rounds"],
        local_steps=config["local_steps"],
        n_clients_per_round=config["n_clients_per_round"],
        n_envs=config["n_envs"],
        use_dp=False,
        eval_every=config["eval_every"],
        n_eval_episodes=config["n_eval_episodes"],
        results_dir=results_dir,
        experiment_name="fl_baseline",
        seed=seed,
    )
    
    
    # experiments 2-4: DP-FL at varying epsilon values
    dp_epsilons = [5000, 500, 100]
    
    for i, eps in enumerate(dp_epsilons, start=2):
        exp_name = f"dp_fl_eps{eps}"
        print(f"\n[Experiment {i+1}/7] DP-FL (epsilon={eps})")
        print("-" * 50)
        results[exp_name] = run_fedrl(
            n_rounds=config["n_rounds"],
            local_steps=config["local_steps"],
            n_clients_per_round=config["n_clients_per_round"],
            n_envs=config["n_envs"],
            use_dp=True,
            dp_epsilon=eps,
            dp_sensitivity=0.2,
            eval_every=config["eval_every"],
            n_eval_episodes=config["n_eval_episodes"],
            results_dir=results_dir,
            experiment_name=exp_name,
            seed=seed,
        )
    
    
    # experiments 5-7: single agent baselines
    planets = ["earth", "mars", "moon"]
    
    for i, planet in enumerate(planets, start=5):
        exp_name = f"single_{planet}"
        print(f"\n[Experiment {i}/7] Single-Agent Baseline ({planet.capitalize()})")
        print("-" * 50)
        results[exp_name] = run_single_agent(
            planet=planet,
            total_steps=config["total_single_steps"],
            n_envs=config["n_envs"],
            eval_every=config["eval_every_single"],
            results_dir=results_dir,
            experiment_name=exp_name,
            seed=seed,
        )
    
    
    # summary
    print(f"\n{'='*67}")
    print("ALL EXPERIMENTS COMPLETE")
    print(f"{'='*67}\n")
    print(f"\nresults saved in: {results_dir}/")
    print("\nbest rewards per experiment:")
    for name, metrics in results.items():
        best = metrics.get_best_reward()
        if best is not None:
            print(f"  {name:25s}: {best:>7.1f}")
    
    return results

# CLI                                                           
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run all FedRL experiments")
    
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--quick", action="store_true", help="quick test run (fewer rounds, steps & envs. etc.)")
    mode.add_argument("--full",  action="store_true", help="full paper-scale run)")
    
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--seed",        type=int, default=42)
    args = parser.parse_args()
    
    config = FULL_CONFIG if args.full else QUICK_CONFIG
    
    run_all_experiments(
        config=config,
        results_dir=args.results_dir,
        seed=args.seed,
    )
