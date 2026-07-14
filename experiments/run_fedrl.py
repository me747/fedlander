import numpy as np
import os 
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.lunar_env import get_client_envs
from federated.client import FederatedClient
from federated.server import FederatedServer
from agent.ppo_agent import PPOAgent
from utils.logger import MetricsLogger, setup_logging, Timer


def build_clients(client_configs, n_envs=4, use_dp=False,
                  dp_epsilon=None, dp_sensitivity=0.2, seed=0):
    """
    builds a list of FederatedClient objects using config dicts from get_client_envs()
    
    Args:
        client_configs: list of dicts from get_client_envs()
        n_envs:         parallel envs per client
        use_dp:         whether clients us DP before sending wts.
        dp_epsilon:     privacy budget
        dp_sensitivity: L2 clipping bound
        seed:           random seed (each client gets seed + index)
    
    Returns:
        list of FederatedClient objects
    """
    clients = []
    for i, cfg in enumerate(client_configs):
        client = FederatedClient(
            client_id=cfg["name"],
            gravity=cfg["gravity"],
            wind_power=cfg["wind_power"],
            turbulence_power=cfg.get("turbulence_power", 0.0),
            n_envs=n_envs,
            seed=seed + i, # to ensure each client behaves slightly different
            use_dp=use_dp,
            dp_epsilon=dp_epsilon,
            dp_sensitivity=dp_sensitivity,
        )
        clients.append(client)
    return clients


def get_initial_weights(client_configs, n_envs=2, seed=0):
    
# creating a dummy agent just to get the ini. wt. shape
    
    tmp_agent = PPOAgent(
        gravity=client_configs[0]["gravity"],
        n_envs=n_envs,
        seed=seed,
        verbose=0,
    )
    weights = tmp_agent.get_weights()
    tmp_agent.close()
    return weights


def evaluate_global_model(server, eval_configs, n_envs=2, n_episodes=5):
    """
    test current global model on all test envs, temporarily load global wts. into a fresh agent and evaluate
    
    Args:
        server:       FederatedServer holding global wts.
        eval_configs: list of env configs to evaluate on
        n_envs:       parallel envs per evaluation agent
        n_episodes:   episodes per env.
    
    Returns:
        mean_reward: float, avg. across all eval envs.
    """
    global_weights = server.get_global_weights()
    all_rewards = []
    
    for cfg in eval_configs:
        agent = PPOAgent(
            gravity=cfg["gravity"],
            wind_power=cfg["wind_power"],
            turbulence_power=cfg.get("turbulence_power", 0.0),
            n_envs=n_envs,
            verbose=0,
        )
        agent.set_weights(global_weights)
        reward = agent.evaluate(n_episodes=n_episodes)
        all_rewards.append(reward)
        agent.close()
    
    return float(np.mean(all_rewards))


def run_fedrl(
    n_rounds=50,
    local_steps=5000,
    n_clients_per_round=5,
    n_envs=4,
    use_dp=False,
    dp_epsilon=500,
    dp_sensitivity=0.2,
    eval_every=5,
    n_eval_episodes=5,
    results_dir="results",
    experiment_name="fedrl_baseline",
    seed=42,
):
    """
    main FedRL training loop
    
    Args:
        n_rounds:            total FL rounds to run
        local_steps:         timesteps each client trains per round
        n_clients_per_round: how many clients to sample per round
        n_envs:              parallel envs per client
        use_dp:              whether to use DP?
        dp_epsilon:          DP's privacy budget
        dp_sensitivity:      DP's clipping bound
        eval_every:          eval global model every N rounds
        n_eval_episodes:     evaluation episodes per env
        results_dir:         where to save results CSV
        experiment_name:     label for curr. experiment run
        seed:                random seed
    """
    np.random.seed(seed)
    setup_logging()
    
    print(f"\n{'='*67}")
    print(f"FedRL Training: {experiment_name}")
    print(f"Rounds: {n_rounds} | Local steps: {local_steps}")
    print(f"Clients/round: {n_clients_per_round}")
    print(f"DP: {use_dp} | epsilon: {dp_epsilon if use_dp else 'N/A'}")
    print(f"{'='*67}\n")
    
    
    # 1. build clients and server
    print("setting up envs..")
    client_configs = get_client_envs(n_envs_per_client=n_envs, seed=seed)
    print(f"total available clients: {len(client_configs)}")
    
    # Get ini. wts. for global model
    initial_weights = get_initial_weights(client_configs, n_envs=2, seed=seed)
    print(f"model size: {len(initial_weights):,} parameters\n")
    
    # create central server
    server = FederatedServer(
        initial_weights=initial_weights,
        n_clients_per_round=n_clients_per_round,
    )
    
    # create all clients
    print("initializing clients..")
    clients = build_clients(
        client_configs=client_configs,
        n_envs=n_envs,
        use_dp=use_dp,
        dp_epsilon=dp_epsilon,
        dp_sensitivity=dp_sensitivity,
        seed=seed,
    )
    client_ids = list(range(len(clients)))
    print(f"created {len(clients)} clients\n")
    
    # eval on various configs
    eval_configs = [
    {"gravity": -9.8,  "wind_power": 0.0,  "turbulence_power": 0.0, "name": "earth_no_wind"},
    {"gravity": -9.8,  "wind_power": 6.0,  "turbulence_power": 0.0, "name": "earth_wind6"},
    {"gravity": -3.73, "wind_power": 5.0,  "turbulence_power": 0.0, "name": "mars_wind5"},
    {"gravity": -3.73, "wind_power": 8.0,  "turbulence_power": 0.0, "name": "mars_wind8"},
    {"gravity": -1.62, "wind_power": 15.0, "turbulence_power": 0.0, "name": "moon_wind15"},
    {"gravity": -1.62, "wind_power": 7.0,  "turbulence_power": 0.0, "name": "moon_wind7"},
    {"gravity": -9.8,  "wind_power": 0.0,  "turbulence_power": 1.5, "name": "turbulence"},
]
    
    # log metrics
    os.makedirs(results_dir, exist_ok=True)
    save_path = os.path.join(results_dir, f"{experiment_name}.csv")
    metrics = MetricsLogger(save_path, experiment_name=experiment_name)
    
    
    # 2. FL training loop
    
    print("starting federated learning..\n")
    
    for round_num in range(1, n_rounds + 1):
        
        with Timer(f"Round {round_num}") as t:
            
            # a) server selects subset of clients to use this round
            selected_ids = server.select_clients(client_ids)
            selected_clients = [clients[i] for i in selected_ids]
            
            # b) train each selected client locally
            client_weight_updates = []
            client_step_counts = []
            
            for client in selected_clients:
                # give client curr. global model
                global_weights = server.get_global_weights()
                client.receive_global_weights(global_weights)
                
                # client trains locally and returns its updated wts.
                updated_weights, steps = client.send_update(local_steps)
                
                client_weight_updates.append(updated_weights)
                client_step_counts.append(steps)
            
            # c) server aggregates all received wts.
            new_global = server.aggregate(
                client_updates=client_weight_updates,
                client_sizes=client_step_counts,
            )
        
        # d) eval global model every "eval_every" rounds
        global_reward = None
        if round_num == 1 or round_num % eval_every == 0:
            global_reward = evaluate_global_model(
                server, eval_configs, n_envs=2, n_episodes=n_eval_episodes
            )
        
        # log curr. round
        metrics.log_round(
            round_num=round_num,
            global_reward=global_reward,
            n_clients=len(selected_ids),
            extra={
                "round_time_s": round(t.elapsed, 1),
                "use_dp": use_dp,
                "dp_epsilon": dp_epsilon if use_dp else None,
            }
        )
    
    
    # 3. final eval & cleanup
    
    print("\nrunning final evaluation on all environments..")
    final_reward = evaluate_global_model(
        server, client_configs, n_envs=2, n_episodes=10
    )
    print(f"final mean reward across all environments: {final_reward:.1f}")
    
    metrics.save()
    
    # clean up client envs
    for client in clients:
        client.close()
    
    print("\ntraining complete!")
    print(f"best reward: {metrics.get_best_reward():.1f}")
    print(f"results saved to: {save_path}")
    
    return metrics


# CLI
def parse_args():
    parser = argparse.ArgumentParser(description="run Federated RL on LunarLander")
    parser.add_argument("--rounds",      type=int,   default=30)
    parser.add_argument("--local_steps", type=int,   default=5000,)
    parser.add_argument("--clients_per_round", type=int, default=5)
    parser.add_argument("--n_envs",      type=int,   default=4)
    parser.add_argument("--use_dp",      action="store_true")
    parser.add_argument("--epsilon",     type=float, default=500)
    parser.add_argument("--eval_every",  type=int,   default=5)
    parser.add_argument("--results_dir", type=str,   default="results")
    parser.add_argument("--name",        type=str,   default="fedrl_run")
    parser.add_argument("--seed",        type=int,   default=42)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    run_fedrl(
        n_rounds=args.rounds,
        local_steps=args.local_steps,
        n_clients_per_round=args.clients_per_round,
        n_envs=args.n_envs,
        use_dp=args.use_dp,
        dp_epsilon=args.epsilon,
        eval_every=args.eval_every,
        results_dir=args.results_dir,
        experiment_name=args.name,
        seed=args.seed,
    )
