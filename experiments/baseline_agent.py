import numpy as np
import os
import argparse
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.ppo_agent import PPOAgent
from environment.lunar_env import PLANET_CONFIGS
from utils.logger import MetricsLogger, setup_logging


# eval envs. to test how well the agent generalizes
EVAL_CONFIGS = [
    {"gravity": -9.8,  "wind_power": 0.0,  "name": "earth_calm"},
    {"gravity": -9.8,  "wind_power": 6.0,  "name": "earth_windy"},
    {"gravity": -3.73, "wind_power": 5.0,  "name": "mars_light_wind"},
    {"gravity": -1.62, "wind_power": 7.0,  "name": "moon_windy"},
]


def run_single_agent(
    planet="earth",
    total_steps=150000,
    wind_power=0.0,
    n_envs=4,
    eval_every=10000,
    n_eval_episodes=5,
    results_dir="results",
    experiment_name=None,
    seed=42,
):
    # train one PPO agent on one environment
    np.random.seed(seed)
    setup_logging()
    
    if planet not in PLANET_CONFIGS:
        raise ValueError(f"Unknown planet '{planet}'. Choose: {list(PLANET_CONFIGS.keys())}")
    
    gravity = PLANET_CONFIGS[planet]["gravity"]
    
    if experiment_name is None:
        experiment_name = f"single_{planet}"
    
    print(f"\n{'='*67}")
    print(f"  single agent baseline: {experiment_name}")
    print(f"  planet: {planet} (gravity={gravity})")
    print(f"  total steps: {total_steps:,}")
    print(f"{'='*67}\n")
    
    
    # set up agent
    agent = PPOAgent(
        gravity=gravity,
        wind_power=wind_power,
        n_envs=n_envs,
        seed=seed,
        verbose=0,
    )
    
    print(f"model parameters: {len(agent.get_weights()):,}\n")
    
    # metrics logging
    os.makedirs(results_dir, exist_ok=True)
    save_path = os.path.join(results_dir, f"{experiment_name}.csv")
    metrics = MetricsLogger(save_path, experiment_name=experiment_name)
    
    # training loop 
    print("training..\n")
    
    steps_done = 0
    eval_round = 0
    
    while steps_done < total_steps: # keep training until total_steps are reached
        # training based on chunk size
        chunk = min(eval_every, total_steps - steps_done) # ensure not to overshoot total_steps
        agent.train(total_timesteps=chunk)
        steps_done += chunk
        eval_round += 1
        
        # eval on this agent's own env.
        own_reward = agent.evaluate(n_episodes=n_eval_episodes)
        
        # also eval on OTHER envs. to test generalization (cross-env. eval.)
        cross_rewards = {}
        for cfg in EVAL_CONFIGS:
            # create a temporary agent with different physics
            eval_agent = PPOAgent(gravity=cfg["gravity"], wind_power=cfg["wind_power"],
                                   n_envs=1, verbose=0)
            eval_agent.set_weights(agent.get_weights())
            cross_rewards[cfg["name"]] = eval_agent.evaluate(n_episodes=n_eval_episodes)
            eval_agent.close()
        
        cross_avg = float(np.mean(list(cross_rewards.values()))) # calc. avg. generalization
        
        print(f"steps: {steps_done:>8,}/{total_steps:,} | own env: {own_reward:>7.1f} | cross-env avg: {cross_avg:>7.1f}")
        
        metrics.log_round(
            round_num=eval_round,
            global_reward=own_reward,
            n_clients=1,
            extra={
                "steps_done": steps_done,
                "cross_env_avg": cross_avg,
                **{f"reward_{k}": i for k, i in cross_rewards.items()}, # add per-env. rewards to the log by renaming keys like "earth" → "reward_earth" & unpack them into this dict.
            }
        )
    
   
    # summary (after training is complete final eval.)
    print("\nfinal cross-environment evaluation:")
    for cfg in EVAL_CONFIGS:
        eval_agent = PPOAgent(gravity=cfg["gravity"], wind_power=cfg["wind_power"],
                               n_envs=1, verbose=0)
        eval_agent.set_weights(agent.get_weights())
        r = eval_agent.evaluate(n_episodes=20)
        eval_agent.close()
        print(f"  {cfg['name']:20s}: {r:>7.1f}")
    
    metrics.save()
    agent.close()
    
    print(f"\nsingle agent training complete!")
    print(f"best reward: {metrics.get_best_reward():.1f}")
    return metrics



# CLI                                                             
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="run single-agent baseline")
    parser.add_argument("--planet",     type=str,   default="earth",
                        choices=["earth", "moon", "mars"])
    parser.add_argument("--steps",      type=int,   default=100000)
    parser.add_argument("--wind",       type=float, default=0.0)
    parser.add_argument("--n_envs",     type=int,   default=4)
    parser.add_argument("--eval_every", type=int,   default=10000)
    parser.add_argument("--results_dir",type=str,   default="results")
    parser.add_argument("--seed",       type=int,   default=42)
    args = parser.parse_args()
    
    run_single_agent(
        planet=args.planet,
        total_steps=args.steps,
        wind_power=args.wind,
        n_envs=args.n_envs,
        eval_every=args.eval_every,
        results_dir=args.results_dir,
        seed=args.seed,
    )
