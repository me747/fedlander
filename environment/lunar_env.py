import gymnasium as gym
import numpy as np 
# initially considered using SubprocVecEnv for true parallel execution for processes, switched to DummyVecEnv though for faster debugging
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

# by default LunarLander only has Earth's gravity, to introduce "heterogenity" in the FL setup, define planet configurations 
# each client resides in a different environment/planet              

PLANET_CONFIGS = {
    "moon":  {"gravity": -1.62,  "description": "Low gravity lunar surface"},
    "earth": {"gravity": -9.8,   "description": "Standard Earth gravity"},
    "mars":  {"gravity": -3.73,  "description": "Medium gravity Mars surface"},
}

def make_single_env(gravity=-9.8, wind_power=0.0, turbulence_power=0.0, seed=0):
    """
    create a single LunarLander env with custom physics.
    
    Args:
        gravity        : magnitude of gravity, and the negative sign implies direction i.e. downwards
        wind_power     : arbitary wind force (0 = no wind)
        turbulence_power: defines an arbitary value for turbulence, making it harder for the rover to land
        seed           : random seed for reproducibility
    
    Returns:
        a function that creates the environment (needed by SubprocVecEnv/DummyVecEnv*)
    """
    def _init():
        env = gym.make(
            "LunarLander-v3",
            gravity=gravity,
            enable_wind=(wind_power > 0),
            wind_power=wind_power,
            turbulence_power=turbulence_power,
        )
        env.reset(seed=seed) 
        return env
    return _init

def make_vectorized_env(gravity=-9.8, wind_power=0.0, turbulence_power=0.0,
                        n_envs=4, seed=0):
    """
    create multiple parallel env to collect more data per training iteration

    Args:
        n_envs: no. of parallel copies of the env that will be ran
    
    Returns:
        vectorized environment that batches multiple env instances and return obs. as a single interface
    """
    env_fns = [
        make_single_env(gravity, wind_power, turbulence_power, seed=seed + i) for i in range(n_envs)
    ]
    
    # Using DummyVecEnv for simplicity right now, might try experimenting later with SubprocVecEnv
    vec_env = DummyVecEnv(env_fns)
    return vec_env

def get_client_envs(n_envs_per_client=4, seed=42):
    """
    for creating environment configs for all clients in the FL system.
    
    to simulate a federated setup
    Baseline: (without wind)
        - Client 0 is on Moon 
        - Client 1 is on Earth  
        - Client 2 is on Mars
            - Client 3 onwards are derived clients with random wind conditions
    
    Returns:
        list of dicts, each with env config for one client
    """
    clients = []
    
    # base client, 1 per planet
    for planet_name, config in PLANET_CONFIGS.items():
        clients.append({
            "name": f"{planet_name}_baseline",
            "gravity": config["gravity"],
            "wind_power": 0.0,
            "turbulence_power": 0.0,
        })
    
    # derived clients, same planets under different wind conditions (to introduce real-world heterogenity)
    rng = np.random.default_rng(seed)
    for planet_name, config in PLANET_CONFIGS.items():
        for i in range(2):  # 2 extra clients per planet i.e. 6 derived clients
            wind = float(rng.uniform(0, 15))  # random wind b/w 0-15 mag.
            clients.append({
                "name": f"{planet_name}_wind_{wind:.1f}",
                "gravity": config["gravity"],
                "wind_power": wind,
                "turbulence_power": 0.0,
            })
    
    return clients

# quick test 
if __name__ == "__main__":
    print("Testing environment creation...\n")
    
    clients = get_client_envs()
    print(f"Created {len(clients)} client environments:")
    for i, c in enumerate(clients):
        print(f"  Client {i}: {c['name']} | gravity={c['gravity']} | wind={c['wind_power']:.1f}")
    
    print("\nTesting single env...")
    env = make_vectorized_env(gravity=-9.8, wind_power=5.0, n_envs=2)
    obs = env.reset() # get initial observation
    print(f"Observation shape: {obs.shape}")
    print(f"Action space: {env.action_space}")
    env.close()
    print("Environment test passed!")
