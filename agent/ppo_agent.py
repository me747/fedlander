import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from environment.lunar_env import make_vectorized_env


class PPOAgent:
    """
    a wrapper around Stable-Baselines3's PPO.
    made specifically so
    - can be trained locally per FL client
    - can extract its weights to send to server
    - load new weights (from server's global model)
    """
    
    def __init__(self, gravity=-9.8, wind_power=0.0, turbulence_power=0.0,
                 n_envs=4, seed=0, verbose=0):
        """
        sets up the env and PPO model
        
        Args:
            gravity        : magnitude of gravity, and the negative sign implies direction i.e. downwards
            wind_power     : arbitary wind force (0 = no wind)
            turbulence_power: defines an arbitary value for turbulence, making it harder for the rover to land
            n_envs         : no. of parallel copies of envs
            seed           : random seed for reproducibility
            verbose        : logging level  
        """
        # storing inputs into the object (can be reused without being passed)
        self.gravity = gravity
        self.wind_power = wind_power
        self.n_envs = n_envs
        self.seed = seed
        
        # create env
        self.env = make_vectorized_env(
            gravity=gravity,
            wind_power=wind_power,
            turbulence_power=turbulence_power,
            n_envs=n_envs,
            seed=seed,
        )
        
        # create the PPO model, using MlpPolicy, this creates a policy network which decides for a given state: action prob.
        self.model = PPO(
            policy="MlpPolicy",
            env=self.env,
            learning_rate=3e-4,   
            n_steps=2048,         
            batch_size=64,        
            n_epochs=10,          
            gamma=0.99,           
            verbose=verbose,
            seed=seed,
        )
    
    def train(self, total_timesteps):
        """
        train the agent locally for fixed no. of steps
        
        Args:
            total_timesteps: no. of environment steps to train for
        """
        self.model.learn(total_timesteps=total_timesteps, reset_num_timesteps=False)
    
    def get_weights(self):
        """
        extract neural network weights & convert them into a single flat array
        
        Returns:
            1D array of all weights concatenated together
        """
        params = self.model.policy.parameters() # grab neural network weights layer by layer
        # flatten each layer's weights and concatenate them all
        flat_weights = np.concatenate([p.data.cpu().numpy().flatten() for p in params])
        return flat_weights
    
    def set_weights(self, flat_weights):
        """
        load the flat array back into model's parameters
        
        Args:
            flat_weights: 1D weight array
        """
        params = list(self.model.policy.parameters())
        
        # calculate out how many weights from the flat array belong to each layer
        i = 0
        for param in params:
            n = param.data.numel()  # get no. of elements in this layer
            layer_weights = flat_weights[i:i+n]
            
            # reshape back to original shape(tensor) and load into model
            param.data = torch.tensor(
                layer_weights.reshape(param.data.shape),
                dtype=param.data.dtype
            )
            i+=n
    
    def evaluate(self, n_episodes=10):
        """
        test the agent and return average reward over n_episodes
        
        Args:
            n_episodes: no. of test episodes to run
        
        Returns:
            mean reward(float) across episodes
        """
        rewards = []
        
        # creating a single env for evaluation (no parallelism needed)
        eval_env = make_vectorized_env(
            gravity=self.gravity,
            wind_power=self.wind_power,
            n_envs=1,
        )
        
        for episode in range(n_episodes):
            obs = eval_env.reset()
            done = False
            total_reward = 0.0
            
            while not done:
                # model looks at state & returns best action
                action, _ = self.model.predict(obs, deterministic=True) 
                obs, reward, done, info = eval_env.step(action) 
                # debugging*: vec env would return 'done' as a numpy array with the boolen value i.e. np.array([False]) & not a plain boolean
                # Wasn't sure if not np.array([False]) behaves the same way as not False,
                # Apparently though numpy single-element arrays unwrap to their scalar value when you use not on them
                # But if it were a plain python list it would fail, because a non-empty list is truthy & not [False] would have returned False
                done = done[0] # Regardless adding this line to explicitly take the first element out of the numpy array, for a cleaner approach
                total_reward += reward[0]  # even with 1 env, vectorized env returns array so take [0] 
            
            rewards.append(total_reward)
        
        eval_env.close()
        return float(np.mean(rewards))
    
    def close(self):
        """clean up env resources"""
        self.env.close()


# quick test to check if PPO agent works 
if __name__ == "__main__":
    print("quick testing PPO agent..")
    
    agent = PPOAgent(gravity=-9.8, wind_power=0.0, n_envs=2, verbose=0)
    
    print(f"Weights shape: {agent.get_weights().shape}")
    print(f"Num parameters: {len(agent.get_weights())}")
    
    print("train for 1000 steps..")
    agent.train(total_timesteps=1000)
    
    print("evaluating..")
    mean_reward = agent.evaluate(n_episodes=3)
    print(f"Mean reward after 1000 steps: {mean_reward:.1f}")
    
    # testing weight extraction & loading
    weights = agent.get_weights()
    agent.set_weights(weights)  # no-op
    print("Weight get/set test passed.") # if this prints consider no crash, & passed
    
    agent.close()
    print("PPO agent test complete.")
