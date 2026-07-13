import numpy as np
import logging

from agent.ppo_agent import PPOAgent
from utils.privacy import apply_differential_privacy

logger = logging.getLogger(__name__)


class FederatedClient:
    """
    1 participant in the FL system 
    each client has own local env, own PPO agent & optionally DP settings

    """

    def __init__(self, client_id, gravity, wind_power=0.0,
                 turbulence_power=0.0, n_envs=4, seed=0,
                 use_dp=False, dp_epsilon=None, dp_sensitivity=0.2):
        """
        Args:
            client_id    : unique id
            gravity      : mag. of gravitational force
            wind_power   : wind force (0 = no wind)
            n_envs       : no. of parallel envs
            use_dp       : rule whether to apply DP or not
            dp_epsilon   : privacy budget
            dp_sensitivity: L2 sensitivity bound for DP clipping
        """
        # store env. settings
        self.client_id = client_id
        self.gravity = gravity
        self.wind_power = wind_power
        self.use_dp = use_dp
        self.dp_epsilon = dp_epsilon
        self.dp_sensitivity = dp_sensitivity
        
        # create local PPO agent with this client's env
        self.agent = PPOAgent(
            gravity=gravity,
            wind_power=wind_power,
            turbulence_power=turbulence_power,
            n_envs=n_envs,
            seed=seed,
            verbose=0,
        )
        
        # track training history
        self.training_steps_done = 0
        self.round_history = []
        
        logger.info(f"client {client_id} initialized (gravity={gravity}, wind={wind_power})")
    
    def receive_global_weights(self, global_weights):
        """
        step 1 in each FL round: receive and load the global model.

        Args:
            global_weights: flat numpy wt. array from the server
        """
        self.weights_before_training = global_weights.copy()  # save for DP delta later
        self.agent.set_weights(global_weights)
        logger.debug(f"client {self.client_id}: loaded global weights")
    
    def local_train(self, local_steps):
        """
        step 2 in each FL round: train locally on own env
        
        Args:
            local_steps: no. of env steps to train for
        
        Returns:
            updated wts. after local training
        """
        logger.debug(f"client {self.client_id}: training for {local_steps} steps")
        self.agent.train(total_timesteps=local_steps)
        self.training_steps_done += local_steps
        
        updated_weights = self.agent.get_weights()
        return updated_weights
    
    def compute_weight_update(self, updated_weights):
        """
        calculate DELTA 
        
        Returns:
            delta: numpy array of weight changes
        """
        return updated_weights - self.weights_before_training
    
    def send_update(self, local_steps):
        """
        main method: it represents a full client round, train + compute update(with DP) + send to server
    
        Args:
            local_steps: no. of env steps to train for (training budget)
        
        Returns:
            weights_to_send: numpy array (full weights or delta, depending on mode)
            steps_done:      int (used for weighted FedAvg)
        """
        # train locally
        updated_weights = self.local_train(local_steps)
        
        if self.use_dp:
            # differential privacy mode:
            # 1. calculate diff. b/w wts.
            delta = self.compute_weight_update(updated_weights)
            
            # 2. apply DP noise to the diff.
            noisy_delta = apply_differential_privacy(
                delta=delta,
                sensitivity=self.dp_sensitivity,
                epsilon=self.dp_epsilon,
            )
            
            # 3. apply noisy delta to the wts. received from the server
            weights_to_send = self.weights_before_training + noisy_delta
            
            logger.debug(f"client {self.client_id}: applied DP (eps={self.dp_epsilon})")
        else:
            # standard FL: just send the full updated wts.
            weights_to_send = updated_weights
        
        # log curr. round
        self.round_history.append({
            "steps": local_steps,
            "total_steps": self.training_steps_done,
            "use_dp": self.use_dp,
        })
        
        return weights_to_send, local_steps
    
    def evaluate(self, n_episodes=10):
        """
        evaluate curr. model performance on this client's env
        
        Returns:
            mean_reward: float (>200 = "passed" for LunarLander scenario)
        """
        return self.agent.evaluate(n_episodes=n_episodes)
    
    def close(self):
        # clean up resources
        self.agent.close()


#  quick test   

if __name__ == "__main__":
    print("testing FederatedClient..")
    
    # create client
    client = FederatedClient(
        client_id="earth_test",
        gravity=-9.8,
        wind_power=0.0,
        n_envs=2,
        seed=0,
    )
    
    # simulate receiving global wts. from server
    initial_weights = client.agent.get_weights()
    print(f"weight vector size: {len(initial_weights)}")
    
    client.receive_global_weights(initial_weights)
    
    # local training
    print("training locally for 500 steps..")
    updated_weights, steps = client.send_update(local_steps=500)
    print(f"updated weights shape: {updated_weights.shape}")
    print(f"weight change (L2 norm): {np.linalg.norm(updated_weights - initial_weights):.4f}")
    
    client.close()
    print("client test passed!")
