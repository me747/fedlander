import numpy as np
import logging

logger = logging.getLogger(__name__)

class FederatedServer:
    """
    central coordinator for FL, holds global model& aggregates client updates using FedAvg
    * can be modified later to experiment with another weighted variant updation method for updating global model

    """
    
    def __init__(self, initial_weights, n_clients_per_round=None):
        """
        Args:
            initial_weights     : starting wts. (from one of the clients)
            n_clients_per_round : no. of clients to sample per round
                                  None = use all clients
        """
        self.global_weights = initial_weights.copy() # global model starts a copy of initial wts.
        
        self.current_round = 0 # track round 

        # no. of clients participating per round
        self.n_clients_per_round = n_clients_per_round 
        
        self.round_history = [] # log history for analysis 
    
    def get_global_weights(self):
        
       # client calls this at the "START" of each round, returns current global model wts. 

        return self.global_weights.copy()
    
    def fedavg(self, client_weights_list, client_sizes=None):
        """
        FedAvg: element-wise wt. avg of client wts.
    
        Args:
            client_weights_list: list of numpy arrays with learned model wts. (one per client)
            client_sizes       : list of ints (training steps per client)
                                 if given, weigh clients by how much they trained
                                 if none, equal weighting
        
        Returns:
            new_global_weights: averaged numpy array
        """
        if len(client_weights_list) == 0: # when no clients responded 
            logger.warning("client weights not received, keeping old global model.")
            return self.global_weights.copy()
        
        # weigh all clients equally, if no training steps per client/client_size given
        if client_sizes is None:
            client_sizes = [1.0] * len(client_weights_list)
        
        total_size = sum(client_sizes) # adding up how much each client contributed
        
        # weighted avg. (larger clients contribute more) 
        new_weights = np.zeros_like(self.global_weights) # ini. empty vector
        for weights, size in zip(client_weights_list, client_sizes):
            new_weights += (size / total_size) * weights # computing wt. avg.
        
        return new_weights
    
    def aggregate(self, client_updates, client_sizes=None):
        """
        main agg. step called once per FL round, applies FedAvg & updates global model

        Args:
            client_updates: list of weight arrays from clients
            client_sizes:   list of ints (training steps per client)
        
        Returns:
            new global weights
        """
        self.current_round += 1
        
        logger.info(f"round {self.current_round}: aggregating {len(client_updates)} clients")
        
        # call & run fedavg.
        new_global = self.fedavg(client_updates, client_sizes)
        
        self.global_weights = new_global # update global stored model
        
        # save info. in round_history for debugging/plotting later
        self.round_history.append({
            "round": self.current_round,
            "n_clients": len(client_updates),
        })
        
        return self.global_weights.copy()
    
    def select_clients(self, all_client_ids):
        """
        randomly select a subset for this round
        
        Args:
            all_client_ids: list of client identifiers
        
        Returns:
            list of selected client ids
        """
        if self.n_clients_per_round is None: # check if subsampling is disabled
            return all_client_ids  # use all
        
        n = min(self.n_clients_per_round, len(all_client_ids)) # don't pick more than available clients
        selected = np.random.choice(all_client_ids, size=n, replace=False).tolist()
        return selected
    
    def get_round_info(self):
        # useful for logging & debugging
        return {
            "current_round": self.current_round,
            "global_weights_norm": float(np.linalg.norm(self.global_weights)),
        }
    
    

#  quick test                                                        
if __name__ == "__main__":
    print("testing Federated-Server..")
    
    # trying simulation with 3 clients with random weights
    w_size = 100 # 100-dim wt. vector
    initial_weights = np.zeros(w_size)
    
    server = FederatedServer(initial_weights, n_clients_per_round=2)
    
    # round 1
    fake_client_weights = [
        np.ones(w_size) * 1.0,   # client 0 trained to all-1s
        np.ones(w_size) * 2.0,   # client 1 trained to all-2s
        np.ones(w_size) * 3.0,   # client 2 trained to all-3s
    ]
    
    selected = server.select_clients([0, 1, 2])
    print(f"selected clients: {selected}")
    
    selected_weights = [fake_client_weights[i] for i in selected]
    new_global = server.aggregate(selected_weights)
    
    print(f"new global weights (first 3): {new_global[:3]}") # inspect only first 3 values, i.e expected to be avg. of selected clients wts.
    print(f"server round: {server.current_round}")
    print("server test passed!")
