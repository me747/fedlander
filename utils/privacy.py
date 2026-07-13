import numpy as np

def clip_delta(delta, sensitivity):
    """
    clip the wt. update to have L2 norm less than or equal to sensitivity
    
    Args:
        delta:       numpy array, wt. update from local training
        sensitivity: max. allowed L2 norm
    
    Returns:
        clipped_delta: numpy array with ||delta||_2 ≤ sensitivity
    """
    current_norm = np.linalg.norm(delta)
    if current_norm > sensitivity:
        # scale down delta to fit within sensitivity bound
        clipped = delta * (sensitivity / current_norm)
    else:
        clipped = delta.copy()
    
    return clipped


def add_laplace_noise(clipped_delta, sensitivity, epsilon):
    """
    add Laplace-distributed noise to clipped delta

    Args:
        clipped_delta: numpy array
        sensitivity:   max. allowed L2 bound (same as clip threshold)
        epsilon:       privacy budget
    
    Returns:
        noisy_delta: numpy array with added noise
    """
    noise_scale = sensitivity / epsilon # scale parameter for laplace noise
   
    # sample noise for every wt. parameter
    noise = np.random.laplace(loc=0.0, scale=noise_scale, size=clipped_delta.shape)
    
    return clipped_delta + noise


def apply_differential_privacy(delta, sensitivity, epsilon):
    """
     DP pipeline: clip -> add noise, this is the main function called by clients before sending updates
    
    Returns:
        privatized_delta: numpy array ready to send to server
    """
    clipped = clip_delta(delta, sensitivity) # step 1: clip 

    privatized = add_laplace_noise(clipped, sensitivity, epsilon) # step 2: add noise
    
    return privatized


def privacy_budget_used(n_rounds, epsilon_per_round):
    """
    simple approximation of total privacy budget used across all rounds

    Args:
        n_rounds: no. of FL rounds
        epsilon_per_round: epsilon used per round
    
    Returns:
        total_epsilon: float
    """
    return n_rounds * epsilon_per_round

# quick test                                       

if __name__ == "__main__":
    
    print("testing differential privacy utilities.")
    
    # simulate wt. update
    np.random.seed(42)
    delta = np.random.randn(1000) * 2.0  # taking a large delta to ensure norm >> 0.2
    
    print(f"original delta L2 norm: {np.linalg.norm(delta):.4f}")
    
    # test if clipping wt. update works (result should be < or equal to 0.2)
    clipped = clip_delta(delta, sensitivity=0.2)
    print(f"clipped delta L2 norm:  {np.linalg.norm(clipped):.4f} (should be < or equal 0.2)")
    assert np.linalg.norm(clipped) <= 0.2 + 1e-6, "clipping failed!" # check if clipped delta's norm is < or equal to 0.2
    
    # test noise at different epsilon values
    print("\nnoise scale at different epsilon values:")
    for eps in [100, 500, 1000, 5000]:
        noisy = apply_differential_privacy(delta, sensitivity=0.2, epsilon=eps)
        noise_added = np.linalg.norm(noisy - clipped) # calculate amt. of noise added from result and initial clipped wt. update
        print(f"  eps={eps:5d}: noise L2 norm = {noise_added:.4f}")
    
    print("\nDP utility test:") # to check if averaging clients is reducing noise significantly
    print("averaging 100 clients should reduce noise significantly")
    n_clients = 100 
    epsilons = [100, 500, 1000, 5000]
    
    for eps in epsilons: # create noise updates for 100 clients where 0 learning happened
        noisy_deltas = [
            apply_differential_privacy(np.zeros(100), sensitivity=0.2, epsilon=eps)
            for _ in range(n_clients)
        ]
        averaged = np.mean(noisy_deltas, axis=0)
        print(f"eps={eps}: avg noise norm after {n_clients} client avg = {np.linalg.norm(averaged):.6f}")
    
    print("\nall DP tests passed!")
