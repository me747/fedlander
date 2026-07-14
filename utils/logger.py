import csv
import os
import logging
import time
from datetime import datetime


def setup_logging(log_level=logging.INFO, log_file=None):
    """
    to setup logging for the whole project once, call this at the start of exp. script once
    
    Args:
        log_level: e.g. logging.INFO, logging.DEBUG
        log_file:  optional path to write logs to file
    """
    handlers = [logging.StreamHandler()]  # to always print to terminal
    
    if log_file: # optional incase file path is specified
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    # core setup
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%b %d %H:%M:%S",
        handlers=handlers,
    )


class MetricsLogger:
    # track & save experiment metrics to a CSV file

    def __init__(self, save_path, experiment_name="fedrl"):
        self.save_path = save_path # location of CSV
        self.experiment_name = experiment_name # storing exp. name for comparing later
        self.rows = [] # list to store training data
        self.start_time = time.time() # start timer
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    def log_round(self, round_num, global_reward=None, n_clients=None,
                  extra=None):
        """
        log metrics for 1 FL round
        
        Args:
            round_num:     which round e.g. 1 or 2
            global_reward: mean reward across eval. episodes
            n_clients:     no. of clients participated this round
            extra:         dict. with any extra fields to log
        """
        elapsed = time.time() - self.start_time # calc. elapsed time
        
        # create a dict. for this round
        row = {
            "experiment": self.experiment_name,
            "round": round_num,
            "elapsed_seconds": round(elapsed, 1),
            "global_reward": global_reward,
            "n_clients": n_clients,
        }
        
        if extra:
            row.update(extra)
        
        self.rows.append(row)
        
        # print progress to console
        reward_str = f"{global_reward:.1f}" if global_reward is not None else "N/A"
        print(f"[Round {round_num:3d}] reward={reward_str:>8s} | "
              f"clients={n_clients} | elapsed={elapsed:.0f}s")
    
    def save(self):
        # write logged rows to CSV
        if not self.rows:
            print("no data to save!")
            return
        
        fieldnames = list(self.rows[0].keys()) # get col. names
        
        with open(self.save_path, "w", newline="") as f: # open the file & write to path
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.rows)
        
        print(f"results saved to: {self.save_path}")
    
    def get_best_reward(self):
        # returns best reward seen so far
        rewards = [
            r["global_reward"] 
            for r in self.rows 
            if r.get("global_reward") is not None # r["global_reward"] (key) was giving an error for key missing, whereas adding .get() returned None safely
            ] 
        return max(rewards) if rewards else None # get best reward


class Timer:
    # for timing code blocks
    
    def __init__(self, name=""):
        self.name = name
        self.elapsed = 0.0
    
    def __enter__(self):
        self.start = time.time()
        return self
    
    def __exit__(self, *args): # apparently *args captures any exceptions that occur inside 'with' block 
        self.elapsed = time.time() - self.start 
        if self.name: # only print if label name is provided 
            print(f"[TIMER] {self.name}: {self.elapsed:.2f}s")


#  quick test                                                       
if __name__ == "__main__":
    setup_logging()
    
    logger = MetricsLogger("results/test_log.csv", experiment_name="test")
    
    # simulate training rounds
    for r in range(1, 6):
        logger.log_round(
            round_num=r,
            global_reward=67.0 + r * 10,
            n_clients=5,
            extra={"epsilon": 500}
        )
    
    logger.save()
    print(f"best reward: {logger.get_best_reward()}")
    
    # test timer
    with Timer("sleeping"):
        time.sleep(0.1)
    
    print("logger test passed!")
