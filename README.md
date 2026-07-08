# fedlander (in Progress..)

Federated Reinforcement Learning With Differential Privacy on LunarLander

Multiple PPO agents train across heterogeneous environments (Moon, Earth, Mars gravity) without sharing raw experience data. 
A central server aggregates learned weights using FedAvg. Differential Privacy (Laplace mechanism) to prevent environment inference from weight updates.

Was Built from scratch for CS 595-03 - Decentralized Machine Learning Systems, Illinois Institute of Technology. (just a cleaner rehash & submission)

## Stack
- Python 3.10+
- Gymnasium (LunarLander-v3)
- Stable-Baselines3(PPO)
- NumPy

## Setup

```bash
pip install -r requirements.txt
```

## Project Structure

```
fedlander/
├── environment/        # LunarLander with custom planet physics
├── agent/              # PPO agent wrapper (working on it)
├── federated/          # FL server + client (coming soon)
├── utils/              # privacy, logging & plotting implementation  (TBD)
└── experiments/        # training scripts (TBD?)
```

