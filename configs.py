import json
import os
from pathlib import Path


SESSION_CONFIG = {
    ## MDP-related Parameters
    "gamma": 0.99,              # Discount factor for future rewards
    "horizon": 100,              # Number of steps in each epoch                  *****     
    "alpha": 0.05,               # Weighting factor for reward function: alpha * delay_reward + (1-alpha) * loss_rate_reward
    
    ## Training-related Parameters
    "buffer_size": 1024,          # Size of the replay buffer
    "batch_size": 128,           # Batch size for training
    "eps": 0.95,                # Exploration rate for the agent
    "tau": 0.005,               # Rate of target network update
    "init_random_epochs": 32,   # Number of initial epochs with random exploration before training starts
    "actor_lr": 1e-6,           # Learning rate for the actor network           ***** <
    "critic_lr": 1e-3,          # Learning rate for the critic network          ***** >
    "nn_layers": [128, 64],    # Configuration of neural network layers

    ## Other Parameters
    "topology": "nsfnet",        # Network topology type: "tanet", "nsfnet", or "5node"
    "total_traffic": 1000,       # Total traffic in the network
    "max_broken_links": 0,      # Maximum number of broken links in the network
    "queue_capacity": 20,       # Maximum queue-length of routers in the network
    "time_limit": 1000000,      # Total simulation duration in terms of environment time
}
