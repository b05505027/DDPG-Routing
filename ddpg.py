import numpy as np
import random
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from environment import Environment
from typing import Dict, Tuple, List, Type
from utils import find_available_run_index
from data_collector import DataCollector
from pathlib import Path
from models import Critic, Actor
from replay_buffer import ReplayBuffer
from enum import Enum
from tqdm import tqdm
import json
import os
from events import TrafficEvent, RecoveryEvent, FailureEvent


class Mode(Enum):
    TRAIN = "train"
    TEST = "test"
    OSPF = "ospf"

class DDPGAgent:
    """DDPGAgent interacting with environment."""

    def __init__(
        self,
        env: Type[Environment],
        buffer_size: int,
        batch_size: int,
        gamma: float,
        eps: float,
        tau: float,
        init_random_epochs: int,
        horizon: int,
        actor_lr: float,
        critic_lr: float,
        dir_path: str,
        nn_layers: List[int],
        mode: Type[Mode],
        topology: str, 
    ):
        """Initialize."""

        topo_config = json.load(open(os.path.join('topology', topology, 'config_files', 'config.json')))
        self.topology = topology
        self.env = env
        self.s_dim = 3 * topo_config['num_links']
        self.a_dim = topo_config['num_links'] * 2
        self.buffer = ReplayBuffer(self.s_dim, self.a_dim, max_size=buffer_size, batch_size=batch_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.eps = eps
        self.tau = tau
        self.initial_random_epochs = init_random_epochs
        self.horizon = horizon
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.dir_path = dir_path
        self.nn_layers = nn_layers
        self.mode = mode

        self.collector = DataCollector(dir_path)

        self.initialize_models()
        self.initialize_memory_and_counters()

    def initialize_models(self):
        """Initializes the models for the agent."""
        self.device = torch.device("cpu")
        self.actor = Actor(self.s_dim, self.a_dim, nn_layers=self.nn_layers).to(self.device)
        self.critic = Critic(self.s_dim + self.a_dim, nn_layers=self.nn_layers).to(self.device)

        if self.mode == Mode.TRAIN:
            self.initialize_target_models()
            self.initialize_optimizers()

    def initialize_target_models(self):
        """Initializes target models for training."""
        self.actor_target = Actor(self.s_dim, self.a_dim, nn_layers=self.nn_layers).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target = Critic(self.s_dim + self.a_dim, nn_layers=self.nn_layers).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

    def initialize_optimizers(self):
        """Initializes optimizers for the agent."""
        self.actor_optimizer = optim.AdamW(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optimizer = optim.AdamW(self.critic.parameters(), lr=self.critic_lr)

    def initialize_memory_and_counters(self):
        """Initializes memory and counters for the agent."""
        self.transition = list()
        self.epochs = 0
        self.step = 0
    
    
    def select_action(self, state: np.ndarray, exploration_rate: float) -> np.ndarray:
        """
        Selects an action based on the current state and exploration rate.
        """

        # Check if it's a test for OSPF (Open Shortest Path First)
        if self.mode == Mode.OSPF:
            # Initialize the action array with 0.1 values
            action = np.full(2 * self.a_dim, 0.1)

            # Set actions to 1 for indices in LARGE_LINKS
            for index in self.env.large_links:
                action[2 * index] = 1
                action[2 * index + 1] = 1

            # Reshape the action array
            action = action.reshape(1, -1)
            return action

        # Take random actions during initial epochs unless it's a test
        if self.epochs < self.initial_random_epochs and self.mode == Mode.TRAIN:
            selected_action = np.random.uniform(0, 1, size=self.a_dim).reshape(1, -1)
        else:
            # Use the actor model for action selection
            selected_action = self.actor(torch.FloatTensor(state).to(self.device)).detach().cpu().numpy()

        # Add noise to the action if it's training
        if self.mode == Mode.TRAIN:
            # Apply beta distribution noise
            noise = exploration_rate * np.random.beta(0.5, 0.5, size=self.a_dim).reshape(1, -1)
            # Ensure action values are within valid range
            selected_action = np.clip(selected_action + noise, 0, 1.0)

        return selected_action

    def store_transition(self, state, next_state, action, reward, is_ratio, done):
        self.transition = [state.reshape(-1), action.reshape(-1), reward, next_state.reshape(-1), is_ratio, done]
        self.buffer.store(*self.transition)


    def update_model(self):
        """
        Update the model by gradient descent. Return the losses and gradients of actor and critic.
        """
        # Sample a batch from the replay buffer
        samples = self.buffer.sample_batch()
        state = torch.FloatTensor(samples["s"]).to(self.device)
        next_state = torch.FloatTensor(samples["next_s"]).to(self.device)
        action = torch.FloatTensor(samples["a"]).to(self.device)
        reward = torch.FloatTensor(samples["r"].reshape(-1, 1)).to(self.device)
        is_ratio = torch.FloatTensor(samples["is_ratio"].reshape(-1, 1)).to(self.device)        # Avis, is_ratio
        done = torch.FloatTensor(samples["done"].reshape(-1, 1)).to(self.device)

        # Calculate target values
        with torch.no_grad():
            next_action = self.actor_target(next_state)
            next_value = self.critic_target(next_state, next_action)
            # expected_value = reward + self.gamma * next_value * (1 - done)
            expected_value = reward + self.gamma * is_ratio * next_value * (1 - done)

        # Update critic
        critic_value = self.critic(state, action)
        critic_loss = F.mse_loss(critic_value, expected_value)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_grad_norm = nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=10)
        self.critic_optimizer.step()

        # Update actor
        actor_loss = -self.critic(state, self.actor(state)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_grad_norm = nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=10)
        self.actor_optimizer.step()

        # Soft update of target networks
        self._target_soft_update()

        return actor_loss.item(), critic_loss.item(), actor_grad_norm.item(), critic_grad_norm.item()

    

    def _target_soft_update(self):
        """Soft-update: target = tau*local + (1-tau)*target."""
        tau = self.tau
        
        for t_param, l_param in zip(
            self.actor_target.parameters(), self.actor.parameters()
        ):
            t_param.data.copy_(tau * l_param.data + (1.0 - tau) * t_param.data)
            
        for t_param, l_param in zip(
            self.critic_target.parameters(), self.critic.parameters()
        ):
            t_param.data.copy_(tau * l_param.data + (1.0 - tau) * t_param.data)
 

    def train(self, importance_sampling):
        """
        Train the agent by iteratively interacting with the environment.
        The training involves selecting actions, storing transitions, and updating models.
        """
        if self.mode != Mode.TRAIN:
            print("DDPG Session not in train mode can't execute train function.")
            return

        # Initialize the state by executing a simulation step with zero actions.
        simulation_output = self.env.simulation_step(np.zeros(self.a_dim, dtype=float))
        next_state = simulation_output['data'][0]  # the first item in data is next_state.

        # Calculate the total number of training epochs based on the event size and horizon.
        event_size = self.env.get_event_size()
        pbar = tqdm(range(1, event_size // self.horizon + 2))
        for self.epochs in pbar:
            
            # Calculate the current exploration rate based on the decay factor and the current epoch.
            exploration_rate = np.power(self.eps, self.epochs)
            score = 0  # Initialize the score for this epoch.
            is_score = 0 # Initialize the score_is for this epoch.
            acc_is_ratio = 1 # Initialize the accumulated is ratio.

            # Iterate over each timestep within the current epoch.
            for self.iteration in range(1, self.horizon + 1):
                state = next_state  # Update the current state.
                # Select an action based on the current state and exploration rate.
                action = self.select_action(state, exploration_rate=exploration_rate)

                # Execute a simulation step with the selected action and unpack the results.
                simulation_output = self.env.simulation_step(action)
                next_state, reward, delay, loss_rate, is_ratio = simulation_output['data']

                # Use a default is ratio of 1.0 if importance sampling is not enabled.
                is_ratio = is_ratio if importance_sampling else 1.0

                # Update the cumulative score.
                score += np.power(self.gamma, self.iteration - 1) * reward
                is_score += acc_is_ratio*np.power(self.gamma, self.iteration - 1) * reward
                acc_is_ratio *= is_ratio

                # Check if the epoch is done.
                done = self.iteration == self.horizon

                # Update the progress bar with the current status.
                pbar.set_description(f"timestep: {self.iteration}/{self.horizon} is_ratio: {is_ratio}, delay: {delay:.4f}, lossrate:{loss_rate:.4f}, num_broken_links:{len(self.env.broken_links)}")

                # Log various metrics.
                self.collector.log_data({'delays': delay, 'lossrates': loss_rate, 'traffics': self.env.get_current_traffic_amount(), 'broken_links_number': len(self.env.broken_links), 'broken_links': self.env.broken_links})
                # Save data into the replay buffer
                self.store_transition(state, next_state, action, reward, is_ratio, done)

            # Update the model and log additional metrics.
            if len(self.buffer) >= self.batch_size and self.epochs > self.initial_random_epochs:
                actor_loss, critic_loss, actor_grad_norm, critic_grad_norm = self.update_model()
                self.collector.log_data({'actor_losses': actor_loss, 'critic_losses': critic_loss, 'actor_grad_norm': actor_grad_norm, 'critic_grad_norm': critic_grad_norm})

            # Log the score and exploration rate for the current epoch. Also log the broken links
            self.collector.log_data({'scores': score, 'is_scores': is_score,  'exploration_rates': exploration_rate})
            self.collector.save_all_data2(action)

            self.env.traffic_manager.episode_end_traffic()
                    
            # Periodically plot and save collected data, and save the model's state.
            if self.epochs % 100 == 0:
                self.collector.plot_all_data()
                self.collector.save_all_data()
                torch.save(self.actor.state_dict(), self.dir_path / Path(f'models/actor_{self.epochs}.ckpt'))
                torch.save(self.critic.state_dict(), self.dir_path / Path(f'models/critic_{self.epochs}.ckpt'))

    def test(self, actor_path, critic_path):

        if self.mode != Mode.TEST  and self.mode != Mode.OSPF:
            print("DDPG Session not in test or ospf mode can't execute test function.")
            return

        if self.mode == Mode.TEST:
            self.actor.load_state_dict(torch.load(actor_path))
            self.critic.load_state_dict(torch.load(critic_path))
            self.actor.eval()
            self.critic.eval()
            self.actor = self.actor.to(self.device)
            self.critic = self.critic.to(self.device)
            
        """Test the agent."""
        # Initialize the state by executing a simulation step with zero actions.
        simulation_output = self.env.simulation_step(np.zeros(self.a_dim, dtype=float))
        next_state = simulation_output['data'][0]  # the first item in data is next_state.

        with torch.no_grad():
         # the testing loop starts here...
            pbar = tqdm(range(1, 1001))
            for self.epochs in pbar:
                score = 0
                for self.iteration in range(1, self.horizon + 1):

                    # get and store the new action (s, a)
                    state = next_state

                    action = self.select_action(state, exploration_rate=0)

                    # Execute a simulation step with the selected action and unpack the results.
                    simulation_output = self.env.simulation_step(action)
                    next_state, reward, delay, loss_rate, is_ratio = simulation_output['data']

                    # Update the cumulative score.
                    score += np.power(self.gamma, self.iteration - 1) * reward

                    # Check if the epoch is done.
                    done = self.iteration == self.horizon

                    # Update the progress bar with the current status.
                    pbar.set_description(f"timestep: {self.iteration}/{self.horizon} is_ratio: {is_ratio}, delay: {delay:.4f}, lossrate:{loss_rate:.4f}, num_broken_links:{len(self.env.broken_links)}")

                    # Log various metrics.
                    self.collector.log_data({'delays': delay, 'lossrates': loss_rate, 'traffics': self.env.get_current_traffic_amount(), 'broken_links_number': len(self.env.broken_links), 'broken_links': self.env.broken_links})
                
                self.env.traffic_manager.episode_end_traffic()
                # Log the score for the current epoch.
                self.collector.log_data({'scores': score})
                print('epochs')
                print(self.epochs)
                if self.epochs % 100 == 0:
                    self.collector.plot_all_data()
                    self.collector.save_all_data()