import torch
import numpy as np
import random
from typing import Dict, Tuple, List
from utils import Simulation
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from matplotlib import pyplot as plt
from tqdm import tqdm
import names
import os
import json
import time
if torch.backends.cudnn.enabled:
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def same_seed(seed):
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  if torch.cuda.is_available():
      torch.cuda.manual_seed_all(seed)

class ReplayBuffer:
    """A simple numpy replay buffer."""

    def __init__(self, s_dim: int, a_dim: int, max_size: int, sample_size: int = 32):
        
        """Initializate."""
        self.s_buffer = np.zeros([max_size, s_dim], dtype=np.float32)
        self.next_s_buffer = np.zeros([max_size, s_dim], dtype=np.float32)
        self.a_buffer = np.zeros([max_size, a_dim], dtype=np.float32)
        self.r_buffer = np.zeros([max_size], dtype=np.float32)
        self.done_buffer = np.zeros([max_size], dtype=np.float32)
        self.max_size, self.sample_size = max_size, sample_size
        self.ptr, self.size, = 0, 0

    def store(
        self,
        s: np.ndarray,
        a: np.ndarray, 
        r: float, 
        next_s: np.ndarray, 
        done: bool,
    ):
        """Store the transition in buffer."""
        self.s_buffer[self.ptr] = s
        self.next_s_buffer[self.ptr] = next_s
        self.a_buffer[self.ptr] = a
        self.r_buffer[self.ptr] = r
        self.done_buffer[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self) -> Dict[str, np.ndarray]:
        """Randomly sample a batch of experiences from memory."""
        idxs = np.random.choice(self.size, size=self.sample_size, replace=False)
        return dict(s=self.s_buffer[idxs],
                    next_s=self.next_s_buffer[idxs],
                    a=self.a_buffer[idxs],
                    r=self.r_buffer[idxs],
                    done=self.done_buffer[idxs])

    def __len__(self) -> int:
        return self.size




class Actor(nn.Module):
    def __init__(
        self, 
        input_dim: int, 
        out_dim: int,
        init_w: float = 3e-3,
    ):
        """Initialize."""
        super(Actor, self).__init__()
        
        self.hidden1 = nn.Linear(input_dim, 128)
        self.hidden2 = nn.Linear(128, 32)
        self.out = nn.Linear(32, out_dim)
        
        self.out.weight.data.uniform_(-init_w, init_w)
        self.out.bias.data.uniform_(-init_w, init_w)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        x = F.relu(self.hidden1(state))
        x = F.relu(self.hidden2(x))
        action = self.out(x).tanh()
        
        return action
    
    
class Critic(nn.Module):
    def __init__(
        self, 
        input_dim: int, 
        init_w: float = 3e-3,
    ):
        """Initialize."""
        super(Critic, self).__init__()
        
        self.hidden1 = nn.Linear(input_dim, 128)
        self.hidden2 = nn.Linear(128, 128)
        self.out = nn.Linear(128, 1)
        
        self.out.weight.data.uniform_(-init_w, init_w)
        self.out.bias.data.uniform_(-init_w, init_w)

    def forward(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        """Forward method implementation."""
        x = torch.cat((state, action), dim=-1)
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        value = self.out(x)
        
        return value



class DDPGAgent:
    """DDPGAgent interacting with environment."""

    def __init__(
        self,
        s_dim: int,
        a_dim: int,
        buffers_size: int,
        sample_size: int,
        gamma: float = 0.99,
        eps: float = 0.95,
        tau: float = 0.005,
        initial_random_steps: int = 1e4,
        total_traffic: int = 1000,
        period: int = 1,
        num_nodes: int = 5,
        actor_lr: float = 3e-4,
        critic_lr: float = 1e-3,
        l2_reg: float = 1e-6,
        session_name: str = "example",
        failure_rate: float = 0.0,
        recovery_rate: float = 0.0,
        record_uniform: bool = False,
    ):
        """Initialize."""
        self.env = Simulation(num_nodes=5, total_traffic=total_traffic, period=period)
        self.a_dim = a_dim
        self.s_dim = s_dim
        self.buffer = ReplayBuffer(s_dim =s_dim, a_dim=a_dim, max_size=buffers_size, sample_size=sample_size)
        self.sample_size = sample_size
        self.gamma = gamma
        self.eps = eps
        self.tau = tau
        self.total_traffic = total_traffic
        self.num_nodes = num_nodes
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.session_name = session_name
        self.period = period
        self.failure_rate = failure_rate
        self.recovery_rate = recovery_rate
        self.record_uniform = record_uniform
        
        

        
        
        # device: cpu / gpu
        self.device = "mps"


        # networks
        self.actor = Actor(s_dim, a_dim).to(self.device)
        self.actor_target = Actor(s_dim, a_dim).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        
        self.critic = Critic(s_dim + a_dim).to(self.device)
        self.critic_target = Critic(s_dim + a_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # optimizer
        self.actor_optimizer = optim.AdamW(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optimizer = optim.AdamW(self.critic.parameters(), lr=self.critic_lr)
        
        # transition to store in memory
        self.transition = list()
        
        # total steps count
        self.total_step = 0

        # mode: train / test
        self.is_test = False

        self.initial_random_steps = initial_random_steps
    
    
    def select_action(self, state: np.ndarray, exploration_rate: float) -> np.ndarray:
        """Select an action from the input state."""
        # if initial random action should be conducted
        if self.total_step < self.initial_random_steps and not self.is_test:
            selected_action = np.random.uniform(-1, 1, size=self.a_dim).reshape(1, -1)
        else:
            selected_action = self.actor(
                torch.FloatTensor(state).to(self.device)
            ).detach().cpu().numpy()
        
        # adding noise
        if not self.is_test:
            # using beta distribution as noise
            noise = exploration_rate*np.pi*(np.random.beta(0.1, 0.1, size=self.a_dim) - 0.5).reshape(1, -1)
            #print('noise', noise)
            selected_action = np.clip(selected_action + noise, -1.0, 1.0)
            #print('selected_action', selected_action)
        
        self.transition = [state.reshape(-1), selected_action.reshape(-1)] # (s, a,)
        return selected_action
    
    def step(self, action: np.ndarray, require_uniform, next_traffic) -> Tuple[np.ndarray, np.float64, bool]:
        """Take an action and return the response of the env."""
        if require_uniform:
            _, reward_uniform, done  = self.env.step(np.ones(self.a_dim, dtype=float), next_traffic=False)
        else:
            reward_uniform = None

        next_state, reward, done  = self.env.step(action, next_traffic=next_traffic)


        if not self.is_test:
            self.transition += [reward, next_state.reshape(-1), done] # (s, a, r, s', done)
            self.buffer.store(*self.transition)

        return next_state, reward, done, reward_uniform

    def update_links(self):
        broken_links = []
        probability = 1
        for i in range(2):
            if i in self.env.broken_links: # currently broken
                if np.random.random() < self.recovery_rate: # will recover
                    probability *= self.recovery_rate
                    continue
                else: # still broken
                    broken_links.append(i)
                    probability *= (1 - self.recovery_rate)
            else: # currently good
                if np.random.random() < self.failure_rate: # will break
                    broken_links.append(i)
                    probability *= (self.failure_rate)
                else: # still good
                    probability *= (1 - self.failure_rate)
                    continue

        self.env.broken_links = broken_links

        return probability

        self.env.broken_links = broken_links




    def update_model(self) -> torch.Tensor:
        """Update the model by gradient descent."""
        device = self.device  # for shortening the following lines
        
        samples = self.buffer.sample_batch()

        state = torch.FloatTensor(samples["s"]).to(device)
        next_state = torch.FloatTensor(samples["next_s"]).to(device)
        action = torch.FloatTensor(samples["a"]).to(device)
        reward = torch.FloatTensor(samples["r"].reshape(-1, 1)).to(device)
        reward = (reward - reward.mean()) / (reward.std() + 1e-8)
        done = torch.FloatTensor(samples["done"].reshape(-1, 1)).to(device)
        
        masks = 1 - done
        next_action = self.actor_target(next_state)
        next_value = self.critic_target(next_state, next_action)
        curr_return = reward + self.gamma * next_value * masks
        

        
        
        # train critic
        values = self.critic(state, action)

        critic_loss = F.mse_loss(values, curr_return)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()

    
        # Clip the gradient norms for stable training.
        grad_norm = nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=10)
        self.critic_optimizer.step()
        
                
        # train actor
        actor_loss = -self.critic(state, self.actor(state)).mean()
        

        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=10)
        self.actor_optimizer.step()
        
        

        
        # target update
        self._target_soft_update()
        
        return actor_loss.data.cpu(), critic_loss.data.cpu()
    
    def train(self, max_steps: int, plotting_interval: int = 5):
        """Train the agent."""
        self.is_test = False
        
        state = self.env.get_current_state()


        actor_losses = []
        critic_losses = []
        scores = []
        scores_uniform = []
        exploration_rates = []
        link_failure_times = []
        
        for self.total_step in tqdm(range(1, max_steps + 1)):
            

            exploration_rate = np.power(self.eps, self.total_step)
            exploration_rates.append(exploration_rate)

            score = 0
            score_uniform = 0

            # if self.total_step == 500:
            #     self.env.broken_links = [0,1]

            for timestep in range(self.period):
                for ministep in range(3): # 3 mini steps in one episode
                    print("Current state: total_step, timestep, ministep: ", self.total_step,timestep, ministep)
                    prob = self.update_links()
                    print('state probablity', prob)
                    
                    action = self.select_action(state.reshape(1,-1), exploration_rate)
                    
                    if ministep == 2:
                        next_traffic = True
                    else:
                        next_traffic = False
                
                    if self.total_step <= 10000 and ministep == 0 and self.record_uniform:
                        next_state, reward, done, reward_uniform = self.step(action, require_uniform=True, next_traffic=False)
                        reward_uniform = reward_uniform * (1 - np.power(self.gamma,3))/(1-self.gamma)
                        score_uniform = reward_uniform + self.gamma * score_uniform
                    else:
                        next_state, reward, done, _ = self.step(action, require_uniform=False, next_traffic=next_traffic)
                    # else:
                    #     next_state, reward, done, _ = self.step(action, require_uniform=False)

                    state = next_state
                    score  = reward + self.gamma * score
                
                    #print('reward', reward)
                    #print('reward uniform', reward_uniform)
                    # print('score', score)
                    # print('score_uniform', score_uniform)
                    print('broken links', self.env.broken_links)
                    print('action', action.reshape(-1))
                    #print('done', done)
                    #print('length of scores_uniform', len(scores_uniform))

                    # if episode ends
                    if done:         
                        # state = self.env.get_current_state()
                        scores.append(score)

                        if self.total_step <= 10000 and self.record_uniform:
                            scores_uniform.append(score_uniform)

                        #print(scores)
                        #print(scores_uniform)

            # if training is ready
            if (
                len(self.buffer) >= self.sample_size 
                and self.total_step > self.initial_random_steps
            ):
                actor_loss, critic_loss = self.update_model()
                actor_losses.append(actor_loss)
                critic_losses.append(critic_loss)
            
            # plotting
            self._plot(
                self.total_step, 
                scores, 
                scores_uniform,
                actor_losses, 
                critic_losses,
                exploration_rates,
                30,
            )
                
        
    def test(self):
        """Test the agent."""
        self.is_test = True
        
        state = self.env.reset()
        done = False
        score = 0
        
        frames = []
        while not done:
            frames.append(self.env.render(mode="rgb_array"))
            action = self.select_action(state)
            next_state, reward, done = self.step(action)

            state = next_state
            score += reward
        
        #print("score: ", score)
        self.env.close()
        
        return frames
    
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
    
    def _plot(
        self, 
        frame_idx: int, 
        scores: List[float],
        scores_uniform: List[float], 
        actor_losses: List[float], 
        critic_losses: List[float], 
        exploration_rates: List[float],
        smoothing: int
    ):

        if smoothing < 20:
            smoothing = 10
        """Plot the training progresses."""
        def subplot(loc: int, title: str, values: List[float], color: str, legend: str):
            plt.subplot(loc)
            plt.title(title)
            plt.plot(np.convolve(values, np.ones(smoothing)/smoothing, mode='valid'), color=color, linewidth=0.5, label=legend)
            plt.plot(np.ones(len(values)) * np.mean(values), "--", color=color)
            plt.legend()

        CB91_Blue = '#2CBDFE'
        CB91_Green = '#47DBCD'
        CB91_Pink = '#F3A0F2'
        CB91_Purple = '#9D2EC5'
        CB91_Violet = '#661D98'
        CB91_Amber = '#F5B14C'
        CB91_Grey = '#BDBDBD'
        subplot_params = [
            (131, f"frame {frame_idx}. score: {np.mean(scores[-10:])}", scores, CB91_Blue, "DDPG"),
            (131, f"frame {frame_idx}. score: {np.mean(scores[-10:])}", scores_uniform, CB91_Green, "uniform"),
            (132, "actor_loss", actor_losses, CB91_Pink, None),
            (133, "critic_loss", critic_losses, CB91_Purple,None),
        ]
        json.dump(scores, open(f"./experiments/{self.session_name}/scores.json", "w"))
        plt.close('all')
        plt.figure(figsize=(30, 5))
        for loc, title, values, color, legend in subplot_params:
            if len(values) > 0:
             subplot(loc, title, values, color, legend)
        
        if len(scores_uniform) > 0:
            plt.subplot(131)
            # plt.plot(np.mean(scores_uniform)*np.ones(len(scores)), "--", color='cyan')
            plt.twinx().plot(exploration_rates, "--", color='#F2BFC8')
        plt.savefig(f"./experiments/{self.session_name}/plot.png")

if __name__ == "__main__":
    for actor_lr in [1e-3, 5e-4, 1e-4, 5e-5, 1e-5]:
        for critic_lr in [3e-3, 5e-4, 1e-4, 5e-5, 1e-5]:
            for period in [5, 10, 20, 30, 40, 50, 100, 150]:
                same_seed(2023)
                session_name = str(int(time.time()))[4:] + "_" + names.get_full_name()
                os.mkdir(f"./experiments/{session_name}")
                config = {
                    "s_dim": 7,
                    "a_dim": 7,
                    "buffers_size": 2048,
                    "sample_size": 64,
                    "gamma": 0.9999,
                    "eps": 0.989, #0.987
                    "initial_random_steps": 64 ,#64 + period * 20,
                    "total_traffic": 300,
                    "period": period,
                    "num_nodes": 5,
                    "actor_lr": actor_lr,
                    "critic_lr": critic_lr,
                    "session_name": session_name,
                    "failure_rate":0.1,
                    "recovery_rate":0.1,
                    "record_uniform": True,
                }
                s = json.dump(config, open(f"./experiments/{session_name}/config.json", "w"), indent=4)
                print(config)
                agent = DDPGAgent(**config)
                agent.train(max_steps=800)