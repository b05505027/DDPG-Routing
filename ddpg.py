import torch
import numpy as np
import random
import sys
from typing import Dict, Tuple, List
from utils import Simulation, Logger, get_state_distribution
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib
from matplotlib import pyplot as plt
from tqdm import tqdm
import names
import os
import json
import time
from memory_profiler import profile
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
        self.is_buffer = np.zeros([max_size], dtype=np.float32)
        self.max_size, self.sample_size = max_size, sample_size
        self.ptr, self.size, = 0, 0

    def store(
        self,
        s: np.ndarray,
        a: np.ndarray, 
        r: float, 
        next_s: np.ndarray, 
        is_ratio: float,
        done: bool,
    ):
        """Store the transition in buffer."""
        self.s_buffer[self.ptr] = s
        self.next_s_buffer[self.ptr] = next_s
        self.a_buffer[self.ptr] = a
        self.r_buffer[self.ptr] = r
        self.done_buffer[self.ptr] = done
        self.is_buffer[self.ptr] = is_ratio
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self) -> Dict[str, np.ndarray]:
        """Randomly sample a batch of experiences from memory."""
        idxs = np.random.choice(self.size, size=self.sample_size, replace=False)
        return dict(s=self.s_buffer[idxs],
                    next_s=self.next_s_buffer[idxs],
                    a=self.a_buffer[idxs],
                    r=self.r_buffer[idxs],
                    is_ratio=self.is_buffer[idxs],
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
        
        self.hidden1 = nn.Linear(input_dim, 300)
        self.hidden2 = nn.Linear(300, 200)
        # self.hidden3 = nn.Linear(128, 64)
        # self.hidden4 = nn.Linear(64, 64)
        # self.hidden5 = nn.Linear(64, 64)
        self.out = nn.Linear(200, out_dim)
        
        self.out.weight.data.uniform_(-init_w, init_w)
        self.out.bias.data.uniform_(-init_w, init_w)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        x = F.relu(self.hidden1(state))
        x = F.relu(self.hidden2(x))
        # x = F.relu(self.hidden3(x))
        # x = F.relu(self.hidden4(x))
        # x = F.relu(self.hidden5(x))
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
        self.hidden3 = nn.Linear(128, 64)
        self.out = nn.Linear(64, 1)
        
        self.out.weight.data.uniform_(-init_w, init_w)
        self.out.bias.data.uniform_(-init_w, init_w)

    def forward(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        """Forward method implementation."""
        x = torch.cat((state, action), dim=-1)
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = F.relu(self.hidden3(x))
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
        test_failure_rate: float = 0.01,
        test_recovery_rate: float = 0.1,
        ministeps: int = 3,
        record_uniform: bool = False,
        max_broken_links: int = 0,
        test_pretrained_model: str = "",
        run_index: int = 0,
        
    ):


        """Initialize."""
        self.env = Simulation(num_nodes=5, total_traffic=total_traffic, period=period, run_index=run_index)
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
        self.test_failure_rate = test_failure_rate
        self.test_recovery_rate = test_recovery_rate
        self.ministeps = ministeps
        self.record_uniform = record_uniform
        self.max_broken_links = max_broken_links
        self.test_pretrained_model = test_pretrained_model
        
        self.logger = Logger("experiments/" + session_name + "/log.txt")
        self.run_index = run_index
        

        
        
        # device: cpu / gpu
        #self.device = "cpu"
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
            noise = exploration_rate*(np.random.beta(0.5, 0.5, size=self.a_dim) - 0.5).reshape(1, -1)
            #self.logger.write('noise', noise)
            selected_action = np.clip(selected_action + noise, -1.0, 1.0)
            #self.logger.write('selected_action', selected_action)
        
        self.transition = [state.reshape(-1), selected_action.reshape(-1)] # (s, a,)
        return selected_action
    
    #@profile(stream=sys.stdout)
    def step(self, action: np.ndarray, require_uniform, next_traffic, is_ratio = 1.0) -> Tuple[np.ndarray, np.float64, bool]:
        """Take an action and return the response of the env."""
        if require_uniform:
            _, reward_uniform, done  = self.env.step(np.ones(self.a_dim, dtype=float), next_traffic=False)
        else:
            reward_uniform = None

        next_state, reward, done  = self.env.step(action, next_traffic=next_traffic)


        if not self.is_test:
            self.transition += [reward, next_state.reshape(-1), is_ratio, done] # (s, a, r, s', is_ratio, done)
            self.buffer.store(*self.transition)

        return next_state, reward, done, reward_uniform

    def update_links(self):
        broken_links = []
        probability = 1
        probability_test = 1

        if self.is_test:
            failure_rate = self.test_failure_rate
            recovery_rate = self.test_recovery_rate
        else:
            failure_rate = self.failure_rate
            recovery_rate = self.recovery_rate

        self.logger.write('failure_rate', failure_rate)
        self.logger.write('recovery_rate', recovery_rate)
        
        for i in range(self.max_broken_links):
            if i in self.env.broken_links: # currently broken
                if np.random.random() < recovery_rate: # will recover
                    probability *= recovery_rate
                    probability_test *= self.test_recovery_rate
                    continue
                else: # still broken
                    broken_links.append(i)
                    probability *= (1 - recovery_rate)
                    probability_test *= (1 - self.test_recovery_rate)
            else: # currently good
                if np.random.random() < failure_rate: # will break
                    broken_links.append(i)
                    probability *= (failure_rate)
                    probability_test *= (self.test_failure_rate)
                else: # still good
                    probability *= (1 - failure_rate)
                    probability_test *= (1 - self.test_failure_rate)
                    continue

        self.env.broken_links = broken_links

        return probability, probability_test





    def update_model(self) -> torch.Tensor:
        """Update the model by gradient descent."""
        device = self.device  # for shortening the following lines
        
        samples = self.buffer.sample_batch()
        
        #self.logger.write('samples', samples)
        state = torch.FloatTensor(samples["s"]).to(device)
        next_state = torch.FloatTensor(samples["next_s"]).to(device)
        action = torch.FloatTensor(samples["a"]).to(device)
        reward = torch.FloatTensor(samples["r"].reshape(-1, 1)).to(device)
        #reward = (reward - reward.mean()) / (reward.std() + 1e-8)
        is_ratio = torch.FloatTensor(samples["is_ratio"].reshape(-1, 1)).to(device)
        done = torch.FloatTensor(samples["done"].reshape(-1, 1)).to(device)
        
        self.logger.write("############## sampling ##############")
        self.logger.write('state', state[:5])
        self.logger.write('next_state', next_state[:5])
        self.logger.write('action', action[:5])
        self.logger.write('reward', reward[:5])
        self.logger.write('is_ratio', is_ratio[:5])
        self.logger.write('##########################################')
        #self.logger.write('reward', reward)
        masks = 1 - done
        next_action = self.actor_target(next_state)
        next_value = self.critic_target(next_state, next_action)
        curr_return = reward + self.gamma * is_ratio * next_value * masks
        

        
        
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
    

    #@profile(stream=sys.stdout)
    def train(self, max_steps: int, plotting_interval: int = 30):
        """Train the agent."""
        self.is_test = False

        # get initial state distribution
        # state_list, train_stationary_distrib = get_state_distribution(
        #                                 n_links = self.a_dim,
        #                                 max_broken_links = self.max_broken_links,
        #                                 failure_rate = self.failure_rate,
        #                                 recovery_rate = self.recovery_rate)
        # state_list, test_stationary_distrib = get_state_distribution(
        #                         n_links = self.a_dim,
        #                         max_broken_links = self.max_broken_links,
        #                         failure_rate = self.test_failure_rate,
        #                         recovery_rate = self.test_recovery_rate)

        
        
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

            # get initial state probability
            # state_str = [0] * self.a_dim
            # for i in self.env.broken_links:
            #     state_str[i] = 1
            # state_str = ''.join(map(str, state_str))
            # train_init_prob = train_stationary_distrib[state_list.index(state_str)]
            # test_init_prob = test_stationary_distrib[state_list.index(state_str)]

            # print(f'state_str', state_str)
            # print(f'train_init_prob: {train_init_prob}')
            # print(f'test_init_prob: {test_init_prob}')
            # is_ratio = test_init_prob / train_init_prob
            # print(f'is_ratio: {is_ratio}')
 
            for timestep in range(self.period):
                for ministep in range(self.ministeps): # 3 mini steps in one episode
                    self.logger.write("====================total_step, timestep, ministep: ", self.total_step,timestep, ministep, "====================")
                    prob, prob_test = self.update_links()
           
                    is_ratio = prob_test / prob
                    
                    
                    self.logger.write('current state', state)
                    action = self.select_action(state.reshape(1,-1), exploration_rate)
                    
                    if ministep == self.ministeps - 1:
                        next_traffic = True
                    else:
                        next_traffic = False
                
                    if self.total_step <= 10000 and ministep == 0 and self.record_uniform:
                        next_state, reward, done, reward_uniform = self.step(action, require_uniform=True, next_traffic=False)
                        reward_uniform = reward_uniform * (1 - np.power(self.gamma,self.ministeps))/(1-self.gamma)
                        score_uniform = reward_uniform + self.gamma * score_uniform
                        score_uniform = score_uniform * is_ratio
                    else:
                        next_state, reward, done, _ = self.step(action, require_uniform=False, next_traffic=next_traffic, is_ratio = is_ratio)
                    # else:
                    #     next_state, reward, done, _ = self.step(action, require_uniform=False)

                    state = next_state
                    score  = reward + self.gamma * score
                
                    #self.logger.write('reward', reward)
                    #self.logger.write('reward uniform', reward_uniform)
                    self.logger.write('score', score)
                    # self.logger.write('score_uniform', score_uniform)
                    self.logger.write(f'broken links: {str(self.env.broken_links):<20}')
                    self.logger.write(f'action:{str(action.reshape(-1)):<20}')
                    #self.logger.write('done', done)
                    #self.logger.write('length of scores_uniform', len(scores_uniform))

                    # if episode ends
                    if done:         
                        # state = self.env.get_current_state()
                        scores.append(score)

                        if self.total_step <= 10000 and self.record_uniform:
                            scores_uniform.append(score_uniform)

                        #self.logger.write(scores)
                        #self.logger.write(scores_uniform)
                    #self.logger.write("============================================================")

            # if training is ready
            if (
                len(self.buffer) >= self.sample_size 
                and self.total_step > self.initial_random_steps
            ):
                actor_loss, critic_loss = self.update_model()
                actor_losses.append(actor_loss)
                critic_losses.append(critic_loss)
            
            # plotting
            if self.total_step % plotting_interval == 0:
                self._plot(
                    self.total_step, 
                    scores, 
                    scores_uniform,
                    actor_losses, 
                    critic_losses,
                    exploration_rates,
                    30,
                )

            if self.total_step % 100 == 0:
                torch.save(self.actor.state_dict(), f"./experiments/{self.session_name}/actor_{self.total_step}.ckpt")
                torch.save(self.critic.state_dict(), f"./experiments/{self.session_name}/critic_{self.total_step}.ckpt")
                


    
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
    

    def test(self, max_steps: int):
        """ load the pretrained model """
        is_random = False
        if self.test_pretrained_model == "random":
            is_random = True
        else:
            self.actor.load_state_dict(torch.load(self.test_pretrained_model))
        

        """Test the agent."""
        self.logger.write(f"Testing environment... total_traffic: {self.total_traffic}, period: {self.period}")
        self.env = Simulation(num_nodes=5, total_traffic=self.total_traffic, period=self.period, run_index=self.run_index)
        self.is_test = True
        self.total_step = 0
        
        


        state = self.env.get_current_state()
        scores = []
        

        # testing loop .........#
        for self.total_step in tqdm(range(1, max_steps + 1)):
            score = 0
            for timestep in range(self.period):
                for ministep in range(3): # 3 mini steps in one episode
                    self.logger.write("Current state: total_step, timestep, ministep: ", self.total_step,timestep, ministep)
                    _ = self.update_links()

                    if is_random:
                        action = np.random.uniform(-1, 1, size=self.a_dim).reshape(1, -1)
                    else:
                        action = self.select_action(state.reshape(1,-1), 0)
                    
                    if ministep == 2:
                        next_traffic = True
                    else:
                        next_traffic = False

                    next_state, reward, done, _ = self.step(action, require_uniform=False, next_traffic=next_traffic)
                    state = next_state
                    score  = reward + self.gamma * score
                
                    self.logger.write('reward', reward)
                    self.logger.write('score', score)
                    self.logger.write('broken links', self.env.broken_links)
                    self.logger.write('action', action.reshape(-1))

                    if done:         
                        scores.append(score)
            

            # plotting

            if self.total_step % 50 ==1:
                self._plot(
                    self.total_step, 
                    scores, 
                    [],
                    [], 
                    [],
                    [],
                    30,
                )
        # testing loop ends .........#




    #@profile(stream=sys.stdout)
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
            if legend:
                plt.legend()

        CB91_Blue = '#2CBDFE'
        CB91_Green = '#47DBCD'
        CB91_Pink = '#F3A0F2'
        CB91_Purple = '#9D2EC5'
        CB91_Violet = '#661D98'
        CB91_Amber = '#F5B14C'
        CB91_Grey = '#BDBDBD'

        if self.is_test:
            first_title = self.test_pretrained_model
        else:
            first_title = "DDPG"
        subplot_params = [
            (131, f"frame {frame_idx}. score: {np.mean(scores[-10:])}", scores, CB91_Blue, first_title),
            (131, f"frame {frame_idx}. score: {np.mean(scores[-10:])}", scores_uniform, CB91_Green, "uniform"),
            (132, "actor_loss", actor_losses, CB91_Pink, None),
            (133, "critic_loss", critic_losses, CB91_Purple,None),
        ]

        if self.is_test:
            json.dump(scores, open(f"./experiments/{self.session_name}/scores_test.json", "w"))
        else:
            json.dump(scores, open(f"./experiments/{self.session_name}/scores.json", "w"))
        if len(scores_uniform) > 0:
            if self.is_test:
                json.dump(scores_uniform, open(f"./experiments/{self.session_name}/scores_uniform_test.json", "w"))
            else:
                json.dump(scores_uniform, open(f"./experiments/{self.session_name}/scores_uniform.json", "w"))
        plt.close('all')
        plt.figure(figsize=(30, 5))
        for loc, title, values, color, legend in subplot_params:
            if len(values) > 0:
             subplot(loc, title, values, color, legend)
        
        if not self.is_test:
            plt.subplot(131)
            plt.twinx().plot(exploration_rates, "--", color='#F2BFC8')
        if self.is_test:
            plt.savefig(f"./experiments/{self.session_name}/plot_test.png")
        else:
            plt.savefig(f"./experiments/{self.session_name}/plot.png")
        plt.clf() 
if __name__ == "__main__":
    
    name = names.get_full_name()
    for actor_lr in [1e-3, 5e-4, 1e-4, 5e-5, 1e-5]:
        for critic_lr in [3e-3, 5e-4, 1e-4, 5e-5, 1e-5]:
            for period in [5, 10, 20, 30, 40, 50, 100, 150]:
                checkpoint = ""
                for x in time.localtime()[:6]:
                    checkpoint += str(x) + "_"
                same_seed(2024)
                session_name = checkpoint + "_" + name
                os.mkdir(f"./experiments/{session_name}")
                config = {
                    "s_dim": 14,
                    "a_dim": 7,
                    "buffers_size": 4096,
                    "sample_size": 64,
                    "gamma": 0.9999,
                    "eps": 0.995, #0.987
                    "initial_random_steps": 64 ,#64 + period * 20,
                    "total_traffic": 300,
                    "period": period,
                    "num_nodes": 5,
                    "actor_lr": actor_lr,
                    "critic_lr": critic_lr,
                    "session_name": session_name,
                    "failure_rate":0.01, #0.01,
                    "recovery_rate":0.1, #0.1,
                    "test_failure_rate":0.001,
                    "test_recovery_rate":0.1,
                    "ministeps": 1, #3,
                    "record_uniform": False,
                    "max_broken_links": 4,
                    "test_pretrained_model":"",
                    "run_index":2,
                }
                s = json.dump(config, open(f"./experiments/{session_name}/config.json", "w"), indent=4)
                print(config)
                agent = DDPGAgent(**config)
                agent.train(max_steps=10000)
                exit(0)

                # problems of generating same actions for all states
                # 1-layer
                # [1, 1, -1, 1, -1, -1, -1]
                # [1, -1, 1, 1, -1, -1, -1]

                # safe states best learned policy 1-layer
                # [-1 -1  1 1 1 -1 1]

                # safe states best learned policy 2-layer

