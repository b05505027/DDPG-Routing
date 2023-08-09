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
        importance_sampling: bool = True,
        record_uniform: bool = False,
        max_broken_links: int = 0,
        test_pretrained_model: str = "",
        run_index: int = 0,
        
    ):


        """Initialize."""
        self.env = Simulation(num_nodes=5, 
                            total_traffic=total_traffic, 
                            period=period, 
                            run_index=run_index, 
                            failure_rate=failure_rate,
                            recovery_rate=recovery_rate,
                            test_failure_rate=test_failure_rate,
                            test_recovery_rate=test_recovery_rate,
                            max_broken_links=max_broken_links,)

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




       


        self.ministeps = ministeps
        self.importance_sampling = importance_sampling
        self.record_uniform = record_uniform
        self.test_pretrained_model = test_pretrained_model
        
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
    
    
    def select_action(self, state: np.ndarray, exploration_rate: float, store_transition: float) -> np.ndarray:
        # take random actions in the beginning
        if self.total_step < self.initial_random_steps and not self.is_test:
            selected_action = np.random.uniform(-1, 1, size=self.a_dim).reshape(1, -1)
        else:
            selected_action = self.actor(torch.FloatTensor(state).to(self.device)).detach().cpu().numpy()
        
        # adding noise
        if not self.is_test:
            # using the beta distribution as noise
            noise = exploration_rate*(np.random.beta(0.5, 0.5, size=self.a_dim) - 0.5).reshape(1, -1)
            selected_action = np.clip(selected_action + noise, -1.0, 1.0)
        
        if store_transition:
            self.transition = [state.reshape(-1), selected_action.reshape(-1)] # (s, a,)
        return selected_action
    
    #@profile(stream=sys.stdout)
    def step(self, action: np.ndarray, next_traffic: np.ndarray, is_done: bool = False, store_transition: bool=True) -> Tuple[np.ndarray, float]:
        
        # during the timeslot, get the current reward and traffic load for each link,
        # which is used to calculate the next state
        link_traffics, reward  = self.env.step(action, next_traffic=next_traffic)

        # update and observe link conditions, get the ratio for importance sampling
        prob, prob_test = self.env.update_links(self.is_test)
        if self.importance_sampling:
            is_ratio = prob_test / prob
        else:
            is_ratio = 1.0

        # synthesize the new state
        failure_state = np.zeros(7, dtype=float).reshape(1,-1)
        for index in self.env.broken_links:
            failure_state[0][index] = 1
        next_state = np.concatenate((link_traffics, failure_state), axis=1)
        
        # store the transition in memory
        if store_transition:
            self.transition += [reward, next_state.reshape(-1), is_ratio, is_done] # (s, a, r, s', is_ratio, done)
            self.buffer.store(*self.transition)

        return next_state, reward




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

        # old implementation
        # curr_return = reward + self.gamma * is_ratio * next_value * masks
        # new implementation

        curr_return = is_ratio * (reward + self.gamma * next_value * masks)
        

        
        
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
        self.logger = Logger("experiments/" + session_name + "/log.txt")

        # initialization
        actor_losses = []
        critic_losses = []
        scores = []
        scores_uniform = []
        exploration_rates = []
        link_failure_times = []
        
        # start interacting with the environment
        for self.total_step in tqdm(range(1, max_steps + 1)):
            
            # calculate current exploration rate
            exploration_rate = np.power(self.eps, self.total_step)
            exploration_rates.append(exploration_rate)
            
            # in the beginning of each episode, reset the score
            score = 0
            score_uniform = 0
                
            # before the episode starts, we generate a new state using OSPF
            state = None
            next_state, _ = self.step(np.ones(self.a_dim, dtype=float),next_traffic=True, is_done=False, store_transition=False)

            for timestep in range(1, self.period + 1):
                    self.logger.write("==================== episode {} timestep {} ====================".format(self.total_step, timestep))
                    # check if it's the last step
                    done = False if timestep < self.period else True
                    
                    # get and store the new action (s, a)
                    state = next_state
                    action = self.select_action(state, exploration_rate, store_transition=True)
                    # get and store the new state(r, s'), plus, update link conditions in the environment
                    next_state, reward = self.step(action,next_traffic=True, is_done=done, store_transition=True)

                    # update the score (G)
                    score  = reward + self.gamma * score
                
                    # logging (s, a, r, s')
                    self.logger.write(f'current state {str(state.reshape(-1)):<20}') # (s, )
                    self.logger.write(f'current action, {str(action.reshape(-1)):<20}') # (a, )
                    self.logger.write(f'current reward, {str(reward):<20}') # (r, )
                    self.logger.write(f'next state, {str(next_state.reshape(-1)):<20}') # (s', )

                    # logging (other information)
                    self.logger.write('score', score) # return (G)
                    self.logger.write(f'broken links: {str(self.env.broken_links):<20}')

                    # if the episode ends
                    if done:
                        scores.append(score)

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
                    frame_idx = self.total_step, 
                    scores = scores, 
                    scores_uniform = scores_uniform,
                    actor_losses = actor_losses, 
                    critic_losses = critic_losses,
                    exploration_rates = exploration_rates,
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
        self.logger = Logger("experiments/" + session_name + "/log_test.txt")
        is_random = False
        if self.test_pretrained_model == "random":
            is_random = True
        else:
            self.actor.load_state_dict(torch.load(self.test_pretrained_model))
            self.actor = self.actor.to(self.device)
            self.critic.load_state_dict(torch.load(self.test_pretrained_model.replace("actor", "critic")))
            self.critic = self.critic.to(self.device)

        """Test the agent."""
        self.logger.write(f"Testing environment... total_traffic: {self.total_traffic}, period: {self.period}")
        #self.env = Simulation(num_nodes=5, total_traffic=self.total_traffic, period=self.period, run_index=self.run_index)
        
        # self.env = Simulation(num_nodes=5, 
        #                     total_traffic=self.total_traffic, 
        #                     period=self.period, 
        #                     run_index=self.run_index, 
        #                     failure_rate=self.failure_rate,
        #                     recovery_rate=self.recovery_rate,
        #                     test_failure_rate=self.test_failure_rate,
        #                     test_recovery_rate=self.test_recovery_rate,
        #                     max_broken_links=self.max_broken_links)
        
        self.is_test = True
        self.total_step = 0
        
        # initialization
        scores = []
        rewards = []
        q_values = []
        
        with torch.no_grad():
         # the testing loop starts here...
            for self.total_step in tqdm(range(1, max_steps + 1)):
                
                # in the begging of each episode, reset the score
                score = 0

                # before the episode starts, we generate a new state using OSPF
                state = None
                next_state, _ = self.step(np.ones(self.a_dim, dtype=float),next_traffic=True, is_done=False, store_transition=False)

                for timestep in range(1, self.period + 1):
                    self.logger.write("==================== episode {} timestep {} ====================".format(self.total_step, timestep))

                    # check if it's the last step
                    done = False if timestep < self.period else True

                    # get (s, a)
                    state = next_state
                    if is_random:
                        action = np.random.uniform(-1, 1, size=(1, self.a_dim))
                    else:
                        action = self.select_action(state, 0, store_transition=False)

                    # Get (r, s'). Plus, update link conditions in the environment
                    next_state, reward = self.step(action, next_traffic=True, is_done=done, store_transition=False)

                    # record rewards and q values
                    rewards.append(reward)

                    # warning: if there's any shape error, check here first
                    q_values.append(self.critic(torch.FloatTensor(state).to(self.device), torch.FloatTensor(action).to(self.device)).cpu().numpy().item())         
                    
                    # update the score (G)
                    score  = reward + self.gamma * score
                    
 
                    # logging (s, a, r, s')
                    self.logger.write(f'current state {str(state.reshape(-1)):<20}') # (s, )
                    self.logger.write(f'current action, {str(action.reshape(-1)):<20}') # (a, )
                    self.logger.write(f'current reward, {str(reward):<20}') # (r, )
                    self.logger.write(f'next state, {str(next_state.reshape(-1)):<20}') # (s', )

                    # logging (other information)
                    self.logger.write('score', score) # return (G)
                    self.logger.write(f'broken links: {str(self.env.broken_links):<20}')

                    # if the episode ends
                    if done:
                        scores.append(score)
                

                # plotting
                if self.total_step % 20 ==1:
                    self._plot(
                        frame_idx = self.total_step, 
                        scores = scores, 
                        rewards = rewards,
                        q_values = q_values,
                    )
        # the testing loop ends here



    #@profile(stream=sys.stdout)
    def _plot(
        self, 
        frame_idx: int, 
        scores: List[float] = [],
        scores_uniform: List[float] = [], 
        actor_losses: List[float] = [], 
        critic_losses: List[float] = [], 
        exploration_rates: List[float] = [],
        rewards: List[float]= [],
        q_values: List[float]= [],
    ):
        """ Colors for plotting """
        CB91_Blue = '#2CBDFE'
        CB91_Green = '#47DBCD'
        CB91_Pink = '#F3A0F2'
        CB91_Purple = '#9D2EC5'
        CB91_Violet = '#661D98'
        CB91_Amber = '#F5B14C'
        CB91_Grey = '#BDBDBD'

        """Plot the training progresses."""
        def subplot(loc: int, title: str, values: List[float], color: str, legend: str, smoothing: int):
            if len(values) == 0:
                return
            plt.subplot(loc)
            plt.title(title)
            plt.plot(np.convolve(values, np.ones(smoothing)/smoothing, mode='valid'), color=color, linewidth=0.5, label=legend)
            plt.plot(np.ones(len(values)) * np.mean(values), "--", color=color)
            if legend:
                plt.legend()
     
        """ Preprocess q values """
        if q_values and rewards:
            q_values = (np.array(q_values)).tolist()
            ''' Group rewards every 10 items, 
                and change the ith value in the group into 
                the sum over the ith to 15th value in the group.'''
            true_q_values = []
            for i in range(0, len(rewards), 10):
                true_q_values.extend([sum(rewards[i+j:i+10]) for j in range(10)])
            """ check for validity """
            assert len(true_q_values) == len(q_values)
        else:
            true_q_values = []
            q_values = []


        """ dump raw values"""
        if self.is_test:
            json.dump(scores, open(f"./experiments/{self.session_name}/scores_test.json", "w"))
            json.dump(true_q_values, open(f"./experiments/{self.session_name}/true_q_values_test.json", "w"))
            json.dump(q_values, open(f"./experiments/{self.session_name}/q_values_test.json", "w"))
            json.dump(scores_uniform, open(f"./experiments/{self.session_name}/scores_uniform_test.json", "w"))
        else:
            json.dump(scores, open(f"./experiments/{self.session_name}/scores.json", "w"))
            json.dump(scores_uniform, open(f"./experiments/{self.session_name}/scores_uniform.json", "w"))

        if self.is_test:
            first_title = self.test_pretrained_model
        else:
            first_title = "DDPG"
        subplot_params = [
            [141, f"frame {frame_idx}. score: {np.mean(scores[-10:])}", scores, CB91_Blue, first_title, 20],
            [141, f"frame {frame_idx}. score: {np.mean(scores[-10:])}", scores_uniform, CB91_Green, "uniform", 20],
            [142, "actor_loss", actor_losses, CB91_Pink, None, 20],
            [143, "critic_loss", critic_losses, CB91_Purple,None, 20],
            [144, f"predicted and true Q values", true_q_values, CB91_Violet, "true_q_value", 20],
            [144, f"predicted and true Q values", q_values, CB91_Amber, "predicted_q_values", 20],
        ]
        plt.close('all')
        plt.figure(figsize=(30, 5))
        for parameters in subplot_params:
            subplot(*parameters)

        # additional plot for exploration rate
        if not self.is_test:
            plt.subplot(141)
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
            for period in [10, 20, 30, 40, 50, 100, 150]:
                checkpoint = ""
                for x in time.localtime()[:6]:
                    checkpoint += str(x) + "_"
                session_name = checkpoint + "_newis_" + name
                os.mkdir(f"./experiments/{session_name}")
                config = {
                    "s_dim": 14,
                    "a_dim": 7,
                    "buffers_size": 4096,
                    "sample_size": 64,
                    "gamma": 0.9999,
                    "eps": 0.9992, #0.995 -> 0.9992 six times longer
                    "initial_random_steps": 64 ,#64 -> 1000
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
                    "importance_sampling": False,
                    "record_uniform": False,
                    "max_broken_links": 4,
                    "test_pretrained_model":f"experiments/0001_01_nois/actor_5000.ckpt",
                    "run_index":2,
                }
                s = json.dump(config, open(f"./experiments/{session_name}/config.json", "w"), indent=4)
                print(config)

                # same_seed(2023)
                # agent = DDPGAgent(**config)
                # agent.train(max_steps=5000)
                same_seed(2024)
                agent = DDPGAgent(**config)
                agent.test(max_steps=1000)
                exit(0)

