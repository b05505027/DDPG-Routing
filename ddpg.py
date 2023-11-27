import torch
import numpy as np
import random
import sys
from typing import Dict, Tuple, List
from utils import Simulation, Logger, get_state_distribution, configs
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

    def __init__(self, s_dim: int, a_dim: int, max_size: int, batch_size: int = 32):
        
        """Initializate."""
        self.s_buffer = np.zeros([max_size, s_dim], dtype=np.float32)
        self.next_s_buffer = np.zeros([max_size, s_dim], dtype=np.float32)
        self.a_buffer = np.zeros([max_size, a_dim], dtype=np.float32)
        self.r_buffer = np.zeros([max_size], dtype=np.float32)
        self.done_buffer = np.zeros([max_size], dtype=np.float32)
        self.is_buffer = np.zeros([max_size], dtype=np.float32)
        self.max_size, self.batch_size = max_size, batch_size
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
        idxs = np.random.choice(self.size, size=self.batch_size, replace=False)
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
        layer_size: list = [300, 200],
    ):
        """Initialize."""
        super(Actor, self).__init__()
        
        self.layer_size = layer_size

        for i in range(len(layer_size)):
            if i == 0:
                setattr(self, f'hidden{i}', nn.Linear(input_dim, layer_size[i]))
            else:
                setattr(self, f'hidden{i}', nn.Linear(layer_size[i-1], layer_size[i]))
        self.out = nn.Linear(layer_size[-1], out_dim)
        
        self.out.weight.data.uniform_(-init_w, init_w)
        self.out.bias.data.uniform_(-init_w, init_w)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        x = state
        for i in range(len(self.layer_size)):
            x = F.relu(getattr(self, f'hidden{i}')(x))
        action = torch.sigmoid(self.out(x))
        
        return action
    
    
class Critic(nn.Module):
    def __init__(
        self, 
        input_dim: int, 
        init_w: float = 3e-3,
        layer_size: list = [300, 200],
    ):
        """Initialize."""
        super(Critic, self).__init__()
        self.layer_size = layer_size
        for i in range(len(layer_size)):
            if i == 0:
                setattr(self, f'hidden{i}', nn.Linear(input_dim, layer_size[i]))
            else:
                setattr(self, f'hidden{i}', nn.Linear(layer_size[i-1], layer_size[i]))

        self.out = nn.Linear(layer_size[-1], 1)
        self.out.weight.data.uniform_(-init_w, init_w)
        self.out.bias.data.uniform_(-init_w, init_w)

    def forward(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        """Forward method implementation."""
        x = torch.cat((state, action), dim=-1)
        for i in range(len(self.layer_size)):
            x = F.relu(getattr(self, f'hidden{i}')(x))
        value = self.out(x)
        
        return value



class DDPGAgent:
    """DDPGAgent interacting with environment."""

    def __init__(
        self,
        s_dim: int = 102,
        a_dim: int = 68,
        buffers_size: int = 8192,
        batch_size: int = 64,
        gamma: float = 0.99,
        eps: float = 0.995,
        tau: float = 0.005,
        initial_random_steps: int = 64,
        total_traffic: int = 2000,
        time_limit: int = 2000000,
        horizon: int = 20,
        num_nodes: int = 17,
        actor_lr: float = 5e-3,
        critic_lr: float = 5e-3,
        session_name: str = "example",
        training_lam_f: float = 1000,
        testing_lam_f: float = 5000,
        lam_r: float = 100,
        importance_sampling: bool = True,
        record_uniform: bool = False,
        max_broken_links: int = 7,
        test_pretrained_actor: str = "",
        test_pretrained_critic: str = "",
        run_index: int = 1,
        layer_size: list = [300, 200],
        is_ospf: bool = False,
        alpha: float=0.1,
        is_test: bool=False,
        
    ):


        """Initialize."""
        self.env = Simulation(num_nodes=num_nodes, 
                            total_traffic=total_traffic, 
                            time_limit=time_limit, 
                            run_index=run_index, 
                            lam_f = training_lam_f, 
                            lam_r = lam_r,
                            lam_f_test = testing_lam_f,
                            max_broken_links=max_broken_links,
                            alpha=alpha,
                            is_test=is_test)

        self.a_dim = a_dim
        self.s_dim = s_dim
        self.buffer = ReplayBuffer(s_dim =s_dim, a_dim=a_dim, max_size=buffers_size, batch_size=batch_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.eps = eps
        self.tau = tau
        self.total_traffic = total_traffic
        self.num_nodes = num_nodes
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.session_name = session_name
        self.horizon = horizon
        self.layer_size = layer_size
        self.timestep = 0
        



       


        self.importance_sampling = importance_sampling
        self.record_uniform = record_uniform
        self.test_pretrained_actor = test_pretrained_actor
        self.test_pretrained_critic = test_pretrained_critic

        
        self.run_index = run_index
        # mode: train / test
        self.is_test = is_test
        # mode: test pretrained / test ospf
        self.is_ospf = is_ospf

        
        
        # device: cpu / gpu
        self.device = "cpu"
        # self.device = "mps"


        if not self.is_ospf:
            # networks
            self.actor = Actor(s_dim, a_dim, layer_size = layer_size).to(self.device)
            self.actor_target = Actor(s_dim, a_dim, layer_size = layer_size).to(self.device)
            self.actor_target.load_state_dict(self.actor.state_dict())
            
            self.critic = Critic(s_dim + a_dim, layer_size = layer_size).to(self.device)
            self.critic_target = Critic(s_dim + a_dim, layer_size = layer_size).to(self.device)
            self.critic_target.load_state_dict(self.critic.state_dict())

        if not self.test:
            # optimizer
            self.actor_optimizer = optim.AdamW(self.actor.parameters(), lr=self.actor_lr)
            self.critic_optimizer = optim.AdamW(self.critic.parameters(), lr=self.critic_lr)
        
        # transition to store in memory
        self.transition = list()
        # total steps count
        self.total_step = 0
        self.initial_random_steps = initial_random_steps
    
    
    def select_action(self, state: np.ndarray, exploration_rate: float, store_transition: float) -> np.ndarray:
        # take random actions in the beginning
        if self.total_step < self.initial_random_steps and not self.is_test:
            selected_action = np.random.uniform(0, 1, size=self.a_dim).reshape(1, -1)
        else:
            selected_action = self.actor(torch.FloatTensor(state).to(self.device)).detach().cpu().numpy()
        
        # adding noise
        if not self.is_test:
            # using the beta distribution as noise
            noise = exploration_rate*(np.random.beta(0.5, 0.5, size=self.a_dim)).reshape(1, -1)
            selected_action = np.clip(selected_action + noise, 0, 1.0)
        
        if store_transition:
            self.transition = [state.reshape(-1), selected_action.reshape(-1)] # (s, a,)
        return selected_action
    
    #@profile(stream=sys.stdout)
    def step(self, action: np.ndarray, next_event: bool, is_done: bool = False, store_transition: bool=True) -> Tuple[np.ndarray, float]:
        
        # during the timeslot, get the current reward and traffic load for each link,
        # which is used to calculate the next state
        qos  = self.env.step(action, next_event=next_event)
        if qos==None:
            return None
        link_traffics, reward, delay, lossrate = qos
        
        if self.importance_sampling:
            is_ratio = self.env.is_ratio
        else:
            is_ratio = 1.0

        if self.is_test:
            self.pbar.set_description(f"timestep: {self.timestep}/{self.horizon} delay: {delay:.4f}, lossrate:{lossrate:.4f}, num_broken_links:{len(self.env.broken_links)}  ")
        else:
            self.pbar.set_description(f"timestep: {self.timestep}/{self.horizon} is_ratio: {is_ratio}, delay: {delay:.4f}, lossrate:{lossrate:.4f}, num_broken_links:{len(self.env.broken_links)  }")

        # synthesize the new state
        failure_state = np.zeros(34, dtype=float).reshape(1,-1)
        for index in self.env.broken_links:
            failure_state[0][index] = 1
        next_state = np.concatenate((link_traffics, failure_state), axis=1)
        
        # store the transition in memory
        if store_transition:
            self.transition += [reward, next_state.reshape(-1), is_ratio, is_done] # (s, a, r, s', is_ratio, done)
            self.buffer.store(*self.transition)

        return next_state, reward, delay, lossrate




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
        curr_return = reward + self.gamma * is_ratio * next_value * masks
        # new implementation

        #curr_return = is_ratio * (reward + self.gamma * next_value * masks)
        

        
        
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
    

    def train(self, plotting_interval: int = 1):

        """Train the agent."""
        self.is_test = False
        self.logger = Logger("experiments/" + self.session_name + "/log.txt")
        
        # initialization
        actor_losses = []
        critic_losses = []
        scores = []
        exploration_rates = []
        link_failure_times = []
        traffics  = []
        delays = []
        lossrates = []

        # get the event size
        event_size = self.env.get_event_size()
        self.pbar = tqdm(range(1, event_size//self.horizon + 2))

        # initialize the state
        state = None
        next_state, _, _, _= self.step(np.zeros(self.a_dim, dtype=float), next_event=True, is_done=False, store_transition=False)

        # start interacting with the environment
        for self.total_step in self.pbar:
            
            # calculate current exploration rate
            exploration_rate = np.power(self.eps, self.total_step)
            exploration_rates.append(exploration_rate)
            
            # in the beginning of each episode, reset the score
            score = 0
            score_uniform = 0
            
            
            for self.timestep in range(1, self.horizon + 1):
                self.logger.write("==================== episode {} timestep {}/{} ====================".format(self.total_step, self.timestep,event_size))
                # check if it's the last step
                done = False if self.timestep <  self.horizon else True
                
                # get and store the new action (s, a)
                state = next_state
                action = self.select_action(state, exploration_rate, store_transition=True)

                # get and store the new state(r, s'), plus, update link conditions in the environment
                next_state, reward, delay, lossrate = self.step(action,next_event=True, is_done=done, store_transition=True)

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
                self.logger.write(f'current traffic:', self.env.current_traffic)

                # if the episode ends
                if done:
                    scores.append(score)
                
                delays.append(delay)
                lossrates.append(lossrate)

                traffics.append(self.env.current_traffic.sum())

            # if training is ready
            if (len(self.buffer) >= self.batch_size and 
                self.total_step > self.initial_random_steps):
                actor_loss, critic_loss = self.update_model()
                actor_losses.append(actor_loss)
                critic_losses.append(critic_loss)
            
            
            # dump values
            json.dump(scores, open(f"./experiments/{self.session_name}/scores.json", "w"))
            json.dump(delays,open(f"./experiments/{self.session_name}/delays.json", "w"))
            json.dump(lossrates,open(f"./experiments/{self.session_name}/lossrates.json", "w"))

                
            
            # plotting
            if self.total_step % plotting_interval == 0:
                self._plot(
                    frame_idx = self.total_step, 
                    scores = scores, 
                    actor_losses = actor_losses, 
                    critic_losses = critic_losses,
                    exploration_rates = exploration_rates,
                    traffics = traffics,
                )

            if self.total_step % 100 == 0:
                print(f'model_saved epoch={self.total_step}')
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
    

    def test(self):

        
        """ load the pretrained model """
        self.logger = Logger("experiments/" + self.session_name + "/log_test.txt")

        if not self.is_ospf:
            try:
                self.actor.load_state_dict(torch.load(self.test_pretrained_actor))
                self.critic.load_state_dict(torch.load(self.test_pretrained_critic))
            except Exception as e:
                print('Error occurs while loading pretrained models')
                print(e)
                exit(0)
            self.actor = self.actor.to(self.device)
            self.critic = self.critic.to(self.device)


        """Test the agent."""
        self.logger.write(f"Testing environment... total_traffic: {self.total_traffic}")
        self.total_step = 0
        
        # initialization
        scores = []
        rewards = []
        traffics  = []
        self.pbar = tqdm(range(1, 1001))

        # initialize the state
        state = None
        next_state, _, _, _, = self.step(np.ones(self.a_dim, dtype=float),next_event=True, is_done=False, store_transition=False)
        
        with torch.no_grad():
         # the testing loop starts here...
             for self.total_step in self.pbar:
                
                # in the begging of each episode, reset the score
                score = 0

                for self.timestep in range(1, self.horizon + 1):
                    self.logger.write("==================== episode {} self.timestep {}/{} ====================".format(self.total_step,self.timestep, self.horizon))
                    
                    # check if it's the last step
                    done = False if self.timestep < self.horizon else True

                    # get and store the new action (s, a)
                    state = next_state

                    # is it a test for ospf?
                    if self.is_ospf:
                        action = np.array([0.1,0.1, 0.1,0.1, 0.1,0.1, 0.1,0.1, 1,1, 1,1, 1,1, 1,1, 1,1, 1,1, 1,1, 
                        0.1,0.1, 0.1,0.1, 0.1,0.1, 1,1, 1,1, 1,1, 1,1, 1,1,
                        0.1,0.1, 0.1,0.1, 1,1, 1,1, 1,1, 1,1, 1,1,
                        0.1,0.1, 1,1, 1,1, 1,1, 1,1, 1,1, 1,1, 1,1]).reshape(1, -1)
                    else:
                        action = self.select_action(state, 0, store_transition=False)

                    # get and store the new state(r, s'), plus, update link conditions in the environment
                    next_state, reward, delay, lossrate = self.step(action,next_event=True, is_done=done, store_transition=False)

                    # update the score (G)
                    score  = reward + self.gamma * score

                    # record rewards and q values
                    rewards.append(reward)

 
                    # logging (s, a, r, s')
                    self.logger.write(f'current state {str(state.reshape(-1)):<20}') # (s, )
                    self.logger.write(f'current action, {str(action.reshape(-1)):<20}') # (a, )
                    self.logger.write(f'current reward, {str(reward):<20}') # (r, )
                    self.logger.write(f'next state, {str(next_state.reshape(-1)):<20}') # (s', )

                    # logging (other information)
                    self.logger.write('score', score) # return (G)
                    self.logger.write(f'broken links: {str(self.env.broken_links):<20}')
                    self.logger.write(f'current traffic:', self.env.current_traffic)

                    # if the episode ends
                    if done:
                        scores.append(score)

                    traffics.append(self.env.current_traffic.sum())
                

                # plotting
                if self.total_step % 20 ==1:
                    self._plot(
                        frame_idx = self.total_step, 
                        scores = scores, 
                        rewards = rewards,
                        traffics = traffics,
                )
        # the testing loop ends here

    def _plot(
        self, 
        frame_idx: int, 
        scores: List[float] = [],
        actor_losses: List[float] = [], 
        critic_losses: List[float] = [], 
        exploration_rates: List[float] = [],
        rewards: List[float] = [],
        traffics: List[float] = []
    ):
    
        # Define plot colors
        Blue='#2CBDFE'
        Green='#47DBCD'
        Pink='#F3A0F2'
        Purple='#9D2EC5'
        Violet='#661D98'
        Amber='#F5B14C'
        Grey='#BDBDBD'

        def create_subplot_params():
            if self.test:
                first_title = self.test_pretrained_actor
            else:
                first_title = "DDPG"
            subplot_params = [
                [411, f"frame {frame_idx}. score: {np.mean(scores[-10:])}", scores, Blue, first_title, 50],
                [412, "actor_loss", actor_losses, Pink, None, 20],
                [413, "critic_loss", critic_losses, Purple,None, 20],
                [414, f"traffic pattern", traffics, Grey, "traffic", 50],
            ]
            return subplot_params
            

        def subplot(loc: int, title: str, values: List[float], color: str, legend: str, smoothing: int):
            if len(values) == 0:
                return
            plt.subplot(loc)
            plt.title(title)
            plt.plot(np.convolve(values, np.ones(smoothing)/smoothing, mode='valid'), color=color, linewidth=0.5, label=legend)
            plt.plot(np.ones(len(values)) * np.mean(values), "--", color=color)
            if legend:
                plt.legend()
            

        def plot_subplots(subplot_params):
            plt.close('all')
            plt.figure(figsize=(12, 16))
            for parameters in subplot_params:
                subplot(*parameters)
            # additional plot for exploration rate
            if not self.is_test:
                plt.subplot(411)
                plt.twinx().plot(exploration_rates, "--", color='#F2BFC8')

        # Main plotting logic
        subplot_params = create_subplot_params()
        plot_subplots(subplot_params)

        # Save plot
        plot_filename = f"./experiments/{self.session_name}/plot{'_test' if self.is_test else ''}.png"
        plt.savefig(plot_filename)
        plt.clf()



def train_agent(training_f: int, 
    importance_sampling: bool=False, 
    name_prefix: str=""):



    session_name = f"f={training_f}"
    if importance_sampling:
        session_name += "_IS"
    if name_prefix:
        session_name = name_prefix + "_" + session_name

    # check if the running index is already used
    # by checking if the file routing_index.xml is being modified
    found_index = False

    for run_index in range(7):
        try:
            ned_path = configs[run_index]["simulation_path"] + "/package.ned"
            if time.time() - os.path.getmtime(ned_path) > 60:
                found_index = True
                break
        except FileNotFoundError:
            print(f"The file {file_path} does not exist.")
            exit(0)

    if not found_index:
        print("All running indices are used. Please try again later.")
        exit(0)
    else:
        print(f"Running index {run_index} is available.")
        print(f"Assigning running index {run_index} to this experiment.")

    config = {
        # Experiment
        "session_name": session_name,
        "run_index": run_index,

        # Environment
        "total_traffic": 3000,
        "training_lam_f": training_f,
        "testing_lam_f": 5000,
        "alpha": 0.1,

        # MDP
        "horizon": 20,
        "gamma": 0.8,
        "importance_sampling": importance_sampling,

        # Replay Buffer
        "buffers_size": 8192,
        "batch_size": 256,

        # Training
        "layer_size": [800, 600, 200],
        "initial_random_steps":64,
    }

    
    os.mkdir(f"./experiments/{session_name}")
    with open(f"./experiments/{session_name}/config.json", "w") as file:
        json.dump(config, file, indent=4)

    print(f'start_training:{session_name}')
    same_seed(2023)
    agent = DDPGAgent(**config)
    agent.train()



def test_agent(session_name: str=None,
    epoch: int=0,
    ospf: bool=False,
    name_prefix:str=""):

    # check if the running index is already used
    # by checking if the file routing_index.xml is being modified
    found_index = False

    for run_index in range(7):
        try:
            ned_path = configs[run_index]["simulation_path"] + "/package.ned"
            if time.time() - os.path.getmtime(ned_path) > 60:
                found_index = True
                break
        except FileNotFoundError:
            print(f"The file {file_path} does not exist.")
            exit(0)

    if not found_index:
        print("All running indices are used. Please try again later.")
        exit(0)
    else:
        print(f"Running index {run_index} is available.")
        print(f"Assigning running index {run_index} to this experiment.")

    if ospf:
        if name_prefix:
            test_session_name = f"{name_prefix}_test_ospf"
        else:
            test_session_name = f"test_ospf"

    else:
        if name_prefix:
            test_session_name = f"{name_prefix}_test_epoch={epoch}_{session_name}"
        else:
            test_session_name = f"test_epoch={epoch}_{session_name}"
    os.mkdir(f"./experiments/{test_session_name}")

    config = {
        # Experiment
        "session_name": test_session_name,
        "run_index": run_index,

        # Environment
        "total_traffic": 3000,
        "testing_lam_f": 5000,
        "alpha": 0.1,

        # MDP
        "horizon": 20,
        "gamma": 0.8,

        # testings
        "is_ospf": ospf,
        "is_test": True
    }

    if not ospf:
        expdir = 'experiments' + '/' + session_name
        config_dir = expdir + '/' + 'config.json'
        with open(config_dir, 'r') as file:
            layer_size = json.load(file)['layer_size']
        config['layer_size'] = layer_size
        config['test_pretrained_actor'] = f'{expdir}/actor_{epoch}.ckpt'
        config['test_pretrained_critic'] = f'{expdir}/critic_{epoch}.ckpt'

    with open(f"./experiments/{test_session_name}/config.json", "w") as file:
        json.dump(config, file, indent=4)
    

    print(f'start_testing:{test_session_name}')
    same_seed(4096)
    agent = DDPGAgent(**config)
    agent.test()




if __name__ == "__main__":

    #train_agent(training_f=1000, importance_sampling=False, name_prefix="Queue=50")
    #train_agent(training_f=1323, importance_sampling=False, name_prefix="new_settingq")
    #test_agent(session_name="f=1323", epoch=200, name_prefix='for_test')
    test_agent(ospf=True, name_prefix="Queue=50")
    

