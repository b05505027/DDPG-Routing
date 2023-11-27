import xml.etree.ElementTree as ET
import os, subprocess
import pandas as pd
import numpy as np
import networkx as nx 
from events import generate_traffic_events, generate_rf_events

configs = [
    {
        "project_path": "/home/jialun/routing",
        "simulation_path": "/home/jialun/samples/drl_routing/simulations",
        "ini":'omnetpp_1_tmp.ini',
        "ned":'package_1_tmp.ned',
        # 'traffic':'traffic_1.xml',
        'routing':'routing_1.xml',
    },
    {
        "project_path": "/home/jialun/routing",
        "simulation_path": "/home/jialun/samples/drl_routing/simulations2",
        "ini":'omnetpp_2_tmp.ini',
        "ned":'package_2_tmp.ned',
        'traffic':'traffic_2.xml',
        'routing':'routing_2.xml',

    },
    {
        "project_path": "/home/jialun/routing",
        "simulation_path": "/home/jialun/samples/drl_routing/simulations3",
        "ini":'omnetpp_3_tmp.ini',
        "ned":'package_3_tmp.ned',
        'traffic':'traffic_3.xml',
        'routing':'routing_3.xml',
    },
        {
        "project_path": "/home/jialun/routing",
        "simulation_path": "/home/jialun/samples/drl_routing/simulations4",
        "ini":'omnetpp_4_tmp.ini',
        "ned":'package_4_tmp.ned',
        'traffic':'traffic_4.xml',
        'routing':'routing_4.xml',
    },
        {
        "project_path": "/home/jialun/routing",
        "simulation_path": "/home/jialun/samples/drl_routing/simulations5",
        "ini":'omnetpp_5_tmp.ini',
        "ned":'package_5_tmp.ned',
        'traffic':'traffic_5.xml',
        'routing':'routing_5.xml',
    },
        {
        "project_path": "/home/jialun/routing",
        "simulation_path": "/home/jialun/samples/drl_routing/simulations6",
        "ini":'omnetpp_6_tmp.ini',
        "ned":'package_6_tmp.ned',
        'traffic':'traffic_6.xml',
        'routing':'routing_6.xml',
    },

    {
        "project_path": "/home/jialun/routing",
        "simulation_path": "/home/jialun/samples/drl_routing/simulations7",
        "ini":'omnetpp_7_tmp.ini',
        "ned":'package_7_tmp.ned',
        'traffic':'traffic_7.xml',
        'routing':'routing_7.xml',
    },
    
]


def binary_states(n_bits):
    if n_bits >= 1:
        states = binary_states(n_bits-1)*2
        
        for i in range(len(states)//2):
            states[i] += str(0)
            
        for i in range(len(states)//2, len(states)):
            states[i] += str(1)
            
    else:
        states = [""]
    return states
        

def get_state_distribution(n_links, max_broken_links, failure_rate, recovery_rate):
    states = binary_states(max_broken_links)
    transition_matrix = np.ones((len(states), len(states)), dtype=np.float32)
    for i in range(len(states)):
        for j in range(len(states)):
            for k in range(max_broken_links):
                if states[i][k] == states[j][k]:
                    if states[i][k] == "0":
                        transition_matrix[i][j] *= 1 - failure_rate
                    else:
                        transition_matrix[i][j] *= 1 - recovery_rate
                else:
                    if states[i][k] == "0":
                        transition_matrix[i][j] *= recovery_rate
                    else:
                        transition_matrix[i][j] *= failure_rate
    for i in range(len(states)):
        states[i] = states[i].zfill(n_links)[::-1]

    # transpose it to get the right order
    transition_matrix = transition_matrix
    eigenvals, eigenvects = np.linalg.eig(transition_matrix)
    
    # print('transition_matrix', transition_matrix)
    # print('eigenvals', eigenvals)
    # print('eigenvects', eigenvects)
    '''
    Find the indexes of the eigenvalues that are close to one.
    Use them to select the target eigen vectors. Flatten the result.
    '''
    close_to_1_idx = np.isclose(eigenvals,1)
    target_eigenvect = eigenvects[:,close_to_1_idx]
    target_eigenvect = target_eigenvect[:,0]
    # Turn the eigenvector elements into probabilites
    stationary_distrib = target_eigenvect / sum(target_eigenvect) 

    return states, stationary_distrib, transition_matrix.T



class Logger:
    def __init__(self, log_path):
        self.log_path = log_path
    
    def write(self, *message):
        message = ' '.join(map(str, message))
        with open(self.log_path, 'a') as f:
            f.write(message + '\n')

class Simulation:
    def __init__(self, num_nodes, total_traffic, time_limit, run_index, lam_f, lam_r, lam_f_test,alpha, is_test,max_broken_links=34):
        self.omnet_init()
        self.num_nodes = num_nodes
        self.total_traffic = total_traffic
        self.time_limit = time_limit
        self.broken_links = []
        self.run_index = run_index
        self.max_broken_links=max_broken_links
        self.is_test = is_test

        self.lam_t = 5

        if is_test:
            self.lam_f = lam_f_test
        else:
            self.lam_f = lam_f

        self.lam_f_test = lam_f_test
        self.lam_r = lam_r
        
        self.alpha = alpha
        

      

        self.large_links = [0,1,2,3,11,12,13,19,20,26]
        self.ports = [
            ((1,2), (2, 1)),
            ((1,3), (3, 1)),
            ((1,4), (4, 1)),
            ((1,5), (5, 1)),
            ((1,6), (6, 1)),
            ((1,7), (7, 1)),
            ((1,8), (8, 1)),
            ((1,9), (9, 1)),
            ((1,10), (10, 1)),
            ((1,11), (11, 1)),
            ((1,12), (12, 1)),
            ((2,3), (3, 2)),
            ((2,4), (4, 2)),
            ((2,5), (5, 2)),
            ((2,6), (6, 2)),
            ((2,7), (7, 2)),
            ((2,8), (8, 2)),
            ((2,9), (9, 2)),
            ((2,10), (10, 2)),
            ((3,4), (4, 3)),
            ((3,5), (5, 3)),
            ((3,13), (13, 3)),
            ((3,14), (14, 3)),
            ((3,15), (15, 3)),
            ((3,16), (16, 3)),
            ((3,17), (17, 3)),
            ((4,5), (5, 4)),
            ((4,11), (11, 4)),
            ((4,12), (12, 4)),
            ((4,13), (13, 4)),
            ((4,14), (14, 4)),
            ((4,15), (15, 4)),
            ((4,16), (16, 4)),
            ((4,17), (17, 4)),
        ]
        

        # generate traffic events
        self.traffic_events = generate_traffic_events(lam_t=self.lam_t, time_limit=self.time_limit)
        self.traffic_number = len(self.traffic_events)
        # generate rf events
        self.events = self.traffic_events
        for i in range(self.max_broken_links):
            self.events.extend(generate_rf_events(link_id=i, lam_f=self.lam_f, lam_r=self.lam_r, time_limit=self.time_limit))
        # sort the events based on the occurrence time
        self.events.sort(key=lambda x: x[0])
        # for e in self.events:
        #     print(e)
        # input()




        self.event_index = 0
        self.current_event = None
        self.next_event = self.get_event()


       
        

        # generate traffics
        self.traffics = self.generate_traffics(traffic_number=self.traffic_number)
        self.traffic_index = 0
        self.current_traffic = np.zeros((self.num_nodes, self.num_nodes), dtype=np.float32)
        

        self.ethernet = {
            (1,2): ("10.0.1.2", "eth2"),
            (1,3): ("10.0.2.3", "eth3"),
            (1,4): ("10.0.3.4", "eth4"),
            (1,5): ("10.0.4.5", "eth5"),
            (1,6): ("10.0.5.6", "eth6"),
            (1,7): ("10.0.6.7", "eth7"),
            (1,8): ("10.0.7.8", "eth8"),
            (1,9): ("10.0.8.9", "eth9"),
            (1,10): ("10.0.9.10", "eth10"),
            (1,11): ("10.0.10.11", "eth11"),
            (1,12): ("10.0.11.12", "eth12"),


            (2,1): ("10.0.1.1", "eth1"),
            (2,3): ("10.0.12.3", "eth3"),
            (2,4): ("10.0.13.4", "eth4"),
            (2,5): ("10.0.14.5", "eth5"),
            (2,6): ("10.0.15.6", "eth6"),
            (2,7): ("10.0.16.7", "eth7"),
            (2,8): ("10.0.17.8", "eth8"),
            (2,9): ("10.0.18.9", "eth9"),
            (2,10): ("10.0.19.10", "eth10"),


            (3,1): ("10.0.2.1", "eth1"),
            (3,2): ("10.0.12.2", "eth2"),
            (3,4): ("10.0.20.4", "eth4"),
            (3,5): ("10.0.21.5", "eth5"),
            (3,13): ("10.0.22.13", "eth13"),
            (3,14): ("10.0.23.14", "eth14"),
            (3,15): ("10.0.24.15", "eth15"),
            (3,16): ("10.0.25.16", "eth16"),
            (3,17): ("10.0.26.17", "eth17"),


            (4,1): ("10.0.3.1", "eth1"),
            (4,2): ("10.0.13.2", "eth2"),
            (4,3): ("10.0.20.3", "eth3"),
            (4,5): ("10.0.27.5", "eth5"),
            (4,11): ("10.0.28.11", "eth11"),
            (4,12): ("10.0.29.12", "eth12"),
            (4,13): ("10.0.30.13", "eth13"),
            (4,14): ("10.0.31.14", "eth14"),
            (4,15): ("10.0.32.15", "eth15"),
            (4,16): ("10.0.33.16", "eth16"),
            (4,17): ("10.0.34.17", "eth17"),


            (5,1): ("10.0.4.1", "eth1"),
            (5,2): ("10.0.14.2", "eth2"),
            (5,3): ("10.0.21.3", "eth3"),
            (5,4): ("10.0.27.4", "eth4"),


            (6,1): ("10.0.5.1", "eth1"),
            (6,2): ("10.0.15.2", "eth2"),


            (7,1): ("10.0.6.1", "eth1"),
            (7,2): ("10.0.16.2", "eth2"),


            (8,1): ("10.0.7.1", "eth1"),
            (8,2): ("10.0.17.2", "eth2"),


            (9,1): ("10.0.8.1", "eth1"),
            (9,2): ("10.0.18.2", "eth2"),


            (10,1): ("10.0.9.1", "eth1"),
            (10,2): ("10.0.19.2", "eth2"),


            (11,1): ("10.0.10.1", "eth1"),
            (11,4): ("10.0.28.4", "eth4"),


            (12,1): ("10.0.11.1", "eth1"),
            (12,4): ("10.0.29.4", "eth4"),


            (13,3): ("10.0.22.3", "eth3"),
            (13,4): ("10.0.30.4", "eth4"),


            (14,3): ("10.0.23.3", "eth3"),
            (14,4): ("10.0.31.4", "eth4"),


            (15,3): ("10.0.24.3", "eth3"),
            (15,4): ("10.0.32.4", "eth4"),


            (16,3): ("10.0.25.3", "eth3"),
            (16,4): ("10.0.33.4", "eth4"),


            (17,3): ("10.0.26.3", "eth3"),
            (17,4): ("10.0.34.4", "eth4"),

        }

        self.edges = [ # notice that here the index of edges starts from 1
            (1, 2, {"weight": 0, "traffic": 0, "index": "1_1"}), 
            (2, 1, {"weight": 0, "traffic": 0, "index": "1_2"}), 

            (1, 3, {"weight": 0, "traffic": 0, "index": "2_1"}), 
            (3, 1, {"weight": 0, "traffic": 0, "index": "2_2"}), 

            (1, 4, {"weight": 0, "traffic": 0, "index": "3_1"}), 
            (4, 1, {"weight": 0, "traffic": 0, "index": "3_2"}), 

            (1, 5, {"weight": 0, "traffic": 0, "index": "4_1"}), 
            (5, 1, {"weight": 0, "traffic": 0, "index": "4_2"}), 

            (1, 6, {"weight": 0, "traffic": 0, "index": "5_1"}), 
            (6, 1, {"weight": 0, "traffic": 0, "index": "5_2"}), 

            (1, 7, {"weight": 0, "traffic": 0, "index": "6_1"}), 
            (7, 1, {"weight": 0, "traffic": 0, "index": "6_2"}), 

            (1, 8, {"weight": 0, "traffic": 0, "index": "7_1"}), 
            (8, 1, {"weight": 0, "traffic": 0, "index": "7_2"}), 

            (1, 9, {"weight": 0, "traffic": 0, "index": "8_1"}), 
            (9, 1, {"weight": 0, "traffic": 0, "index": "8_2"}), 

            (1, 10, {"weight": 0, "traffic": 0, "index": "9_1"}), 
            (10, 1, {"weight": 0, "traffic": 0, "index": "9_2"}), 

            (1, 11, {"weight": 0, "traffic": 0, "index": "10_1"}), 
            (11, 1, {"weight": 0, "traffic": 0, "index": "10_2"}), 

            (1, 12, {"weight": 0, "traffic": 0, "index": "11_1"}), 
            (12, 1, {"weight": 0, "traffic": 0, "index": "11_2"}), 

            (2, 3, {"weight": 0, "traffic": 0, "index": "12_1"}), 
            (3, 2, {"weight": 0, "traffic": 0, "index": "12_2"}), 

            (2, 4, {"weight": 0, "traffic": 0, "index": "13_1"}), 
            (4, 2, {"weight": 0, "traffic": 0, "index": "13_2"}), 

            (2, 5, {"weight": 0, "traffic": 0, "index": "14_1"}), 
            (5, 2, {"weight": 0, "traffic": 0, "index": "14_2"}), 

            (2, 6, {"weight": 0, "traffic": 0, "index": "15_1"}), 
            (6, 2, {"weight": 0, "traffic": 0, "index": "15_2"}), 

            (2, 7, {"weight": 0, "traffic": 0, "index": "16_1"}), 
            (7, 2, {"weight": 0, "traffic": 0, "index": "16_2"}), 

            (2, 8, {"weight": 0, "traffic": 0, "index": "17_1"}), 
            (8, 2, {"weight": 0, "traffic": 0, "index": "17_2"}), 

            (2, 9, {"weight": 0, "traffic": 0, "index": "18_1"}), 
            (9, 2, {"weight": 0, "traffic": 0, "index": "18_2"}), 

            (2, 10, {"weight": 0, "traffic": 0, "index": "19_1"}), 
            (10, 2, {"weight": 0, "traffic": 0, "index": "19_2"}), 

            (3, 4, {"weight": 0, "traffic": 0, "index": "20_1"}), 
            (4, 3, {"weight": 0, "traffic": 0, "index": "20_2"}), 

            (3, 5, {"weight": 0, "traffic": 0, "index": "21_1"}), 
            (5, 3, {"weight": 0, "traffic": 0, "index": "21_2"}), 

            (3, 13, {"weight": 0, "traffic": 0, "index": "22_1"}), 
            (13, 3, {"weight": 0, "traffic": 0, "index": "22_2"}), 

            (3, 14, {"weight": 0, "traffic": 0, "index": "23_1"}), 
            (14, 3, {"weight": 0, "traffic": 0, "index": "23_2"}), 

            (3, 15, {"weight": 0, "traffic": 0, "index": "24_1"}), 
            (15, 3, {"weight": 0, "traffic": 0, "index": "24_2"}), 

            (3, 16, {"weight": 0, "traffic": 0, "index": "25_1"}), 
            (16, 3, {"weight": 0, "traffic": 0, "index": "25_2"}), 

            (3, 17, {"weight": 0, "traffic": 0, "index": "26_1"}), 
            (17, 3, {"weight": 0, "traffic": 0, "index": "26_2"}), 

            (4, 5, {"weight": 0, "traffic": 0, "index": "27_1"}), 
            (5, 4, {"weight": 0, "traffic": 0, "index": "27_2"}), 

            (4, 11, {"weight": 0, "traffic": 0, "index": "28_1"}), 
            (11, 4, {"weight": 0, "traffic": 0, "index": "28_2"}), 

            (4, 12, {"weight": 0, "traffic": 0, "index": "29_1"}), 
            (12, 4, {"weight": 0, "traffic": 0, "index": "29_2"}), 

            (4, 13, {"weight": 0, "traffic": 0, "index": "30_1"}), 
            (13, 4, {"weight": 0, "traffic": 0, "index": "30_2"}), 

            (4, 14, {"weight": 0, "traffic": 0, "index": "31_1"}), 
            (14, 4, {"weight": 0, "traffic": 0, "index": "31_2"}), 

            (4, 15, {"weight": 0, "traffic": 0, "index": "32_1"}), 
            (15, 4, {"weight": 0, "traffic": 0, "index": "32_2"}), 

            (4, 16, {"weight": 0, "traffic": 0, "index": "33_1"}), 
            (16, 4, {"weight": 0, "traffic": 0, "index": "33_2"}), 

            (4, 17, {"weight": 0, "traffic": 0, "index": "34_1"}), 
            (17, 4, {"weight": 0, "traffic": 0, "index": "34_2"}), 

        ]

        self.G = nx.DiGraph()
        for i in range(1, self.num_nodes + 1):
            self.G.add_node(i)
        self.G.add_edges_from(self.edges)
    
    def resume_simulation(self):
        self.event_index = 0
        self.current_event = None
        self.next_event = self.get_event()
        self.traffic_index = 0
        self.current_traffic = np.zeros((self.num_nodes, self.num_nodes), dtype=np.float32)
        self.broken_links = []
        return


    def get_event_size(self):
        return len(self.events)
    def step(self, action, next_event = False):
        if next_event:
            self.current_event = self.next_event
            self.next_event = self.get_event()
            if self.current_event == None: # terminate
                return None
            
            bn = len(self.broken_links)
            rate_t = 1/self.lam_t
            rate_r = 1/self.lam_r
            rate_f = 1/self.lam_f
            rate_f_test = 1/self.lam_f_test

            if self.current_event[1].startswith("traffic"):
                self.current_traffic += self.get_traffic()
                self.current_traffic = np.floor(self.current_traffic)
                # turn zeros into ones
                self.current_traffic = self.current_traffic + (self.current_traffic == 0)
                

                
                px = 1/rate_t / ((34 - bn)*rate_f_test + bn*rate_r + rate_t)
                qx = 1/rate_t / ((34 - bn)*rate_f + bn*rate_r + rate_t)
                self.is_ratio =  px / qx


            elif self.current_event[1].startswith("failure"):

                px = (34 - bn)*rate_f_test / ((34 - bn)*rate_f_test + bn*rate_r + rate_t)
                qx = (34 - bn)*rate_f / ((34 - bn)*rate_f + bn*rate_r + rate_t)

                self.broken_links.append(int(self.current_event[1].split('_')[1]))
                self.is_ratio = px / qx

            elif self.current_event[1].startswith("recovery"):

                px = bn*rate_r / ((34 - bn)*rate_f_test + bn*rate_r + rate_t)
                qx = bn*rate_r / ((34 - bn)*rate_f + bn*rate_r + rate_t)

                self.broken_links.remove(int(self.current_event[1].split('_')[1]))
                self.is_ratio = px / qx

            if self.next_event != None:
                duration = self.next_event[0] - self.current_event[0]
            else:
                duration = self.time_limit - self.current_event[0]
        


        weights = self.action_transform(action)
        self.apply_weights(weights)
        self.apply_traffic_and_duration(self.current_traffic, duration + 1.01)
        self.apply_broken_links()
        self.run_simulation()

        delay, lossrate, link_traffiics = self.analyze_qos()
        link_traffiics = link_traffiics.reshape(1, -1)
        
        reward = self.alpha*self.delay_reward(delay) + (1 - self.alpha)*self.lossrate_reward(lossrate)

        self.adjust_traffic() # set the remaining traffic matrix

        # print('delay', delay)
        # print('lossrate', lossrate)
        # print('broken_links', self.broken_links)
        # print('---------------')

        

        return link_traffiics, reward, delay, lossrate
    

    def update_links(self, is_test):
        broken_links = []
        probability = 1
        probability_test = 1

        if is_test:
            failure_rate = self.test_failure_rate
            recovery_rate = self.test_recovery_rate
        else:
            failure_rate = self.failure_rate
            recovery_rate = self.recovery_rate

        
        for i in range(self.max_broken_links):
            if i in self.broken_links: # currently broken
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

        self.broken_links = broken_links
        return probability, probability_test


    
    def generate_traffics(self, traffic_number=10):
        traffics = []
        
        traffic_scales = np.abs((np.sin(np.linspace(0, 100*np.pi, traffic_number + 1)))*(np.sin(10*np.linspace(0, 100*np.pi, traffic_number + 1))+1))+0.1

        
        for i in range(traffic_number + 1): 
            random_vector = np.random.exponential(10, size=(self.num_nodes))
            traffic_size = self.total_traffic * traffic_scales[i]
            traffic = random_vector * random_vector.reshape(-1, 1)
            traffic = traffic / np.sum(traffic) * traffic_size
            traffics.append(traffic)
        from matplotlib import pyplot as plt
        plt.plot(traffic_scales)
        plt.savefig('traffic_sizes.png')
        
        return traffics
    
    def get_traffic(self):
        traffic = self.traffics[self.traffic_index]
        self.traffic_index += 1
        return traffic
    def get_event(self):
        if self.event_index >= len(self.events):
            return None
        else:
            event = self.events[self.event_index]
            self.event_index += 1
            return event


    # lossrate reward functino
    def lossrate_reward(self, lossrate):
        return -np.tanh(lossrate - 0.1)

    # delay reward function
    def delay_reward(self, delay):
        return -np.tanh(delay - 0.3)

    def action_transform(self, action: np.ndarray) -> np.ndarray:
        action = action.reshape(-1)
        action = action + 0.001
        action = action.tolist()
        return action


    # initialize omnetpp.ini
    def omnet_init(self):
        print('Initializing omnetpp.ini...')
        # set path to run Fifo simulation in DOS command prompt
        path1 = "/home/jialun/omnetpp-6.0/bin" # bin directory
        #path2 = '/'.join(['..','..','..','tools',  g'win64', 'mingw64', 'bin']) # mingw64 directory
        os.environ['PATH'] = ':'.join([path1,os.environ['PATH']]) # add to Path environment variable



    # run simulation
    def run_simulation(self):
        # cmd = "../../../bin/opp_run.exe -r 0 -m -u Cmdenv -c traffic100 -n .;../src omnetpp.ini"                                                                
        # cmd = " ".join(["opp_run",  "-r" , "0", "-m", "-u", "Cmdenv", "-c", "traffic100", '-n', '.;..\\src;..\\..\\inet\\src;..\\..\\inet\\examples;..\\..\\inet\\tutorials;..\\..\\inet\\showcases'])
        os.chdir(configs[self.run_index]["simulation_path"])
        process = subprocess.run("pwd", shell=True, capture_output=True)
        #print('pwd:', process.stdout.decode('utf-8'))

        
        #cmd = " ".join(["opp_run", "-l" ,"../../inet4.5/src/inet", "-r" , "0", "-m", "-u", "Cmdenv", '-n', ":".join([".","../src","../../inet4.5/src","../../inet4.5/examples","../../inet4.5/tutorials","../../inet4.5/showcases"])])
        # os.system(cmd)

        cmd = " opp_run -r 0 --result-dir='results' -m -u Cmdenv -c General -n .:../../inet4.5/examples:../../inet4.5/showcases:../../inet4.5/src:../../inet4.5/tests/networks:../../inet4.5/tutorials --image-path=../../inet4.5/images -l ../../inet4.5/src/INET omnetpp.ini "

        trial = 5
        while trial > 0:
            process = subprocess.run(cmd, shell=True, capture_output=True)
            #print(f'output: {process.stdout.decode("utf-8")}')
            if process.returncode != 0:
                print(f'output: {process.stdout.decode("utf-8")}')
                print(f'output: {process.stderr}')
                print(f'return code = {process.returncode}')
                trial -= 1
            else:
                # print(f'output: {process.stdout.decode("utf-8")}')
                # print(f'output: {process.stderr}')
                # print(f'return code = {process.returncode}')
                break
        if trial == 0:
            print('Simulation failed')
            exit(1)
        os.chdir(configs[self.run_index]['project_path'])
        #print(process.stdout.decode('utf-8'))


    # analyze qos
    def analyze_qos(self):
        os.chdir(configs[self.run_index]["simulation_path"])
        # create vectors.csv
        

        message = ""
        # create vectors.csv
        cmd = " ".join(["opp_scavetool", "export", "-o", "vectors.csv", "./results/*.vec"])
        process = subprocess.run(cmd, shell=True, capture_output=True)
        # print(f'output: {process.stdout.decode("utf-8")}')
        # print(f'output: {process.stderr}')
        message = process.stdout.decode("utf-8")
        if "empty" in message:
            print('Simulation failed')
            print(message)
            exit(1)


        # create scalars.csv
        cmd = " ".join(["opp_scavetool", "export", "-o", "scalars.csv", "./results/*.sca"])
        process = subprocess.run(cmd, shell=True, capture_output=True)
        # print(f'output: {process.stdout.decode("utf-8")}')
        # print(f'output: {process.stderr}')

        # analyze vectors.csv
        df = pd.read_csv("vectors.csv", low_memory=False)[['attrname','module', 'name','vecvalue']]
        df = df.query("name=='endToEndDelay:vector' & vecvalue.notna()")
        data = df.to_dict('records')


        app_delays = []
        for d in data:
            delays = np.array(d['vecvalue'].split(' '), dtype=float) # end-to-end delays for each sent packets
            length = delays.size # number of packets sent
            mean = np.mean(delays) # end-to-end delay averaged over all packets
            module = d['module']
            source = module.split('.')[1][4:]
            destination = module.split('.')[2][-2:-1]

            # print('length', length)
            # print('source, destination', source, destination, 'mean', mean)
            if source == destination or mean == 0:
                continue
            else:
                app_delays.append(mean)
        # print('app_delays', app_delays)
        # print('len(app_delays)', len(app_delays))
        # print('app_delays', np.mean(app_delays))
        # input()
        # input()
        if len(app_delays) != 0:
            avg_delay = np.mean(app_delays)
        else:
            avg_delay = 0
    
        # analyze scalars.csv
        '''['run', 'type', 'module', 'name', 'attrname', 'attrvalue', 'value',
       'count', 'sumweights', 'mean', 'stddev', 'min', 'max', 'underflows',
       'overflows', 'binedges', 'binvalues']'''
        df = pd.read_csv("scalars.csv", low_memory=False)[['module', 'name','value']]

        # analyze the traffic on each link
        df_outgoing = df.query("name=='outgoingPackets:count' & value.notna()")
        data_outgoing = df_outgoing.to_dict('records')

        link_traffics = [0] * len(self.ports) * 2

        for data in data_outgoing:
            outgoing_packets = int(data['value'])
            module = data['module']
            source = int(module.split('.')[1][4:])
            port = int(module.split('.')[2][-2:-1])

            
            # print('source, port', source, port)
            # print('name', data['module'])
            # print('total_packets', outgoing_packets)
            for i in range(len(self.ports)):
                # print('self.ports[i]', self.ports[i])
                # print('(socure, port)', (source, port))
                if (source, port) in self.ports[i]:
                    index = self.ports[i].index((source, port))
                    #print('contribution to link', 2*i+1, outgoing_packets)
                    link_traffics[2*i + index] += outgoing_packets
        link_traffics = np.array(link_traffics)
        #link_traffics = link_traffics / np.sum(link_traffics)
        link_traffics = link_traffics / 100000

        # analyze packet-drop rate
        df_lost_drop = df.query("name=='droppedPacketsQueueOverflow:count' & value.notna()")
        df_lost_down = df.query("name=='packetDropInterfaceDown:count' & value.notna()")
        df_total = df.query("name=='incomingPackets:count' & value.notna()")
        
        data_total = df_total.to_dict('records')
        data_lost_drop = df_lost_drop.to_dict('records')
        data_lost_down = df_lost_down.to_dict('records')

        data = list(zip(data_total, data_lost_drop, data_lost_down))
        loss_rates = []
        for scalar in data:
            total_packets = int(scalar[0]['value'])
            lost_packets = int(scalar[1]['value']) + int(scalar[2]['value'])
            # print('name', scalar[0]['module'])
            # print('total_packets', total_packets)
            # print('drop', scalar[1]['value'])
            # print('down', scalar[2]['value'])
            if total_packets != 0:
                loss_rates.append(lost_packets / total_packets)
            else:
                continue
        
        assert len(loss_rates)!= 0 # must be at least one loss rate

        #print('loss_rates', np.mean(loss_rates))
        avg_lossrate = np.mean(loss_rates)


        os.chdir(configs[self.run_index]['project_path'])
        return avg_delay, avg_lossrate, link_traffics

    def adjust_traffic(self):
        os.chdir(configs[self.run_index]["simulation_path"])
        # analyze received traffic and calculate the remaining traffic matrix
        df = pd.read_csv("vectors.csv", low_memory=False)[['attrname','module', 'name','vecvalue']]
        df = df.query("name=='packetReceived:vector(packetBytes)' & vecvalue.notna()")
        data = df.to_dict('records')
        complete_traffic = np.zeros((self.num_nodes, self.num_nodes), dtype=np.float32)
        for d in data:
            bytes_received = np.array(d['vecvalue'].split(' '), dtype=np.float32) 
            bytes_sum = np.sum(bytes_received) # amount of bytes received
            module = d['module'] # Network.host4.app[4]
            
            source = module.split('.')[1][4:]
            destination = module.split('.')[2][4:-1]
            # print('module', module)
            # print(source, destination, bytes_sum/1000 - self.current_traffic[int(source)-1][int(destination)-1])
            # input()
            if int(destination) == 0: 
                continue
            complete_traffic[int(source)-1][int(destination)-1] += bytes_sum/1000 # in kB
        # print('current_traffic_sum:', np.sum(self.current_traffic))
        # print('comeplete_traffic_sum:', np.sum(complete_traffic))

        self.current_traffic = self.current_traffic - complete_traffic + np.ones((self.num_nodes, self.num_nodes), dtype=np.float32)
        self.current_traffic = np.floor(self.current_traffic)
        # print('remaining_traffic', self.current_traffic)
        os.chdir(configs[self.run_index]['project_path'])
        return


    # applying link weights
    def apply_weights(self, link_weights):
        # set weights
        for edge in self.G.edges.data():
            index = int(edge[2]['index'][:edge[2]['index'].find("_")]) - 1
            edge[2]['weight'] = link_weights[index]

        #print('now the edge data is', [x[2]['weight'] for x in self.G.edges.data()])
        #input()c

        # calculate the shortest paths
        shortest_paths = nx.shortest_path(self.G,  weight="weight")


        routing_string = ""

        # count the hops
        #hops = np.zeros([self.num_nodes+1, self.num_nodes+1])

        for source in range(self.num_nodes):
            current_hops = 0
            source = source + 1
            paths = shortest_paths[source] # the shortest paths from source to all other reachable nodes
            for destination in paths.keys():
                if source != destination:
                    #hops[source][destination] = len(paths[destination])
                    next_hop = paths[destination][1]
                    gateway, interface = self.ethernet[(source, next_hop)]
                    routing_string += f'	<route hosts="node{source}" destination="192.168.{destination}.0" netmask="255.255.255.0" gateway="{gateway}" interface="{interface}"/>\n'
        
        # print('maxhops',np.max(hops))


        routing_path = configs[self.run_index]["simulation_path"] + "/routing.xml"
        routing_template = open(configs[self.run_index]['routing'], 'r').read()
        routing_template = routing_template.replace("<SHORTEST_PATH_ROUTING_CONFIG>", routing_string)
        with open(routing_path, 'w') as f:
            f.write(routing_template)

    
    def apply_broken_links(self):
        ned_path = configs[self.run_index]["simulation_path"] + "/package.ned"
        ned_template = open(configs[self.run_index]['ned'], 'r').read() # ned file template


        # links = {
        #     "connected": " <--> Eth10M <--> ",
        #     "disconnected": " <--> Eth10M {  disabled = true; } <--> ",
        # }

        
        links1G = {
            "connected": ' <--> Eth1G { @display("ls=#CE995C,3,s"); } <--> ',
            "disconnected": " <--> Eth1G {  disabled = true; } <--> ",
        }

        links100M = {
            "connected": ' <--> Eth100M { @display("ls=#277A5B,2,s"); } <--> ',
            "disconnected": " <--> Eth100M {  disabled = true; } <--> ",
        }



        
        
        connection_strings = []

        # mapping self.ports to the string format like ("node1.ethg[0]", "node3.ethg[0]")
        port_strings = list(map(lambda x: (f"node{x[0][0]}.ethg[{x[0][1]}]", f"node{x[1][0]}.ethg[{x[1][1]}]"), self.ports))



        for i in range(34):
            if i in self.large_links:
                connection_strings.append("        " + port_strings[i][0] + links1G["connected"] + port_strings[i][1] + ";\n")
            else:
                connection_strings.append("        " + port_strings[i][0] + links100M["connected"] + port_strings[i][1] + ";\n")
        for j in self.broken_links:
            if j in self.large_links:
                connection_strings[j] = "        " + port_strings[j][0] + links1G["disconnected"] + port_strings[j][1] + ";\n"
            else:
                connection_strings[j] = "        " + port_strings[j][0] + links100M["disconnected"] + port_strings[j][1] + ";\n"

        connection_string = "".join(connection_strings)
        #print(connection_string)
        #input()
        ned_template = ned_template.replace("<CONNECTIONS>", connection_string)
        with open(ned_path, 'w') as f:
            f.write(ned_template)
        

    def apply_traffic_and_duration(self, traffic, duration):
        ini_path = configs[self.run_index]["simulation_path"] + "/omnetpp.ini"
        ini_template = open(configs[self.run_index]['ini'], 'r').read()
        traffic = traffic.reshape(self.num_nodes, self.num_nodes) # traffic  matrix

        traffic_string = ""
        for source in range(self.num_nodes):
            for destination in range(self.num_nodes):
                amount = str(int(traffic[source][destination]))
                traffic_string += f'*.host{source+1}.app[{destination+1}].sendBytes = {amount}kB\n'

        ini_template = ini_template.replace("<TRAFFIC_PATTERN>", traffic_string)
        ini_template = ini_template.replace("<TIME_LIMIT>", str(duration) + "s")

        with open(ini_path, 'w') as f:
            f.write(ini_template)
    