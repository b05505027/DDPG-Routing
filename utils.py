import xml.etree.ElementTree as ET
import os, subprocess
import pandas as pd
import numpy as np
import networkx as nx 

configs = [
    {
        "project_path": "/Users/liangjialun/Desktop/routing",
        "simulation_path": "/Users/liangjialun/Downloads/samples/drl_routing/simulations",
        "ini":'omnetpp_1_tmp.ini',
        "ned":'package_1_tmp.ned',
        'traffic':'traffic_1.xml',
    },
    {
        "project_path": "/Users/liangjialun/Desktop/routing",
        "simulation_path": "/Users/liangjialun/Downloads/samples/drl_routing/simulations2",
        "ini":'omnetpp_2_tmp.ini',
        "ned":'package_2_tmp.ned',
        'traffic':'traffic_2.xml',

    },
    {
        "project_path": "/Users/liangjialun/Desktop/routing",
        "simulation_path": "/Users/liangjialun/Downloads/samples/drl_routing/simulations3",
        "ini":'omnetpp_3_tmp.ini',
        "ned":'package_3_tmp.ned',
        'traffic':'traffic_3.xml',

    }
]

class Logger:
    def __init__(self, log_path):
        self.log_path = log_path
    
    def write(self, *message):
        message = ' '.join(map(str, message))
        with open(self.log_path, 'a') as f:
            f.write(message + '\n')

class Simulation:
    def __init__(self, num_nodes, total_traffic, period, run_index):
        self.omnet_init()
        self.num_nodes = num_nodes
        self.total_traffic = total_traffic
        self.static_traffic = False
        self.period = period
        self.periodic_index = 0
        self.broken_links = []
        self.run_index = run_index

        

        # create a modeling graph
        self.edges = [
            (1, 3, {"weight": 0, "traffic": 0, "index": 1}),
            (1, 2, {"weight": 0, "traffic": 0, "index": 2}),
            (2, 4, {"weight": 0, "traffic": 0, "index": 3}),
            (3, 4, {"weight": 0, "traffic": 0, "index": 4}),
            (3, 5, {"weight": 0, "traffic": 0, "index": 5}),
            (4, 5, {"weight": 0, "traffic": 0, "index": 6}),
            (2, 3, {"weight": 0, "traffic": 0, "index": 7}),
        ]
        self.G = nx.Graph()
        for i in range(1, self.num_nodes):
            self.G.add_node(i)
        self.G.add_edges_from(self.edges)

        # generate traffics and states
        if self.period:
            self.periodic_traffics = self.generate_periodic_traffics()
            self.current_traffic = self.get_periodic_traffic()
            self.current_state = self.generate_state(None, self.current_traffic)
            self.new_traffic = self.get_periodic_traffic()
        else:
            # current new traffic
            self.current_traffic = self.generate_traffic()
            self.current_state = self.generate_state(None, self.current_traffic)
            # dummy new traffic
            self.new_traffic = self.generate_traffic(None, self.new_traffic)

    def quantize_traffic(self, traffic):
        traffic = traffic / self.total_traffic
        # if traffic <= 0.1:
        #     return 1
        # elif traffic <= 0.2:
        #     return 2
        # elif traffic <= 0.3:
        #     return 3
        # else:
        #     return 4
        return traffic
    def update_graph(self):
        self.G.remove_edges_from(list(self.G.edges()))
        self.G.add_edges_from(list(filter(lambda x: x[2]['index']-1 not in self.broken_links, self.edges)))
    '''
        Generate the specific state (traffics on each edges) 
        according to the weights (action) and the current traffic
    '''
    def generate_state(self, weights, traffic_matrix):
        # check if any link is broken
        self.update_graph()
        
        if weights is None:
            weights = np.ones(7)
        # reshape the traffic matrix to (num_nodes x num_nodes)
        traffic_matrix = traffic_matrix.reshape(self.num_nodes,-1)

        # initialize the graph
        self.rebuild_graph()

        # set weights
        for edge in self.G.edges.data():
            index = edge[2]['index'] - 1
            edge[2]['weight'] = weights[index]
    
        # calculate the shortest paths
        shortest_paths = nx.shortest_path(self.G,  weight="weight")


        # for item in shortest_paths.items():
        #     print(item)
        # input()


        for source in range(self.num_nodes):
            source = source + 1
            paths = shortest_paths[source] # the shortest paths from source to all other reachable nodes
            for destination in paths.keys():
                path = paths[destination]
                for i in range(len(path)-1):
                    self.G.edges[path[i], path[i+1]]['traffic'] += traffic_matrix[source-1, destination-1]
    
        traffic_state = np.zeros(7, dtype=float)
        failure_state = np.zeros(7, dtype=float)

        
        for edge in self.G.edges.data():
            index = edge[2]['index'] - 1
            traffic_state[index] = self.quantize_traffic(edge[2]['traffic'])
            self.quantize_traffic(edge[2]['traffic'])
        for index in self.broken_links:
            failure_state[index] = 1
        new_state = np.concatenate((traffic_state, failure_state), axis=0)
        new_state = new_state.reshape(1, -1)
        #print('new state', new_state)
        return new_state


    def rebuild_graph(self):
        nx.set_edge_attributes(self.G, 0, 'traffic')
        nx.set_edge_attributes(self.G, 0, 'weight')
    
    def step(self, action, next_traffic = False):
        weights = self.action_transform(action)

        #print('current traffic', self.current_traffic)
        self.apply_weights(weights)
        self.apply_traffic(self.current_traffic)
        self.apply_broken_links()
        self.run_simulation()


        #for i in range(100000):
        delay, lossrate = self.analyze_qos()

        #print('delay_reward:', self.delay_reward(delay))
        #print('lossrate_reward:', 0.2*self.lossrate_reward(lossrate))
        # input()
        reward = 1*self.delay_reward(delay) + 0.5*self.lossrate_reward(lossrate)

        # renew states
        if next_traffic:
            # if we choose to use the next traffic, weights are initialized to None
            # to generate the next traffic. Otherwise, we use the current weights.
            weights = None
            self.current_traffic = self.new_traffic
            self.new_traffic = self.generate_traffic()
            if self.periodic_index % self.period == 2:
                done = 1
            else:
                done = 0
        else:
            done = 0

        self.current_state = self.generate_state(weights, self.current_traffic)

        return self.current_state, reward, done 

    def get_current_state(self):
        return self.current_state


    def generate_traffic(self):
        if self.period:
            return self.get_periodic_traffic()
        else:
            random_vector = np.random.exponential(10, size=(self.num_nodes))
            traffic = random_vector * random_vector.reshape(-1, 1)
            traffic = traffic / np.sum(traffic) * self.total_traffic + 1
            traffic = traffic.reshape(1, -1)
            return traffic
    
    def generate_periodic_traffics(self):
        traffics = []
        for i in range(self.period):
            random_vector = np.random.exponential(10, size=(self.num_nodes))
            traffic = random_vector * random_vector.reshape(-1, 1)
            traffic = traffic / np.sum(traffic) * self.total_traffic + 1
            traffic = traffic.reshape(1, -1)
            traffics.append(traffic)
        return traffics
    
    def get_periodic_traffic(self):
        if self.periodic_index % self.period != 0:
            update_rate = 0.3 + np.random.rand() * 0.1
            traffic = update_rate * self.periodic_traffics[self.periodic_index % self.period] + (1-update_rate) * self.current_traffic
        else:
            traffic = self.periodic_traffics[0]
        self.periodic_index += 1
        return traffic



    
        
    # lossrate reward functino
    def lossrate_reward(self, lossrate):
        return -5*np.tanh(3*lossrate)

    # delay reward function
    def delay_reward(self, delay):
        return -delay

    def action_transform(self, action: np.ndarray) -> np.ndarray:
        action = action.reshape(-1)
        action = (action + 1) / 2 * 5 + 1
        action = action.astype(int)
        action = action.tolist()
        action = list(map(lambda x: x+(x==0), action))
        return action


    # initialize omnetpp.ini
    def omnet_init(self):
        print('Initializing omnetpp.ini...')
        print('os.environ[\'PATH\'] = ', os.environ['PATH'])
        # set path to run Fifo simulation in DOS command prompt
        path1 = "/Users/liangjialun/Downloads/omnetpp-6.0.1/bin" # bin directory
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

        cmd = " opp_run -r 0 --result-dir='results' -m -u Cmdenv -c General -n .:../src:../../inet4.5/examples:../../inet4.5/showcases:../../inet4.5/src:../../inet4.5/tests/validation:../../inet4.5/tests/networks:../../inet4.5/tutorials -x 'inet.common.selfdoc;inet.linklayer.configurator.gatescheduling.z3;inet.emulation;inet.showcases.visualizer.osg;inet.examples.emulation;inet.showcases.emulation;inet.transportlayer.tcp_lwip;inet.applications.voipstream;inet.visualizer.osg;inet.examples.voipstream' --image-path=../../inet4.5/images -l ../../inet4.5/src/INET omnetpp.ini "

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

        cmd = " ".join(["opp_scavetool", "export", "-o", "vectors.csv", "./results/*.vec"])
        process = subprocess.run(cmd, shell=True, capture_output=True)
        # print(f'output: {process.stdout.decode("utf-8")}')
        # print(f'output: {process.stderr}')
        message = process.stdout.decode("utf-8")
        if "empty" in message:
            input()


        # create scalars.csv
        cmd = " ".join(["opp_scavetool", "export", "-o", "scalars.csv", "./results/*.sca"])
        process = subprocess.run(cmd, shell=True, capture_output=True)
        # print(f'output: {process.stdout.decode("utf-8")}')
        # print(f'output: {process.stderr}')

        # analyze vectors.csv
        df = pd.read_csv("vectors.csv")[['attrname','module', 'name','vecvalue']]
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

            #print('source, destination', source, destination, 'mean', mean)
            if source == destination or mean == 0:
                continue
            else:
                app_delays.append(mean)
        #print('app_delays', np.mean(app_delays))
        # input()
        if len(app_delays) != 0:
            avg_delay = np.mean(app_delays)
        else:
            avg_delay = 0
    
        # analyze scalars.csv
        '''['run', 'type', 'module', 'name', 'attrname', 'attrvalue', 'value',
       'count', 'sumweights', 'mean', 'stddev', 'min', 'max', 'underflows',
       'overflows', 'binedges', 'binvalues']'''
        df = pd.read_csv("scalars.csv")[['module', 'name','value']]
        df_total = df.query("name=='incomingPackets:count' & value.notna()")
        df_lost_drop = df.query("name=='droppedPacketsQueueOverflow:count' & value.notna()")
        df_lost_down = df.query("name=='packetDropInterfaceDown:count' & value.notna()")
   
        
        data_total = df_total.to_dict('records')
        data_lost_drop = df_lost_drop.to_dict('records')
        data_lost_down = df_lost_down.to_dict('records')

        data = list(zip(data_total, data_lost_drop, data_lost_down))
        loss_rates = []
        for scalar in data:
            total_packets = int(scalar[0]['value'])


            lost_packets = int(scalar[1]['value']) + int(scalar[2]['value'])
            #print('name', scalar[0]['module'])
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
        return avg_delay, avg_lossrate


    # applying link weights
    def apply_weights(self, link_weights):
        # link wieghts is an array of link weights
        weight_path = configs[self.run_index]["simulation_path"] + "/weights.xml"
        tree = ET.parse(weight_path)
        root = tree.getroot()

        #print('link_weights', link_weights)

        for i in range(len(link_weights)):
            link_node = root.findall(f"./autoroute/link[@name='link{i+1}']")[0]
            link_node.set("cost", str(link_weights[i]))
        tree.write(weight_path)
    
    def apply_broken_links(self):
        ned_path = configs[self.run_index]["simulation_path"] + "/package.ned"
        ned_template = open(configs[self.run_index]['ned'], 'r').read()
        links = {
            "connected": " <--> Eth10M <--> ",
            "disconnected": " <--> Eth10M {  disabled = true; } <--> ",
        }

        ports = [
            ("node1.ethg[0]", "node3.ethg[0]"),
            ("node1.ethg[1]", "node2.ethg[0]"),
            ("node2.ethg[1]", "node4.ethg[0]"),
            ("node3.ethg[1]", "node4.ethg[1]"),
            ("node3.ethg[2]", "node5.ethg[0]"),
            ("node4.ethg[2]", "node5.ethg[1]"),
            ("node2.ethg[2]", "node3.ethg[3]"),
        ]
        
        connection_strings = []
        for i in range(7):
            connection_strings.append("        " + ports[i][0] + links["connected"] + ports[i][1] + ";\n")
        for j in self.broken_links:
            connection_strings[j] = "        " + ports[j][0] + links["disconnected"] + ports[j][1] + ";\n"

        connection_string = "".join(connection_strings)
        #print(connection_string)
        #input()
        ned_template = ned_template.replace("<CONNECTIONS>", connection_string)
        with open(ned_path, 'w') as f:
            f.write(ned_template)
        
        

    def apply_traffic(self, traffic):
        ini_path = configs[self.run_index]["simulation_path"] + "/omnetpp.ini"
        ini_template = open(configs[self.run_index]['ini'], 'r').read()
        traffic = traffic.reshape(self.num_nodes, self.num_nodes)
        traffic_path = configs[self.run_index]['traffic']
        tree = ET.parse(traffic_path)
        root = tree.getroot()
        traffic_string = ""

        for source in range(self.num_nodes):
            for destination in range(self.num_nodes):
                node = root.findall(f"./source[@node='{source+1}']/destination[@node='{destination+1}']/[@traffic]")[0]
                amount = str(int(traffic[source][destination]))
                node.set("traffic", amount)
                traffic_string += f'*.host{source+1}.app[{destination+1}].sendBytes = {amount}kB\n'
        tree.write(traffic_path)
        #print('traffic_string', traffic_string)
        ini_template = ini_template.replace("<TRAFFIC_PATTERN>", traffic_string)
        with open(ini_path, 'w') as f:
            f.write(ini_template)

# if __name__ == "__main__":
    
    
#     # sim = Simulation(num_nodes=5, total_traffic=1000, period=10)
#     # state = sim.generate_state()
#     # sim.set_traffic(state)
#     # sim.step(action=np.random.uniform(-1, 1, size=7).reshape(1, -1))

    