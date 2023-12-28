import numpy as np
import networkx as nx
import os
import pandas as pd
import subprocess
import json
from events import EventManager
from traffics import TrafficManager
from events import TrafficEvent, RecoveryEvent, FailureEvent

class Environment:
    def __init__(self, 
                total_traffic, 
                run_index,
                lam_f, 
                lam_f_test, 
                alpha,
                max_broken_links,
                queue_capacity,
                topology,
                time_limit = 1000000):
        
        # Add omnet++ environment variables
        self._omnet_init()

        # simulation config 
        self.run_index = run_index
        self.total_traffic = total_traffic
        self.time_limit = time_limit
        self.topology = topology.lower()
        self.alpha = alpha
        self.lam_f = lam_f
        self.lam_f_test = lam_f_test
        self.max_broken_links = max_broken_links
        self.queue_capacity = queue_capacity

        # build the network topology and keep track of the link status
        self.broken_links = []
        self._initialize_network_topology()


        # initialize event, traffic managers and QoS analyzer
        self.event_manager = EventManager(self.lam_t, self.max_broken_links, self.lam_f, self.lam_r, self.time_limit)
        self.traffic_manager = TrafficManager(self.num_nodes, self.total_traffic)
        self.qos_analyzer = QosAnalzer(self.run_index, self.oment_simulation_paths[self.run_index], self.ports)


    def _initialize_network_topology(self):
        # setting up basic topology configurations
        # including:
        # num_nodes, num_links, lam_t, lam_f_test
        # lam_r, edges and ethernet

        self.config = json.load(open(os.path.join('topology', self.topology, 'config_files', 'config.json')))
        for key, value in self.config.items():
            setattr(self, key, value)
        self.rate_t = 1/self.lam_t
        self.rate_r = 1/self.lam_r
        self.rate_f = 1/self.lam_f
        self.rate_f_test = 1/self.lam_f_test
        self.edges = json.load(open(self.edges, 'r'))
        self.ethernet = json.load(open(self.ethernet, 'r'))
        self.ports = json.load(open(self.ports, 'r'))
        self.large_links = json.load(open(self.large_links, 'r'))
        
        # topology Graph Creation
        self.G = nx.DiGraph()
        for i in range(1, self.num_nodes + 1):
            self.G.add_node(i)
        self.G.add_edges_from(self.edges)

    def _omnet_init(self):
        print('Initializing omnetpp.ini...')
        path1 = "/home/jialun/omnetpp-6.0/bin" # bin directory
        os.environ['PATH'] = ':'.join([path1,os.environ['PATH']]) # add to Path environment variable


    def simulation_step(self, action):
        """
        Proceeds with the simulation based on the given action and the next event.
        Returns a dictionary with the simulation status and related data.
        """
        current_event = self.event_manager.get_next_event()

        # Check if there are no more events
        if not current_event:  # Terminate if no event is left
            return {'status': 'terminated', 'data': None}

        # Calculate importance sampling ratio
        is_ratio = self._calculate_is_ratio(current_event)

        # Handle different types of events
        self._handle_event(current_event)

        # Setup and run the simulation
        self._setup_simulation(action, current_event) 
        self._run_simulation()

        # Analyze QoS and calculate the reward
        delay, loss_rate, link_traffics = self.qos_analyzer.analyze_qos()
        reward = self._calculate_reward(delay, loss_rate)

        # Update the traffic matrix based on consumed traffic
        consumed_traffic = self._calculate_consumed_traffic()
        self.traffic_manager.consume_traffic(consumed_traffic)

        # synthesize the new state
        failure_states = np.zeros(self.num_links, dtype=float).reshape(1,-1)
        for index in self.broken_links:
            failure_states[0][index] = 1
        next_state = np.concatenate((link_traffics, failure_states), axis=1)
        

        return {'status': 'running', 'data': [next_state, reward, delay, loss_rate, is_ratio]}

    def _handle_event(self, event):
        """
        Handles the given event based on its type.
        """
        if isinstance(event, TrafficEvent):  # Handle traffic events
            self.traffic_manager.add_upcoming_traffic()
        else:  # Handle other events, like broken links
            self._handle_broken_links(event)

    def _calculate_reward(self, delay, loss_rate):
        """
        Calculates the reward based on delay and loss rate.
        """
        return self.alpha * self.delay_reward(delay) + (1 - self.alpha) * self.lossrate_reward(loss_rate)       # sys1.0
    
    def _setup_simulation(self, action, event):
        self._configure_weights(self.action_transform(action))
        self._configure_traffic_settings(self.traffic_manager.get_current_traffic_matrix(), event.duration + 1.01)
        self._configure_link_status()


    def _calculate_is_ratio(self, current_event):
        """
        Calculate the importance sampling ratio based on the type of the current event.

        :param current_event: The current event being processed.
        :return: The calculated importance sampling ratio.
        """
        bn = len(self.broken_links)
        if isinstance(current_event, TrafficEvent):
            px = self.rate_t / ((self.max_broken_links- bn) * self.rate_f_test + bn * self.rate_r + self.rate_t)
            qx = self.rate_t / ((self.max_broken_links- bn) * self.rate_f + bn * self.rate_r + self.rate_t)
            return px / qx

        if isinstance(current_event, FailureEvent):
            px = (self.max_broken_links- bn) * self.rate_f_test / ((self.max_broken_links- bn) * self.rate_f_test + bn * self.rate_r + self.rate_t)
            qx = (self.max_broken_links- bn) * self.rate_f / ((self.max_broken_links- bn) * self.rate_f + bn * self.rate_r + self.rate_t)
            return px / qx

        if isinstance(current_event, RecoveryEvent):
            px = bn * self.rate_r / ((self.max_broken_links- bn) * self.rate_f_test + bn * self.rate_r + self.rate_t)
            qx = bn * self.rate_r / ((self.max_broken_links- bn) * self.rate_f + bn * self.rate_r + self.rate_t)
            return px / qx

    def _handle_broken_links(self, current_event):
        """
        Update the list of broken links based on the current event.

        :param current_event: The current event being processed.
        """
        if isinstance(current_event, FailureEvent):
            self.broken_links.append(current_event.link_id)
        elif isinstance(current_event, RecoveryEvent):       
            self.broken_links.remove(current_event.link_id)
    
    def get_event_size(self):
        return self.event_manager.get_event_size()


    

    def _run_simulation(self):
        origin_project_path = os.getcwd()
        os.chdir(self.oment_simulation_paths[self.run_index])
        cmd = " opp_run -r 0 --result-dir='results' -m -u Cmdenv -c General -n .:../../inet4.5/examples:../../inet4.5/showcases:../../inet4.5/src:../../inet4.5/tests/networks:../../inet4.5/tutorials --image-path=../../inet4.5/images -l ../../inet4.5/src/INET omnetpp.ini "

        trial = 5
        while trial > 0:
            process = subprocess.run(cmd, shell=True, capture_output=True)
            if process.returncode != 0: # some error just happened
                print(f'output: {process.stdout.decode("utf-8")}')
                print(f'output: {process.stderr}')
                print(f'return code = {process.returncode}')
                trial -= 1
            else:
                break
        if trial == 0:
            print('Simulation failed')
            exit(1)

        os.chdir(origin_project_path)

    

    def _calculate_consumed_traffic(self):

        # This function analyze received traffic and calculate the remaining traffic matrix
        origin_project_path = os.getcwd()
        os.chdir(self.oment_simulation_paths[self.run_index])

        
        df = pd.read_csv("vectors.csv", low_memory=False)[['attrname','module', 'name','vecvalue']]
        df = df.query("name=='packetReceived:vector(packetBytes)' & vecvalue.notna()")
        data = df.to_dict('records')
        completed_traffic = np.zeros((self.num_nodes, self.num_nodes), dtype=np.float32)
        for d in data:
            bytes_received = np.array(d['vecvalue'].split(' '), dtype=np.float32) 
            bytes_sum = np.sum(bytes_received) # amount of bytes received
            module = d['module'] # the application of data transfer e.g.: Network.host4.app[4]
            source = module.split('.')[1][4:]
            destination = module.split('.')[2][4:-1]
            if int(destination) == 0: # ignore application 0
                continue
            completed_traffic[int(source)-1][int(destination)-1] += bytes_sum/1000 # in kBs

        os.chdir(origin_project_path)

        return completed_traffic
    
    
    def _configure_weights(self, link_weights):
        """
        Create routing strategy and files with specified link weights.
        """
        # Set weights
        for edge in self.G.edges.data():
            index = int(edge[2]['index'].split("_")[0]) - 1
            edge[2]['weight'] = link_weights[index]

        shortest_paths = nx.shortest_path(self.G, weight="weight")
        routing_string = self._generate_routing_config(shortest_paths)

        routing_path = os.path.join(self.oment_simulation_paths[self.run_index], "routing.xml")
        with open(self.config['routing_file'], 'r') as file:
            routing_template = file.read()

        routing_template = routing_template.replace("<RUN_ID>", str(self.run_index + 1))
        routing_template = routing_template.replace("<SHORTEST_PATH_ROUTING_CONFIG>", routing_string)

        with open(routing_path, 'w') as file:
            file.write(routing_template)

    def _generate_routing_config(self, shortest_paths):
        """
        Generate routing configuration string from shortest paths.
        """
        routing_string = ""
        for source in range(self.num_nodes):
            current_hops = 0
            source += 1
            paths = shortest_paths[source]  # Shortest paths from source to all other reachable nodes

            for destination in paths.keys():
                if source != destination:
                    next_hop = paths[destination][1]
                    gateway, interface = self.ethernet[f'({source}, {next_hop})']
                    routing_string += f'    <route hosts="node{source}" destination="192.168.{destination}.0" netmask="255.255.255.0" gateway="{gateway}" interface="{interface}"/>\n'
        return routing_string
    

    def _configure_traffic_settings(self, traffic, duration):
        # This function apply the traffic to the networks
        ini_path = os.path.join(self.oment_simulation_paths[self.run_index], "omnetpp.ini")
        ini_template = open(self.config['ini_file'], 'r').read()
        traffic = traffic.reshape(self.num_nodes, self.num_nodes) # traffic  matrix
        traffic_string = ""
        for source in range(self.num_nodes):
            for destination in range(self.num_nodes):
                amount = str(int(traffic[source][destination]))
                traffic_string += f'*.host{source+1}.app[{destination+1}].sendBytes = {amount}kB\n'

        ini_template = ini_template.replace("<TRAFFIC_PATTERN>", traffic_string)
        ini_template = ini_template.replace("<TIME_LIMIT>", str(duration) + "s")
        ini_template = ini_template.replace("<RUN_ID>", str(self.run_index + 1))
        ini_template = ini_template.replace("<QUEUE_CAP>", str(self.queue_capacity))

        with open(ini_path, 'w') as f:
            f.write(ini_template)

    
    def _configure_link_status(self):

        """Update the broken link status and write it into the ned file."""
        ned_path = os.path.join(self.oment_simulation_paths[self.run_index], "package.ned")
        with open(self.config['ned_file'], 'r') as file:
            ned_template = file.read()

        links_100M = {
            "connected": ' <--> Eth100M { @display("ls=#CE995C,3,s"); } <--> ',
            "disconnected": " <--> Eth100M {  disabled = true; } <--> ",
        }

        links_10M = {
            "connected": ' <--> Eth10M { @display("ls=#277A5B,2,s"); } <--> ',
            "disconnected": " <--> Eth10M {  disabled = true; } <--> ",
        }

        connection_strings = []
        port_strings = [(f"node{port[0][0]}.ethg[{port[0][1]}]", f"node{port[1][0]}.ethg[{port[1][1]}]") for port in self.ports]

        for i in range(self.num_links):
            link_type = links_100M if i in self.large_links else links_10M
            connection_strings.append(f"        {port_strings[i][0]}{link_type['connected']}{port_strings[i][1]};\n")

        for j in self.broken_links:
            link_type = links_100M if j in self.large_links else links_10M
            connection_strings[j] = f"        {port_strings[j][0]}{link_type['disconnected']}{port_strings[j][1]};\n"

        connection_string = "".join(connection_strings)
        ned_template = ned_template.replace("<RUN_ID>", str(self.run_index + 1))
        ned_template = ned_template.replace("<CONNECTIONS>", connection_string)

        with open(ned_path, 'w') as file:
            file.write(ned_template)
    
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

    def get_current_traffic_amount(self):
        return self.traffic_manager.get_current_traffic_matrix().sum().item()


class QosAnalzer():
    def __init__(self, run_index, simulation_path, ports):
        self.run_index = run_index
        self.simulation_path = simulation_path
        self.ports = ports

    def analyze_qos(self):
        self._create_csv_files()
        avg_delay = self._analyze_end_to_end_delay()
        avg_loss_rate = self._analyze_packet_loss_rate()
        link_traffics = self._analyze_link_traffic()

        return avg_delay, avg_loss_rate, link_traffics

    def _create_csv_files(self):
        """
        Creates 'vectors.csv' and 'scalars.csv' from the simulation output files.
        """

        origin_project_path = os.getcwd()
        # Changing to the directory where the simulation results are stored
        os.chdir(self.simulation_path)

        # Command to create 'vectors.csv' from the '.vec' files
        vector_cmd = " ".join(["opp_scavetool", "export", "-o", "vectors.csv", "./results/*.vec"])
        vector_process = subprocess.run(vector_cmd, shell=True, capture_output=True)

        # Command to create 'scalars.csv' from the '.sca' files
        scalar_cmd = " ".join(["opp_scavetool", "export", "-o", "scalars.csv", "./results/*.sca"])
        scalar_process = subprocess.run(scalar_cmd, shell=True, capture_output=True)

        # Changing back to the original project directory
        os.chdir(origin_project_path)

    def _analyze_end_to_end_delay(self):
        """
        Analyzes the end-to-end delay of packets from the vectors.csv file.
        Returns the average delay.
        """
        try:
            # Reading the vectors.csv file
            df = pd.read_csv(os.path.join(self.simulation_path, "vectors.csv"), low_memory=False)
            df = df[['attrname', 'module', 'name', 'vecvalue']]
            
            # Filtering for end-to-end delay data
            end_to_end_delay_df = df.query("name == 'endToEndDelay:vector' and vecvalue.notna()")

            app_delays = []
            for record in end_to_end_delay_df.to_dict('records'):
                delays = np.array(record['vecvalue'].split(' '), dtype=float)
                mean_delay = np.mean(delays) if delays.size > 0 else 0

                module = record['module']
                source = module.split('.')[1][4:]
                destination = module.split('.')[2][-2:-1]

                if source != destination and mean_delay > 0:
                    app_delays.append(mean_delay)

            # Calculating the average delay
            avg_delay = np.mean(app_delays) if app_delays else 0

            return avg_delay

        except Exception as e:
            print(f"Error in analyzing end-to-end delay: {e}")
            exit(1)

    def _analyze_packet_loss_rate(self):
        """
        Analyzes the packet loss rate from the scalars.csv file.
        Returns the average packet loss rate.
        """
        # Reading the scalars.csv file
        df = pd.read_csv(os.path.join(self.simulation_path, "scalars.csv"), low_memory=False)
        df = df[['module', 'name', 'value']]

        # Filtering data for packet loss and total packets
        df_lost_drop = df.query("name == 'droppedPacketsQueueOverflow:count' and value.notna()")
        df_lost_down = df.query("name == 'packetDropInterfaceDown:count' and value.notna()")
        df_total = df.query("name == 'incomingPackets:count' and value.notna()")

        data_total = df_total.to_dict('records')
        data_lost_drop = df_lost_drop.to_dict('records')
        data_lost_down = df_lost_down.to_dict('records')

        # Calculating packet loss rate
        data = list(zip(data_total, data_lost_drop, data_lost_down))
        loss_rates = []
        for scalar in data:
            total_packets = int(scalar[0]['value'])
            lost_packets = int(scalar[1]['value']) + int(scalar[2]['value'])
            if total_packets != 0:
                loss_rates.append(lost_packets / total_packets)
            else:
                continue
        assert len(loss_rates)!= 0 # must be at least one loss rate
        avg_loss_rate = np.mean(loss_rates)

        return avg_loss_rate
            
    def _analyze_link_traffic(self):
        """
        Analyzes the traffic on each link from the scalars.csv file.
        Returns an array representing the traffic on each link.
        """
        try:
            # Reading the scalars.csv file
            df = pd.read_csv(os.path.join(self.simulation_path, "scalars.csv"), low_memory=False)
            df = df[['module', 'name', 'value']]

            # Filtering data for outgoing packets
            df_outgoing = df.query("name == 'outgoingPackets:count' and value.notna()")

            # Initializing the traffic array for each link
            link_traffics = np.zeros(len(self.ports) * 2)

            for record in df_outgoing.to_dict('records'):
                outgoing_packets = int(record['value'])
                module_parts = record['module'].split('.')
                source = int(module_parts[1][4:])  # Extracting source node number
                port = int(module_parts[2][-2:-1])  # Extracting port number

                # Identifying the link index in the traffic array
                for i, port_pair in enumerate(self.ports):
                    if (source, port) in port_pair:
                        index = port_pair.index((source, port))
                        link_traffics[2 * i + index] += outgoing_packets

            # Optionally, normalize or scale the traffic data if needed
            link_traffics /= 100000  # Example scaling, adjust as needed

            return link_traffics.reshape(1, -1)

        except Exception as e:
            print(f"Error in analyzing link traffic: {e}")
            exit(1)
    
    
    
    
