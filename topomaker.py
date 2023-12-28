import os
import json
class TopoMaker:
    def __init__(self, name, adjacency_matrix, link_info):
        self.name = name
        self.adjacency_matrix = adjacency_matrix
        self.link_info = link_info
        self.num_nodes = len(adjacency_matrix)
        self.num_links = len(link_info)

        self.port_data = self.generate_port_data()
        self.edges_data = self.generate_edges_data()
        self.ethernet_data = self.generate_ethernet_data()

        self.omnet_config_content = self.generate_omnet_config()
        self.ned_file_content = self.generate_ned_file()
        self.routing_xml_content = self.generate_routing_xml()

    # Function to generate port data
    def generate_port_data(self):
        port_data = []
        port_counters = [0] * (self.num_nodes + 1)

        for i in range(1, self.num_nodes+1):
            for j in range(i, self.num_nodes+1):
                if self.adjacency_matrix[i-1][j-1] == 1:
                    port_i = port_counters[i]
                    port_j = port_counters[j]
                    port_counters[i] += 1
                    port_counters[j] += 1
                    port_data.append([[i, port_i], [j, port_j]])

        return port_data

    # Function to generate edges data
    def generate_edges_data(self):
        edges_data = []
        for lid, (a, b) in enumerate(self.link_info, start=1):
            edge_data_1 = [a, b, {"weight": 0, "traffic": 0, "index": f"{lid}_1"}]
            edge_data_2 = [b, a, {"weight": 0, "traffic": 0, "index": f"{lid}_2"}]
            edges_data.extend([edge_data_1, edge_data_2])
        return edges_data
    
    # Helper function to format IP address
    def format_ip(self, lid, node):
        return f"10.0.{lid}.{node}"

    # Function to generate ethernet data
    def generate_ethernet_data(self):
        ethernet_data = {}

        # Mapping ports data to links
        link_to_ports = {}
        for link in self.port_data:
            # Creating a tuple (a, b) for each link
            a, b = link[0][0], link[1][0]
            a_port, b_port = link[0][1], link[1][1]

            link_to_ports[(a, b)] = a_port
            link_to_ports[(b, a)] = b_port

        # Constructing Ethernet data
        for lid, (a, b) in enumerate(link_info, start=1):
            a_to_b_key = f"({a}, {b})"
            b_to_a_key = f"({b}, {a})"

            # Fetching port numbers
            a_port = link_to_ports.get((a, b), None)
            b_port = link_to_ports.get((b, a), None)

            # Creating Ethernet data entries
            ethernet_data[a_to_b_key] = [self.format_ip(lid, b), f"eth{a_port}"]
            ethernet_data[b_to_a_key] = [self.format_ip(lid, a), f"eth{b_port}"]

        return ethernet_data

    # Function to generate OMNeT++ configuration file content
    def generate_omnet_config(self):
        num_apps = self.num_nodes + 1  # Number of nodes + 1

        config_content = '''[General]
network = drl_routing.simulations<RUN_ID>.Network

# recording
*.node*.**.vector-recording = false
*.host*.eth[*].**.vector-recording = false
*.host*.tcp.**.vector-recording = false
*.host*.lo.**.vector-recording = false

# scalar recording
*.host*.**.scalar-recording = false
*.node*.ethernet.**.scalar-recording = false
*.node*.ipv4.**.scalar-recording = false
*.node*.lo[*].**.scalar-recording = false
*.node*.eth[*].mac.**.scalar-recording = false

# queuing settings...
**.node*.eth[*].queue.typename = "DropTailQueue"
**.node*.eth[*].queue.packetCapacity = <QUEUE_CAP>

## Visualizer settings
sim-time-limit = <TIME_LIMIT>

# Visualizer settings...

# IP settings 
# Using inline XML configuration
*.configurator.config.addStaticRoutes = false
*.configurator.config = xmldoc("routing.xml")

# Application settings
*.host*.numApps = {num_apps} # number of apps
        '''.format(num_apps=num_apps)


        for k in range(1, self.num_nodes + 1):
            config_content += '''
*.host*.app[{k}].typename = "TcpSessionApp"
*.host*.app[{k}].connectAddress = "host{k}"
*.host*.app[{k}].connectPort = 1000
            '''.format(k=k)

        config_content += '''
*.host*.app[0].typename = "TcpEchoApp" # server application type
*.host*.app[0].localPort = 1000 # TCP server listen port

<TRAFFIC_PATTERN>

        '''

        return config_content

    # Function to generate .ned file content
    def generate_ned_file(self):
    
        ned_content = '''package drl_routing.simulations<RUN_ID>;
import inet.node.ethernet.Eth10M;
import inet.examples.inet.ipv4hook.MyHost;
import inet.examples.inet.ipv4hook.MyRouter;
import inet.networklayer.configurator.ipv4.Ipv4NetworkConfigurator;
import inet.node.ethernet.Eth100M;
import inet.node.ethernet.Eth1G;
import inet.node.inet.Router;
import inet.node.inet.StandardHost;

network Network
{
    parameters:
        @display("bgb=1022,469");

    submodules:
        configurator: Ipv4NetworkConfigurator {
                @display("p=511,37");
        }
        '''
        # Create router and host submodules
        for k in range(1, self.num_nodes + 1):
            ports_used = sum(1 for key in self.ethernet_data if f'({k},' in key)
            ned_content += f'''
        node{k}: Router {{
            @display("p={k*100},{k*100}");
            numEthInterfaces = {ports_used+1};
        }}
        host{k}: StandardHost {{
            @display("p={k*100-50},{k*100-50}");
            numEthInterfaces = 1;
        }}'''

        ned_content += '''
connections allowunconnected: // network level connections

<CONNECTIONS>
        '''

        # Add connections for each host to its router
        for k in range(1, self.num_nodes + 1):
            ports_used = sum(1 for key in self.ethernet_data if f'({k},' in key)
            ned_content += f'''host{k}.ethg++ <--> Eth10M <--> node{k}.ethg[{ports_used}];\n'''

        ned_content += '''}\n'''

        return ned_content


    # Function to generate routing.xml file content
    def generate_routing_xml(self):
        routing_xml = "<config>\n"

        # Host Interfaces Configuration
        for k in range(1, self.num_nodes + 1):
            routing_xml += f'    <interface hosts="host{k}" names="eth0" address="192.168.{k}.2" netmask="255.255.255.0"/>\n'
        
        # Node Interfaces Configuration
        for node in range(1, self.num_nodes + 1):
            ports_used = sum(1 for key in self.ethernet_data if f'({node},' in key)
            routing_xml += f'    <interface hosts="node{node}" names="eth{ports_used}" address="192.168.{node}.1" netmask="255.255.255.0"/>\n'
        
        # Node to Node Interface Configuration
        for key, value in self.ethernet_data.items():
            src_node, des_node = eval(key)
            des_ip = self.ethernet_data[f'({des_node}, {src_node})'][0]
            routing_xml += f'    <interface hosts="node{src_node}" names="{value[1]}" address="{des_ip}" netmask="255.255.255.0"/>\n'
        routing_xml += f'    <interface hosts="**" address="10.x.x.x" netmask="255.x.x.x" />\n'
        # Basic Host Routing Configuration
        routing_xml += '''\n\n<!-- basic host configuration -->\n'''
        for k in range(1, self.num_nodes + 1):
            routing_xml += f'''    <route hosts="host{k}" destination="192.168.{k}.0" netmask="255.255.255.0" gateway="*" interface="eth0"/>\n'''
            routing_xml += f'''    <route hosts="host{k}" destination="0.0.0.0" netmask="0.0.0.0" gateway="192.168.{k}.1" interface="eth0"/>\n'''

        # Basic Node Routing Configuration
        routing_xml += '''\n\n<!-- basic node configuration -->\n'''
        for node in range(1, self.num_nodes + 1):
            ports_used = sum(1 for key in self.ethernet_data if f'({node},' in key)
            routing_xml += f'    <route hosts="node{node}" destination="192.168.{node}.0" netmask="255.255.255.0" gateway="*" interface="eth{ports_used}"/>\n'
            for key, value in self.ethernet_data.items():
                if f'({node},' in key:
                    dest_ip = value[0]
                    routing_xml += f'    <route hosts="node{node}" destination="{dest_ip}" netmask="255.255.255.255" gateway="*" interface="{value[1]}"/>\n'

        routing_xml += "<SHORTEST_PATH_ROUTING_CONFIG>\n</config>"
        return routing_xml

        
    def generate_topology_config(self, project_path):
        config = {
                'num_nodes': self.num_nodes,
                'num_links': self.num_links,
                'lam_t': 1,
                'lam_r': 20,
                'edges': f"topology/{self.name}/edge_files/edges.json",
                'ethernet': f"topology/{self.name}/edge_files/ethernet.json",
                'ports': f"topology/{self.name}/edge_files/ports.json",
                'large_links': f"topology/{self.name}/edge_files/large_links.json",
                'ini_file': f"topology/{self.name}/config_files/omnetpp_tmp.ini",
                'ned_file': f"topology/{self.name}/config_files/package_tmp.ned",
                'routing_file': f"topology/{self.name}/config_files/routing_tmp.xml",
                'oment_simulation_paths': [
                    os.path.join(project_path, "simulations" + str(i)) for i in range(1, 51)
                ]
        }

        config_path = f"topology/{self.name}/config_files"
        os.makedirs(config_path, exist_ok=True)
        with open(os.path.join(config_path, 'config.json'), 'w') as file:
            json.dump(config, file, indent=4)

        return config_path
    def dump_topology_data(self, project_path):

        self.generate_topology_config(project_path)
        # Define file paths
        edge_file_path = f"topology/{self.name}/edge_files/edges.json"
        ethernet_file_path = f"topology/{self.name}/edge_files/ethernet.json"
        ports_file_path = f"topology/{self.name}/edge_files/ports.json"
        large_links_file_path = f"topology/{self.name}/edge_files/large_links.json"
        ini_file_path = f"topology/{self.name}/config_files/omnetpp_tmp.ini"
        ned_file_path = f"topology/{self.name}/config_files/package_tmp.ned"
        routing_file_path = f"topology/{self.name}/config_files/routing_tmp.xml"

        # Create necessary directories
        os.makedirs(os.path.dirname(edge_file_path), exist_ok=True)
        os.makedirs(os.path.dirname(ini_file_path), exist_ok=True)

        # Dump JSON data
        for path, data in zip([edge_file_path, ethernet_file_path, ports_file_path], 
                            [self.edges_data, self.ethernet_data, self.port_data]):
            with open(path, 'w') as file:
                json.dump(data, file, indent=4)

        # Dump large links (empty list)
        with open(large_links_file_path, 'w') as file:
            json.dump([], file, indent=4)

        # Write INI, NED, and XML files
        for path, content in zip([ini_file_path, ned_file_path, routing_file_path], 
                                [self.omnet_config_content, self.ned_file_content, self.routing_xml_content]):
            with open(path, 'w') as file:
                file.write(content)


if __name__ == '__main__':

    # Example usage
    # adjacency_matrix = [
    #     [0, 1, 1, 0, 0],
    #     [1, 0, 1, 1, 0],
    #     [1, 1, 0, 1, 1],
    #     [0, 1, 1, 0, 1],
    #     [0, 0, 1, 1, 0]
    # ]
    #link_info = [(1, 3), (1, 2), (2, 4), (3, 4), (3, 5), (4, 5), (2, 3)]
    
        
    # adjacency_matrix = [
    #     [0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    #     [1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1],
    #     [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    #     [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
    #     [0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0],
    #     [0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0],
    #     [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0]
    # ]

    # link_info = [
    #     (1, 2), (1, 3), (1, 4), (2, 3), (2, 8), (3, 6), (4, 5), (4, 9),
    #     (5, 6), (5, 7), (6, 13), (6, 14), (7, 8), (8, 11), (9, 10), (9, 12),
    #     (10, 11), (10, 13), (11, 12), (11, 14), (12, 13)
    # ]

    adjacency_matrix = [
        [0, 1, 0, 0, 0, 0, 1, 0, 0],
        [1, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 1, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 1],
        [0, 0, 1, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 1, 0, 0],
        [1, 0, 0, 0, 0, 1, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 1],
        [0, 0, 0, 1, 0, 0, 0, 1, 0,],
    
    ]

    link_info = [
        (1, 2), (1, 7), (2, 3), (3, 4), (3, 5), (4,9),  (5,6), (6,7), (7,8), (8,9)
    ]

    topology_name = "nsfnet2"
    simulation_path = "/home/jialun/samples/drl_routing" # The base path for the OMNET++ experiment folders.
    
    topo_maker = TopoMaker(topology_name, adjacency_matrix, link_info)
    topo_maker.dump_topology_data(simulation_path)

    print("Finish generating data of topology {}!".format(topology_name))
