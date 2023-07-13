package drl_routing.simulations;

import inet.node.ethernet.Eth10M;
import inet.examples.inet.ipv4hook.MyHost;
import inet.examples.inet.ipv4hook.MyRouter;
import inet.networklayer.configurator.ipv4.Ipv4NetworkConfigurator;
import inet.node.ethernet.Eth100M;
import inet.node.ethernet.Eth1G;
import inet.node.ethernet.EtherHost;
import inet.node.inet.Router;
import inet.node.inet.StandardHost;
import inet.visualizer.integrated.IntegratedVisualizer;


@license(LGPL);
//
// TODO documentation
//


network Network
{
    parameters:
        //int numClients; // number of clients in the network
        @display("bgb=1022,469");
    submodules:
        configurator: Ipv4NetworkConfigurator {
            @display("p=511,37");
        }
        node1: Router {
            @display("p=243,192");
        }
        node2: Router {
            @display("p=243,339");
        }
        node3: Router {
            @display("p=437,192");
        }
        node4: Router {
            @display("p=437,339");
        }
        node5: Router {
            @display("p=595,260");
        }
        host1: StandardHost {
            @display("p=169,144");
        }
        host2: StandardHost {
            @display("p=133,340");
        }
        host3: StandardHost {
            @display("p=437,111");
        }
        host5: StandardHost {
            @display("p=686,260");
        }
        host4: StandardHost {
            @display("p=437,411");
        }
        visualizer: IntegratedVisualizer {
            @display("p=565,109");
        }
    connections: // network level connections
        node1.ethg++ <--> Eth10M <--> node3.ethg++;
        node1.ethg++ <--> Eth10M <--> node2.ethg++;
        node2.ethg++ <--> Eth10M <--> node4.ethg++;
        node3.ethg++ <--> Eth10M <--> node4.ethg++;
        node3.ethg++ <--> Eth10M <--> node5.ethg++;
        node4.ethg++ <--> Eth10M <--> node5.ethg++;
        node2.ethg++ <--> Eth10M <--> node3.ethg++;
        host1.ethg++ <--> Eth10M <--> node1.ethg++;
        host2.ethg++ <--> Eth10M <--> node2.ethg++;
        host4.ethg++ <--> Eth10M <--> node4.ethg++;
        host3.ethg++ <--> Eth10M <--> node3.ethg++;
        host5.ethg++ <--> Eth10M <--> node5.ethg++;
}

