package drl_routing.simulations6;

import inet.node.ethernet.Eth10M;
import inet.examples.inet.ipv4hook.MyHost;
import inet.examples.inet.ipv4hook.MyRouter;
import inet.networklayer.configurator.ipv4.Ipv4NetworkConfigurator;
import inet.node.ethernet.Eth100M;
import inet.node.ethernet.Eth1G;
import inet.node.inet.Router;
import inet.node.inet.StandardHost;
import inet.visualizer.common.IntegratedVisualizer;


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
            numEthInterfaces = 5;
        }
        node2: Router {
            @display("p=243,339");
            numEthInterfaces = 5;
        }
        node3: Router {
            @display("p=437,192");
            numEthInterfaces = 5;
        }
        node4: Router {
            @display("p=437,339");
            numEthInterfaces = 5;
        }
        node5: Router {
            @display("p=595,260");
            numEthInterfaces = 5;
        }
        host1: StandardHost {
            @display("p=169,144");
            numEthInterfaces = 1;
        }
        host2: StandardHost {
            @display("p=133,340");
            numEthInterfaces = 1;
        }
        host3: StandardHost {
            @display("p=437,111");
            numEthInterfaces = 1;
        }
        host5: StandardHost {
            @display("p=686,260");
            numEthInterfaces = 1;
        }
        host4: StandardHost {
            @display("p=437,411");
            numEthInterfaces = 1;
        }
        Visualizer: IntegratedVisualizer {
            @display("p=550,111");
        }
    connections allowunconnected: // network level connections


<CONNECTIONS>


        host1.ethg++ <--> Eth10M <--> node1.ethg[2];
        host2.ethg++ <--> Eth10M <--> node2.ethg[3];
        host4.ethg++ <--> Eth10M <--> node4.ethg[3];
        host3.ethg++ <--> Eth10M <--> node3.ethg[4];
        host5.ethg++ <--> Eth10M <--> node5.ethg[2];

}

