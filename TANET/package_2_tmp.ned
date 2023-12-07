package drl_routing.simulations2;

import inet.node.ethernet.Eth10M;
import inet.examples.inet.ipv4hook.MyHost;
import inet.examples.inet.ipv4hook.MyRouter;
import inet.networklayer.configurator.ipv4.Ipv4NetworkConfigurator;
import inet.node.ethernet.Eth100M;
import inet.node.ethernet.Eth1G;
import inet.node.inet.Router;
import inet.node.inet.StandardHost;


@license(LGPL);
//
// TODO documentation
//


network Network
{
    parameters:
        //int numClients; // number of clients in the network
        @display("bgb=1380.74,1724.344");
    submodules:
        configurator: Ipv4NetworkConfigurator {
            @display("p=191.828,73.78");
        }
        node1: Router {
            @display("p=1024.488,446.896;is=s");
            numEthInterfaces = 18;
        }
        node2: Router {
            @display("p=746.232,687.208;is=s");
            numEthInterfaces = 18;
        }
        node3: Router {
            @display("p=440.57202,801.04004;is=s");
            numEthInterfaces = 18;
        }
        node4: Router {
            @display("p=514.352,1203.668;is=s");
            numEthInterfaces = 18;
        }
        node5: Router {
            @display("p=1056.108,238.20401;is=s");
            numEthInterfaces = 18;
        }
        host1: StandardHost {
            @display("p=1172.048,560.728");
            numEthInterfaces = 18;
        }
        host2: StandardHost {
            @display("p=309.876,678.776");
            numEthInterfaces = 1;
        }
        host3: StandardHost {
            @display("p=193.936,824.228");
            numEthInterfaces = 18;
        }
        host5: StandardHost {
            @display("p=1113.024,122.264");
            numEthInterfaces = 18;
        }
        host4: StandardHost {
            @display("p=265.608,1365.984");
            numEthInterfaces = 18;
        }
        node6: Router {
            @display("p=834.768,269.824;is=s");
            numEthInterfaces = 18;
        }
        node7: Router {
            @display("p=756.77203,406.844;is=s");
            numEthInterfaces = 18;
        }
        node8: Router {
            @display("p=535.432,619.752;is=s");
            numEthInterfaces = 18;
        }
        node9: Router {
            @display("p=573.37604,480.624;is=s");
            numEthInterfaces = 18;
        }
        node10: Router {
            @display("p=1180.48,425.816;is=s");
            numEthInterfaces = 18;
        }
        node11: Router {
            @display("p=994.976,642.94;is=s");
            numEthInterfaces = 18;
        }
        node12: Router {
            @display("p=950.708,963.356;is=s");
            numEthInterfaces = 18;
        }
        node13: Router {
            @display("p=562.836,944.38403;is=s");
            numEthInterfaces = 18;
        }
        node14: Router {
            @display("p=811.58,954.924;is=s");
            numEthInterfaces = 18;
        }
        node15: Router {
            @display("p=461.652,1094.052;is=s");
            numEthInterfaces = 18;
        }
        node16: Router {
            @display("p=638.724,1452.412;is=s");
            numEthInterfaces = 18;
        }
        node17: Router {
            @display("p=379.44,1347.012;is=s");
            numEthInterfaces = 18;
        }
        host6: StandardHost {
            @display("p=809.472,84.32");
            numEthInterfaces = 18;
        }
        host7: StandardHost {
            @display("p=634.508,200.26001");
            numEthInterfaces = 18;
        }
        host8: StandardHost {
            @display("p=389.98,512.244");
            numEthInterfaces = 18;
        }
        host9: StandardHost {
            @display("p=478.51602,335.172");
            numEthInterfaces = 18;
        }
        host10: StandardHost {
            @display("p=1256.368,320.41602");
            numEthInterfaces = 18;
        }
        host11: StandardHost {
            @display("p=1091.944,796.82404");
            numEthInterfaces = 18;
        }
        host12: StandardHost {
            @display("p=1030.812,1106.7001");
            numEthInterfaces = 18;
        }
        host13: StandardHost {
            @display("p=183.396,988.65204");
            numEthInterfaces = 18;
        }
        host14: StandardHost {
            @display("p=832.66003,1397.604");
            numEthInterfaces = 18;
        }
        host15: StandardHost {
            @display("p=236.09601,1176.264");
            numEthInterfaces = 18;
        }
        host16: StandardHost {
            @display("p=630.292,1629.484");
            numEthInterfaces = 18;
        }
        host17: StandardHost {
            @display("p=449.004,1479.816");
            numEthInterfaces = 18;
        }

    connections allowunconnected: // network level connections


<CONNECTIONS>

        host1.ethg++ <--> Eth10M <--> node1.ethg[0];
        host2.ethg++ <--> Eth10M <--> node2.ethg[0];
        host3.ethg++ <--> Eth10M <--> node3.ethg[0];
        host4.ethg++ <--> Eth10M <--> node4.ethg[0];
        host5.ethg++ <--> Eth10M <--> node5.ethg[0];
        host6.ethg++ <--> Eth10M <--> node6.ethg[0];
        host7.ethg++ <--> Eth10M <--> node7.ethg[0];
        host8.ethg++ <--> Eth10M <--> node8.ethg[0];
        host9.ethg++ <--> Eth10M <--> node9.ethg[0];
        host10.ethg++ <--> Eth10M <--> node10.ethg[0];
        host11.ethg++ <--> Eth10M <--> node11.ethg[0];
        host12.ethg++ <--> Eth10M <--> node12.ethg[0];
        host13.ethg++ <--> Eth10M <--> node13.ethg[0];
        host14.ethg++ <--> Eth10M <--> node14.ethg[0];
        host15.ethg++ <--> Eth10M <--> node15.ethg[0];
        host16.ethg++ <--> Eth10M <--> node16.ethg[0];
        host17.ethg++ <--> Eth10M <--> node17.ethg[0];




  	    

}

