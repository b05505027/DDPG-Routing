# Getting Started
## building of omnetpp (windows version)
1. Go to [download omnetpp v5.5.1](https://omnetpp.org/download/old) and unzip it in the current directory.
Then cd to the 'omnetpp-5.5.1' directory.
Checkout configure.user, in line 21, replace 'PREFER_CLANG=yes' with 'PREFER_CLANG=no'.
2. Start mingwenv and press any key to start unpacking the tool chain. (may takes a while)
3. Type ./configure to set up our IDE (it takes only for seconds)
4. Type make to build the library.
5. Start the IDE and create a workspace directory at omnetpp-5.5.1\workspace.
6. Be sure to install the INET package when you first launch the IDE.
7. Right click the INET package > build project.
8. Create your own project in the workspace.
9. Right click the project icon > properties > project references > check inet4


## building of omnetpp (mac version)


1. source ./setenv
2. ./configure
3cl. make


<!-- 
## building of INET Framework
1. Go to  .\omnetpp-5.5.1 and type 'git clone https://github.com/inet-framework/inet.git' to clone the repo.
2. Import it into your workspace.
3. Right click the icon in the explorer and select 'build project'. -->


