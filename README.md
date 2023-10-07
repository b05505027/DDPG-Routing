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


## building of omnetpp (linux version)

1.  sudo nano /etc/apt/sources.list
2. Add this entry to the file and save:
deb [trusted=yes] http://cz.archive.ubuntu.com/ubuntu bionic main universe
3. sudo apt-get update
4. sudo apt-get install build-essential gcc g++ bison flex perl tcl-dev tk-dev libxml2-dev zlib1g-dev default-jre doxygen graphviz libwebkitgtk-1.0-0
5. Download OMNeT++: https://omnetpp.org/download/old
6. Unpacking: $ tar xvfz omnetpp-5.0-src.tgz
7. vi ~/.bachrc and add: export PATH=$PATH:/home/jialun/omnetpp-5.5.1/bin export OMNET_DIR=:/home/jialun/omnetpp-5.5.1
8. source ~/.bashrc
9. Edit configure.user file: vi omnetpp-5.5.1/configure.user set value WITH_TKENV=no and WITH_QTENV=no and WITH_OSG=no and WITH_OSGEARTH=no 
10. Type terminal: $ ./configure
11. When ./configure has finished, compile OMNeT++. Type in the terminal: $ make

---- make sure you have generated your public ssh key ---
sh public/private key pair set beforec
1. cd ~/.ssh && ssh-keygen (DSA or RSA)
2. Next you need to copy this to your clipboard. Add your key to your account via the website.
3. Finally setup your .gitconfig.
git config --global user.name "bob"
git config --global user.email bob@...  
------------------------------------------------------------
12. wget ghttps://github.com/inet-framework/inet/releases/download/v4.1.2/inet-4.1.2-src.tgz and tar zxvf to unzip it.
13. Change to the INET directory and source the setenv script. $ source setenv
14. Type make makefiles. This should generate the makefiles for you automatically.
---------------------- python3 not found -------------------
1. whereis python3
2. sudo ln -s /usr/bin/python3 /usr/bin/python
-----------------------------------------------------------
15. Type make to build the inet executable (release version).
16. You can run specific examples by changing into the example's directory and executing inet.
