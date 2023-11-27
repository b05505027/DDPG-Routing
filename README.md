# Getting Started

This document provides comprehensive instructions for training and testing the ddpg agents, as well as steps for setting up OMNeT++ for network simulations on both Windows and Linux platforms.

## Usage

### Training
To initiate the training process, use the `train_agent` function:

```python
train_agent(training_f=1000, importance_sampling=False, name_prefix="")
```

#### Parameters

- `training_f` (float): Lambda failure value for training.
- `importance_sampling` (bool): Enable (`True`) or disable (`False`) importance sampling.
- `name_prefix` (str): Prefix for experiment repository names.

#### Advanced Configuration

Adjust the following parameters in the `config` dictionary within the `train_agent` function for advanced configurations:

- `run_index` (int): Index of the `.ini` or `.ned` file. This is typically assigned by the system.
- `total_traffic` (float): Amplitude of sinusoidal traffic pattern.
- `testing_lam_f` (float): Lambda failure for the testing environment.
- `alpha` (float): Assigns the weighting of the delay reward in the reward function, with the loss rate reward weighted as \(1 - \alpha\).
- `horizon` (int): The length of an episode in training.
- `gamma` (float): The discount factor used in the Markov Decision Process (MDP).
- `buffer_size` (int): The size of the replay buffer.
- `batch_size` (int): The number of samples taken from the replay buffer for training.
- `layer_size` (list of int): Specifies the sizes of the hidden layers for both the actor and critic networks.
- `initial_random_steps` (int): The number of epochs at the start of training during which random actions are executed before actual training begins.


### Testing

To start the testing process, use the `test_agent` function:

```python
test_agent(session_name="f=1000", epoch=200, name_prefix='new_setting')
# or for OSPF testing
test_agent(ospf=True)
```

#### Parameters

- `session_name` (str): Session name containing the model for testing.
- `epoch` (int): Model epoch for testing.
- `ospf` (bool): Use OSPF for testing (`True`) or not (`False`).
- `name_prefix` (str): Prefix for experiment repository names for identification.

#### Advanced Configuration

Refer to the training section for additional parameter adjustments.

## OMNeT++ Installation

### Windows Version

1. Download and unzip OMNeT++ v5.5.1 from [here](https://omnetpp.org/download/old). Navigate to 'omnetpp-5.5.1' directory.
2. In `configure.user`, change 'PREFER_CLANG=yes' to 'PREFER_CLANG=no'.
3. Open `mingwenv`, press any key to unpack the toolchain.
4. Run `./configure` to set up the IDE, followed by `make` to build the library.
5. Start the IDE and create a workspace at `omnetpp-5.5.1/workspace`.
6. Install and build the INET package.
7. Create and configure your project in the workspace, ensuring INET is referenced.

### Linux Version

1. Add the Ubuntu bionic main universe repository: `sudo nano /etc/apt/sources.list`, then add: `deb [trusted=yes] http://cz.archive.ubuntu.com/ubuntu bionic main universe`.
2. Update and install dependencies: `sudo apt-get update`, `sudo apt-get install [list of dependencies]`.
3. Download and unpack OMNeT++ from [here](https://omnetpp.org/download/old).
4. Update environment variables in `~/.bashrc` and source it.
5. Modify `configure.user` as per requirements (disabling certain features).
6. Run `./configure` and `make` to compile OMNeT++.

### SSH Key and Git Configuration

1. Generate an SSH key pair: `cd ~/.ssh && ssh-keygen` (choose DSA or RSA).
2. Copy the public key to your clipboard and add it to your account via the website.
3. Set up Git configuration:
   ```
   git config --global user.name "Your Name"
   git config --global user.email your_email@example.com
   ```

### INET Framework Setup

1. Download INET from [INET releases](https://github.com/inet-framework/inet/releases) (e.g., `inet-4.1.2-src.tgz`).
2. Unpack INET: `tar zxvf inet-4.1.2-src.tgz`.
3. Navigate to the INET directory and source the setenv script: `source setenv`.
4. Generate makefiles: `make makefiles`.
5. Build the INET executable: `make`.