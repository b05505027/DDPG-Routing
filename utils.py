import os
import random
import numpy as np
import torch

def set_pid_file(run_index, pid):
    """Write the PID to a file for a given index."""
    with open(f'./lock_files/pid_{run_index}.txt', 'w') as f:
        f.write(str(pid))

def get_pid(run_index):
    """Get the PID from the file for a given index."""
    try:
        with open(f'./lock_files/pid_{run_index}.txt', 'r') as f:
            return int(f.read().strip())
    except FileNotFoundError:
        # If the file doesn't exist, assume the index is free
        return None

def is_pid_running(pid):
    """Check if a PID is still running."""
    try:
        os.kill(pid, 0)  # 0 signal does not kill the process
    except OSError as e:
        return False
    else:
        return True

def find_available_run_index():
    """Find an available run index and set its PID file."""
    current_pid = os.getpid()  # Get current process ID
    found_index = False

    for run_index in range(50):

        pid = get_pid(run_index)
        if pid is None or not is_pid_running(pid):

            # Set the PID file to indicate the index is in use by current process
            set_pid_file(run_index, current_pid)
            found_index = True

            return run_index

    if not found_index:
        return None

def same_seed(seed):
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  if torch.cuda.is_available():
      torch.cuda.manual_seed_all(seed)


def create_directory_with_alias(directory_path, alias_suffix=1):
    origin_path = directory_path
    # Check if the directory at 'directory_path' already exists
    while os.path.exists(directory_path):
        # If it exists, create an alias directory name by appending a numeric suffix
        alias_directory = f"{origin_path}({alias_suffix})"
        alias_suffix += 1
        # Update 'directory_path' to the alias directory path for the next iteration
        directory_path = alias_directory

    # When a non-existing directory name is found, create the directory
    os.makedirs(directory_path)
    # Return the full path of the newly created directory
    return directory_path

    

        

    