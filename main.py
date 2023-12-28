import json
import os
from configs import SESSION_CONFIG
from ddpg import DDPGAgent, Mode
from environment import Environment
import re
from utils import create_directory_with_alias, find_available_run_index, same_seed

def train_agent(train_f: float, test_f: float, importance_sampling: bool, prefix="default"):
    """
    Train the DDPG agent with given frequencies, importance sampling option, and a name prefix.
    
    :param train_f: f for training.
    :param test_f: f for testing.
    :param importance_sampling: Boolean indicating if importance sampling is to be used.
    :param prefix: Prefix for the session name.
    """
    same_seed(2023)
    run_index = find_available_run_index()
    if run_index == None:
        print("All running indices are used. Please try again later.")
        exit(0)
    else:
        print("Use runnning index : {}".format(run_index))

    session_name = '_'.join(filter(None, [prefix, f"train_f={train_f}", f"test_f={test_f}", 'is' if importance_sampling else '']))
    directory_path = os.path.join("experiments", SESSION_CONFIG['topology'], session_name)
    directory_path = create_directory_with_alias(directory_path)
    _ = create_directory_with_alias(os.path.join(directory_path, 'models'))

    SESSION_CONFIG.update({
        'train_f': train_f,
        'test_f': test_f,
        'run_index': run_index,
        'importance_sampling': importance_sampling,
        'dir_path': directory_path,
        'mode': Mode.TRAIN
    })

    with open(os.path.join(directory_path, "config.json"), "w") as file:
        json.dump(SESSION_CONFIG, file, default=lambda x: str(x), indent=4)

    ddpg = initialize_ddpg_agent(SESSION_CONFIG)
    
    print(f'Training start for session: {session_name}')
    ddpg.train(importance_sampling=importance_sampling)

def test_agent(test_f: float, model_path: str, model_epoch: int, is_ospf: bool, prefix="default"):
    """
    Test the DDPG agent with a specific model, model_epoch, and OSPF option, along with a name prefix.
    
    :param test_f: f for testing.
    :param model_path: Path to the model for testing.
    :param model_epoch: Epoch number for the model.
    :param is_ospf: Boolean indicating if OSPF mode is to be used.
    :param prefix: Prefix for the session name.
    """
    same_seed(2024)
    run_index = find_available_run_index()
    if run_index == None:
        print("All running indices are used. Please try again later.")
        exit(0)

    if is_ospf:
        session_name = "ospf"
        mode = Mode.OSPF
        actor_path, critic_path = "", ""
    else:
        old_config = json.load(open(os.path.join(model_path, '..', 'config.json'), 'r'))
        session_name = re.search(r'/([^/]+)/models', model_path).group(1)
        mode = Mode.TEST
        actor_path = os.path.join(model_path, f"actor_{model_epoch}.ckpt")
        critic_path = os.path.join(model_path, f"critic_{model_epoch}.ckpt")

    session_name = '_'.join(filter(None, [prefix, f'test_evaluate_f={test_f}', session_name]))
    directory_path = os.path.join("experiments", SESSION_CONFIG['topology'], session_name)
    directory_path = create_directory_with_alias(directory_path)

    SESSION_CONFIG.update({
        'train_f': test_f,
        'test_f': test_f,
        'run_index': run_index,
        'dir_path': directory_path,
        'mode': mode,
        'is_ospf': is_ospf
    })

    with open(os.path.join(directory_path, "config.json"), "w") as file:
        json.dump(SESSION_CONFIG, file, default=lambda x: str(x), indent=4)

    ddpg = initialize_ddpg_agent(SESSION_CONFIG)
    print(f'Testing start for session: {session_name}')
    ddpg.test(actor_path, critic_path)

def initialize_ddpg_agent(config):
    """
    Initialize the DDPG agent with the provided configuration.

    :param config: A dictionary containing configuration parameters.
    :return: An instance of the DDPGAgent class.
    """
    return DDPGAgent(
        env=Environment(
            total_traffic=config['total_traffic'],
            run_index=config['run_index'],
            lam_f=config['train_f'],
            lam_f_test=config['test_f'],
            alpha=config['alpha'],
            max_broken_links=config['max_broken_links'],
            queue_capacity=config['queue_capacity'],
            topology=config['topology'],
            time_limit=config['time_limit'],
        ),
        buffer_size=config['buffer_size'],
        batch_size=config['batch_size'],
        gamma=config['gamma'],
        eps=config['eps'],
        tau=config['tau'],
        init_random_epochs=config['init_random_epochs'],
        horizon=config['horizon'],
        actor_lr=config['actor_lr'],
        critic_lr=config['critic_lr'],
        dir_path=config['dir_path'],
        nn_layers=config['nn_layers'],
        mode=config['mode'],
        topology=config['topology'],
    )


# Train the agent
train_agent(train_f=1000, test_f=1000, importance_sampling=False, prefix="Experiment_b1000_hor_100_gamma0.99_traffic_1000(test_nobroken)")
#train_agent(train_f=100, test_f=1000, importance_sampling=True, prefix="")


# Test the agent
# pretrained models
# test_agent(test_f=1000, 
#             model_path="experiments/5node/train_f=100_test_f=1000/models", 
#             model_epoch=100, 
#             is_ospf=False, 
#             prefix="")
# ospf
# test_agent(test_f=1000, 
#             model_path="", 
#             model_epoch=100, 
#             is_ospf=True, 
#             prefix="")