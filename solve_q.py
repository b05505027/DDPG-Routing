import numpy as np
import torch 
from utils import get_state_distribution
import matplotlib.pyplot as plt
import random
import os
import tqdm

# states is a list of strings describing the broken links
states, test_stationary_distrib, test_transition_matrix = get_state_distribution(
                        n_links = 7,
                        max_broken_links = 4,
                        failure_rate = 0.001,
                        recovery_rate = 0.1)


for i in tqdm.tqdm(range(500)):
    f = 0.001 * random.randint(1, 100)
    r = 0.001 * random.randint(1, 1000)
    if i == 0:
        f = 0.01
        r = 0.1
    states, train_stationary_distrib, train_transition_matrix = get_state_distribution(
                                    n_links = 7,
                                    max_broken_links = 4,
                                    failure_rate = f,
                                    recovery_rate = r)

    ##### Plot the similarity matrix #####


    # set default font sizes
    plt.rcParams.update({'font.size': 16})
    fig, axs = plt.subplots(2,3)
    # set figure size
    fig.set_size_inches(36, 18)


    # draw the stationary distribution in the first subplot
    axs[0][0].bar(range(len(states)), test_stationary_distrib)
    axs[0][0].set_ylim((0, 1))
    axs[0][0].set_title('Test Stationary Distribution (failure_rate={}, recovery_rate={})'.format(0.001, 0.1))
    axs[0][0].set_ylabel('Probability')
    axs[0][0].set_xlabel('State')
    axs[0][0].set_xticks(range(len(states)))
    axs[0][0].grid(True)


    # draw the transition matrix in the second subplot with colorbar
    axs[1][0].imshow(test_transition_matrix, cmap=plt.cm.Blues)
    axs[1][0].set_title('Test Transition Matrix (failure_rate={}, recovery_rate={})'.format(0.001, 0.1))
    axs[1][0].set_ylabel('From')
    axs[1][0].set_xlabel('To')
    axs[1][0].set_xticks(range(len(states)))
    axs[1][0].set_yticks(range(len(states)))
    axs[1][0].grid(True)
    # add colorbar
    cbar = fig.colorbar(axs[1][0].imshow(test_transition_matrix, cmap=plt.cm.Blues), ax=axs[1][0])





    # draw the stationary distribution in the first subplot
    axs[0][1].bar(range(len(states)), train_stationary_distrib, color='orange')
    axs[0][1].set_ylim((0, 1))
    axs[0][1].set_title('Train Stationary Distribution (failure_rate={}, recovery_rate={})'.format(f, r))
    axs[0][1].set_ylabel('Probability')
    axs[0][1].set_xlabel('State')
    axs[0][1].set_xticks(range(len(states)))
    axs[0][1].grid(True)


    # draw the transition matrix in the second subplot with colorbar
    axs[1][1].set_title('Train Transition Matrix (failure_rate={}, recovery_rate={})'.format(f, r))
    axs[1][1].set_ylabel('From')
    axs[1][1].set_xlabel('To')
    axs[1][1].set_xticks(range(len(states)))
    axs[1][1].set_yticks(range(len(states)))
    axs[1][1].grid(True)
    # add colorbar
    cbar = fig.colorbar(axs[1][1].imshow(train_transition_matrix, cmap=plt.cm.Blues), ax=axs[1][1])



    # draw the stationary distribution in the first subplot
    axs[0][2].bar(range(len(states)), train_stationary_distrib - test_stationary_distrib, color='purple')
    # set y axis range to (0, 1)
    axs[0][2].set_ylim((-1, 1))
    axs[0][2].set_title('Stationary Distribution Difference')
    axs[0][2].set_ylabel('Probability')
    axs[0][2].set_xlabel('State')
    axs[0][2].set_xticks(range(len(states)))
    axs[0][2].grid(True)


    # draw the transition matrix in the second subplot with colorbar
    tau_matrix = test_transition_matrix/train_transition_matrix
    tau_matrix = np.nan_to_num(tau_matrix, nan=0, posinf=0, neginf=0)
    # normalize tau_matrix
    tau_matrix = tau_matrix / np.sum(tau_matrix)
    axs[1][2].set_title('Tau Matrix')
    axs[1][2].set_ylabel('From')
    axs[1][2].set_xlabel('To')
    axs[1][2].set_xticks(range(len(states)))
    axs[1][2].set_yticks(range(len(states)))
    axs[1][2].grid(True)
    # add colorbar
    cbar = fig.colorbar(axs[1][2].imshow(tau_matrix, cmap=plt.cm.Blues), ax=axs[1][2])


    # for i in range(len(states)):
    #     print('{i}: {state}'.format(i=i, state=states[i]))



    tau_matrix = tau_matrix.flatten()
    # calculate the entropy of tau_matrix
    entropy = -np.sum(tau_matrix * np.log(tau_matrix))
    # l2 weight of the tau matrix
    l2= np.sum(np.power(tau_matrix, 2))

    # calculate the entropy of the training stationary distribution
    entropy_train = -np.sum(train_stationary_distrib * np.log(train_stationary_distrib))
    
    score =  entropy_train  - 3*l2 + entropy
    print('entropy_train: {e}'.format(e=entropy_train))
    print('l2_norm: {l}'.format(l=l2))
    os.makedirs('distributions', exist_ok=True)
    if i==0:
        plt.savefig('distributions/baseline_score={s}_f={f}_r={r}_baselne.png'.format(s=score,f=f, r=r))
    else:
        plt.savefig('distributions/score={s}_f={f}_r={r}.png'.format(s=score,f=f, r=r))
    plt.close()
