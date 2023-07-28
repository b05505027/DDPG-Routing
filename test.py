import numpy as np
from utils import get_state_distribution

# def binary_states(n_bits):
#     if n_bits >= 1:
#         states = binary_states(n_bits-1)*2
        
#         for i in range(len(states)//2):
#             states[i] += str(0)
            
#         for i in range(len(states)//2, len(states)):
#             states[i] += str(1)
            
#     else:
#         states = [""]
#     return states
        

# def get_state_distribution(n_links, max_broken_links, failure_rate, recovery_rate):
#     states = binary_states(max_broken_links)
#     transition_matrix = np.ones((len(states), len(states)), dtype=np.float32)
#     for i in range(len(states)):
#         for j in range(len(states)):
#             for k in range(max_broken_links):
#                 if states[i][k] == states[j][k]:
#                     if states[i][k] == "0":
#                         transition_matrix[i][j] *= 1 - failure_rate
#                     else:
#                         transition_matrix[i][j] *= 1 - recovery_rate
#                 else:
#                     if states[i][k] == "0":
#                         transition_matrix[i][j] *= failure_rate
#                     else:
#                         transition_matrix[i][j] *= recovery_rate
#     for i in range(len(states)):
#         states[i] = states[i].zfill(n_links)[::-1]

#     # transpose it to get the right order
#     transition_matrix = transition_matrix.T
#     eigenvals, eigenvects = np.linalg.eig(transition_matrix)

#     '''
#     Find the indexes of the eigenvalues that are close to one.
#     Use them to select the target eigen vectors. Flatten the result.
#     '''
#     close_to_1_idx = np.isclose(eigenvals,1)
#     target_eigenvect = eigenvects[:,close_to_1_idx]
#     target_eigenvect = target_eigenvect[:,0]
#     # Turn the eigenvector elements into probabilites
#     stationary_distrib = target_eigenvect / sum(target_eigenvect) 

#     return states, stationary_distrib

if __name__ == '__main__':
    states, stationary_distrib = get_state_distribution(7, 4, 0.01, 0.1)
    print(states)
    print(stationary_distrib)

    # state '0011000'
    print(stationary_distrib[states.index('0011000')])