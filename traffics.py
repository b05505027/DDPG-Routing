import numpy as np

class TrafficManager:
    """
    Manages traffic matrices for a network of nodes. Generates traffic matrices in batches 
    for efficiency and allows for managing and updating these matrices.
    """

    def __init__(self, num_nodes, total_traffic, traffic_batch_size=100):
        """
        Initializes the TrafficManager with a specified number of nodes, total traffic, 
        and batch size for traffic matrix generation.
        """
        self.num_nodes = num_nodes
        self.total_traffic = total_traffic
        self.traffic_batch_size = traffic_batch_size
        self.traffic_matrices = self._generate_traffics(self.traffic_batch_size)
        self.traffic_index = 0
        self.current_traffic_matrix = np.zeros((num_nodes, num_nodes), dtype=np.float32)
        self.add_upcoming_traffic()

    def get_current_traffic_matrix(self):
        """
        Returns the current traffic matrix.
        """
        return self.current_traffic_matrix
    
    def add_upcoming_traffic(self):
        """
        Updates the current traffic matrix by adding the next matrix from the batch.
        """
        self.current_traffic_matrix += self._get_next_traffic()
        self.current_traffic_matrix = np.floor(self.current_traffic_matrix)
        self.current_traffic_matrix += (self.current_traffic_matrix == 0)
    
    def consume_traffic(self, traffic_amount):
        """
        Reduces the current traffic matrix by a specified traffic amount.
        """
        self.current_traffic_matrix -= traffic_amount 
        self.current_traffic_matrix = np.floor(self.current_traffic_matrix)
        self.current_traffic_matrix += (self.current_traffic_matrix == 0)

    def episode_end_traffic(self):

        self.current_traffic_matrix = np.ones((self.num_nodes, self.num_nodes), dtype=np.float32)

    def _get_next_traffic(self):
        """
        Retrieves the next traffic matrix from the batch, generating a new batch if necessary.
        """
        if self.traffic_index >= len(self.traffic_matrices):
            self.traffic_matrices = self._generate_traffics(self.traffic_batch_size)
            self.traffic_index = 0

        traffic_matrix = self.traffic_matrices[self.traffic_index]
        self.traffic_index += 1
        return traffic_matrix

    def _generate_traffics(self, traffic_number):
        """
        Generates a batch of traffic matrices using a combination of uniform distribution
        and Gaussian noise.
        """
        traffic_matrices = []
        uniform_traffic = np.ones(traffic_number)
        gaussian_noise = np.random.normal(0, 0.1, traffic_number)
        traffic_scales = uniform_traffic + gaussian_noise

        for i in range(traffic_number):
            random_vector = np.random.exponential(10, size=(self.num_nodes))
            traffic_size = self.total_traffic * traffic_scales[i]
            traffic = np.outer(random_vector, random_vector)
            traffic = traffic / np.sum(traffic) * traffic_size
            traffic_matrices.append(traffic)

        return traffic_matrices



# class TrafficManager:
#     """
#     Manages traffic matrices for a network of nodes, generating them in batches for efficiency.
#     """

#     def __init__(self, num_nodes, total_traffic, traffic_batch_size=100):
#         """
#         Initializes the TrafficManager with a given number of nodes, total traffic, and batch size.
#         """
#         self.num_nodes = num_nodes
#         self.total_traffic = total_traffic
#         self.traffic_batch_size = traffic_batch_size
#         self.traffic_matrices = self._generate_traffics(self.traffic_batch_size)
#         self.traffic_index = 0
#         self.current_traffic_matrix = np.zeros((num_nodes, num_nodes), dtype=np.float32)
#         self._update_traffic_matrix() # set inital update_traffic

#     def get_current_traffic_matrix(self):
#         return self.current_traffic_matrix
    
#     def add_upcoming_traffic(self):
#         self.current_traffic_matrix += self._get_traffic_matrix()
#         self.current_traffic_matrix = np.floor(self.current_traffic_matrix)
#         # turn zeros into ones
#         self.current_traffic_matrix = self.current_traffic_matrix + (self.current_traffic_matrix == 0)
    
#     def consume_traffic(self, traffic_amount):
#         self.current_traffic = self.current_traffic - traffic_amount 
#         self.current_traffic = np.floor(self.current_traffic)
#         # turn zeros into ones
#         self.current_traffic_matrix = self.current_traffic_matrix + (self.current_traffic_matrix == 0)

#     def _get_traffic_matrix(self):
#         """
#         Retrieves the next traffic matrix. Generates a new batch of matrices if the current batch is exhausted.
#         """
#         if self.traffic_index >= len(self.traffic_matrices):
#             self.traffic_matrices = self._generate_traffics(self.batch_size)
#             self.traffic_index = 0

#         traffic_matrix = self.traffic_matrices[self.traffic_index]
#         self.traffic_index += 1
#         self.current_traffic_matrix = traffic_matrix
#         return traffic_matrix

#     def _generate_traffics(self, traffic_number):
#         """
#         Generates a batch of traffic matrices.
#         """
#         traffic_matrices = []
#         uniform_traffic = np.ones(traffic_number) * 0.5
#         gaussian_noise = np.random.normal(0, 0.1, traffic_number)
#         traffic_scales = uniform_traffic + gaussian_noise

#         for i in range(traffic_number):
#             random_vector = np.random.exponential(10, size=(self.num_nodes))
#             traffic_size = self.total_traffic * traffic_scales[i]
#             traffic = random_vector * random_vector.reshape(-1, 1)
#             traffic = traffic / np.sum(traffic) * traffic_size
#             traffic_matrices.append(traffic)

#         # plt.plot(traffic_scales)
#         # plt.savefig('traffic_sizes1.png')

#         return traffic_matrices
