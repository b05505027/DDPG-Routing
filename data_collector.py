import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import json
import numpy as np
from pathlib import Path
import os

class DataCollector:
    def __init__(self, dir_path):
        self.data = {}  # Use a dictionary to store data
        self.dir_path = Path(dir_path)

    def log_data(self, records):
        for data_type, value in records.items():
            if data_type not in self.data:
                self.data[data_type] = []  # Create a list for the data type if it doesn't exist
            self.data[data_type].append(value)

    def plot_all_data(self, window_size=10):
        sns.set_theme(style="whitegrid")
        plot_types = list(self.data.keys())
        no_plot_types = ['exploration_rates', 'broken_links']
        for dtype in no_plot_types:
            try:
                plot_types.remove(dtype)
            except ValueError:
                continue
        num_plots = len(plot_types)

        # Define columns and rows
        num_columns = 3
        num_rows = num_plots // num_columns + 1

        plt.figure(figsize=(num_columns * 6, num_rows * 4))

        colors = sns.color_palette("tab10", n_colors=num_plots)

        no_moving_average_types = ["exploration_rates"]

        for i, (data_type, color) in enumerate(zip(plot_types, colors), start=1):
            # Determine the row and column for the subplot
            row = (i - 1) // num_columns
            col = (i - 1) % num_columns

            # Calculate subplot index
            subplot_index = row * num_columns + col + 1

            # Create subplot
            ax = plt.subplot(num_rows, num_columns, subplot_index)

            if self.data[data_type] not in no_moving_average_types and len(self.data[data_type]) >= window_size:
                moving_avg = np.convolve(self.data[data_type], np.ones(window_size) / window_size, mode='valid')
            else:
                moving_avg = self.data[data_type]

            # Calculate and plot the average line
            average_value = np.mean(self.data[data_type])
            plt.axhline(y=average_value, color='r', linestyle='--', label='Average')

            sns.lineplot(data=moving_avg, label=data_type, color=color)
            plt.xlabel('Time')
            plt.ylabel('Value')
            plt.title(f'{data_type.capitalize()} Data Plot')

            if (data_type == 'scores' or data_type == 'is_scores') and 'exploration_rates' in self.data:
                ax2 = ax.twinx()
                sns.lineplot(data=self.data['exploration_rates'], label='Exploration Rates', color='#F2BFC8', linestyle="--")
                ax2.set_ylabel('Exploration Rate')
                ax2.legend(loc='upper right')

            ax.legend(loc='upper left')

        plt.tight_layout()

        plot_filename = self.dir_path / Path('plot.png')
        plt.savefig(plot_filename)
        plt.close('all')
    
    def save_all_data(self):

        # Save each data type to a separate file
        for data_type, values in self.data.items():
            file_path = self.dir_path / Path(f'{data_type}.json')
            with open(file_path, 'w') as file:
                json.dump(values, file, indent=4)

    def save_all_data2(self, action):
        file_path = self.dir_path / Path('action.json')
        with open(file_path, 'a') as file:
            # f.write(str(action.tolist()[0])+"\n")
            json.dump(action.tolist()[0], file, indent=4)
        # f.close()