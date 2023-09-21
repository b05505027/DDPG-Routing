import json
import matplotlib.pyplot as plt
import numpy as np

def load_json_files(file_paths):
    data_lists = []
    for file_path in file_paths:
        try:
            with open(file_path, "r") as file:
                data = json.load(file)
                data_lists.append(data)
        except FileNotFoundError:
            print(f"File not found: {file_path}")
        except json.JSONDecodeError:
            print(f"Invalid JSON format in file: {file_path}")
    return data_lists

def create_plot(data, labels=None):
    print(len(data[2]))
    means = [np.mean(dataset) for dataset in data]
    std_devs = [np.std(dataset) for dataset in data]
    x_values = np.arange(len(data))
    plt.figure(figsize=(12, 6))
    #yerr=std_devs,
    colors = ['royalblue', 'royalblue', 'royalblue', 'royalblue', 'royalblue', 'orange', 'orange']
    plt.style.use('ggplot')
    plt.bar(x_values, means,  capsize=10, align='center', alpha=0.7, color=colors)
    #set y axis range
    plt.ylim(-6.5,-5.0)
    plt.xticks(x_values, labels)
    plt.title("Scores of models with different failure rates")
    plt.xlabel("models")
    plt.ylabel("scores")

    plt.savefig("f_plot.png", dpi=300, bbox_inches="tight")

if __name__ == "__main__":
    file_paths = ["experiments/test_f=50/scores_test.json", 
                    "experiments/test_f=100/scores_test.json",
                    "experiments/test_f=200/scores_test.json",
                    #"experiments/test_f=500/scores_test.json",
                    "experiments/[4300]test_f=500/scores_test.json",
                    # "experiments/test_f=600/scores_test.json",
                    # "experiments/test_f=700/scores_test.json",
                    # "experiments/test_f=800/scores_test.json",
                    # "experiments/test_f=900/scores_test.json",
                    "experiments/test_f=1000/scores_test.json",
                    "experiments/test_is_f=50/scores_test.json",
                    "experiments/test_is_f=200/scores_test.json",
                ]
    data_lists = load_json_files(file_paths)

    # Optional: Specify labels for the box plot
    labels = ["f = 1/50", "f=1/100", "f = 1/200","f = 1/500", "f = 1/1000", "f = 1/50 (IS)", "f = 1/200 (IS)"]

    create_plot(data_lists, labels)