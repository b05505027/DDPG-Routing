import numpy as np 
import matplotlib.pyplot as plt
import json

configs_train = [
    {
        "score_path": "experiments/001_01_is/scores.json",
        "smoothing": 200,
        "color": '#2CBDFE', # blue
        "legend": "001_01_is",
        "scaling": 1,
    },
    {
        "score_path": "experiments/001_01_nois/scores.json",
        "smoothing": 200,
        "color": '#47DBCD', # green
        "legend": "001_01_nois",
        "scaling": 1,
    },
    {
        "score_path": "experiments/0001_01_nois/scores.json",
        "smoothing": 200,
        "color": '#F3A0F2', # pink
        "legend": "0001_01_nois",
        "scaling": 1,
    },
]


configs_test = [
     {
        "score_path": "experiments/test_001_01_is/scores_test.json",
        "smoothing": 100,
        "color": '#2CBDFE', # blue
        "legend": "001_01_is",
        "scaling": 1,
    },
    {
        "score_path": "experiments/test_001_01_nois/scores_test.json",
        "smoothing": 100,
        "color": '#47DBCD', # green
        "legend": "001_01_nois",
        "scaling": 1,
    },
    {
        "score_path": "experiments/test_0001_01_nois/scores_test.json",
        "smoothing": 100,
        "color": '#F3A0F2', # pink
        "legend": "0001_01_nois",
        "scaling": 1,
    },
    {
        "score_path": "experiments/test_random/scores_test.json",
        "smoothing": 100,
        "color": '#BDBDBD', # grey
        "legend": "test_random",
        "scaling": 1,
    },
]




configs_q = [
     {
        "score_path": "experiments/test_001_01_is/q_values1_test.json",
        "smoothing": 100,
        "color": '#2CBDFE', # blue
        "legend": "Q_values_from_001_01_is",
        "scaling": 1,
    },
    {
        "score_path": "experiments/test_001_01_is/q_values2_test.json",
        "smoothing": 100,
        "color": '#47DBCD', # green
        "legend": "Q_values_from_001_01_nois",
        "scaling": 1,
    },
    {
        "score_path": "experiments/test_001_01_is/true_q_values_test.json",
        "smoothing": 100,
        "color": '#F3A0F2', # pink
        "legend": "true_Q_values",
        "scaling": 1,
    },
]

configs_q2 = [
     {
        "score_path": "experiments/test_0001_01_nois/q_values1_test.json",
        "smoothing": 100,
        "color": '#2CBDFE', # blue
        "legend": "Q_values_from_0001_01_nois",
        "scaling": 1,
    },
    # {
    #     "score_path": "experiments/test_001_01_is/q_values2_test.json",
    #     "smoothing": 100,
    #     "color": '#47DBCD', # green
    #     "legend": "Q_values_from_001_01_nois",
    #     "scaling": 1,
    # },
    # {
    #     "score_path": "experiments/test_random/scores_test.json",
    #     "smoothing": 100,
    #     "color": '#BDBDBD', # grey
    #     "legend": "test_random",
    #     "scaling": 1,
    # },
]



for i in range(4):
    if i == 0:
        configs = configs_train
        title = "training_scores"
    elif i == 1:
        configs = configs_test
        title = "testing_scores"
    elif i == 2:
        configs = configs_q
        title = "q_values_during_testing (actor:001_01_is)"
    elif i == 3:
        configs = configs_q2
        title = "q_values_during_testing (actor:0001_01_nois)"
        

    plt.figure(figsize=(40, 20))
    plt.rcParams.update({'font.size': 30})
    ax = plt.gca()
    ax.grid(True)
    plt.subplot(111)
    plt.title(title)

    min_length = 100000000
    for config in configs:
        scores = json.load(open(config['score_path'], 'r')) 
        min_length = min(min_length, len(scores))
    for config in configs:
        scores = json.load(open(config['score_path'], 'r'))[:min_length]
        scores = np.array(scores) * config['scaling']
        plt.plot(np.convolve(scores, np.ones(config['smoothing'])/config['smoothing'], mode='valid'), color=config['color'], linewidth=5, label=config['legend'] + "_smoothing=" + config['smoothing'].__str__())
        plt.plot(np.ones(len(scores)) * np.mean(scores), "--", color=config['color'], linewidth=5)
    plt.legend()
    plt.savefig(title+".png")