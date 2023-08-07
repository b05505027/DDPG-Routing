import numpy as np 
import matplotlib.pyplot as plt
import json

configs_train = [
    {
        "score_path": "experiments/001_01_train_is_1_step/scores.json",
        "smoothing": 200,
        "color": '#2CBDFE', # blue
        "legend": "001_01_train_is_1_step",
        "scaling": 3,
    },
    {
        "score_path": "experiments/001_01_train_nois_1_step/scores.json",
        "smoothing": 200,
        "color": '#47DBCD', # green
        "legend": "001_01_train_nois_1_step",
        "scaling": 3,
    },
    {
        "score_path": "experiments/0001_01_train_nois_1_step/scores.json",
        "smoothing": 200,
        "color": '#F3A0F2', # pink
        "legend": "0001_01_train_nois_1_step",
        "scaling": 3,
    },
    {
        "score_path": "experiments/001_01_train_nois_3_step/scores.json",
        "smoothing": 200,
        "color": '#9D2EC5', # purple
        "legend": "001_01_train_nois_3_step",
        "scaling": 1,
    },
]


configs_test = [
    {
        "score_path": "experiments/test_001_01_train_nois_3_step/scores_test.json",
        "smoothing": 100,
        "color": '#2CBDFE', # blue
        "legend": "test_001_01_train_nois_3_step",
        "scaling": 1,
    },
    {
        "score_path": "experiments/test_0001_01_train_nois_1_step/scores_test.json",
        "smoothing": 100,
        "color": '#47DBCD', # green
        "legend": "test_0001_01_train_nois_1_step",
        "scaling": 1,
    },
    {
        "score_path": "experiments/_____test_001_01_train_nois_1_step/scores_test.json",
        "smoothing": 100,
        "color": '#F3A0F2', # pink
        "legend": "test_001_01_train_nois_1_step",
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





for i in range(2):
    if i == 0:
        configs = configs_train
        title = "training_scores"
    else:
        configs = configs_test
        title = "testing_scores"
        

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