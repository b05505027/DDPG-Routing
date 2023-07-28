import numpy as np 
import matplotlib.pyplot as plt
import json

configs = [
    {
        "score_path": "experiments/0005_005_train_3_step_2_layer/scores.json",
        "smoothing": 100,
        "color": '#2CBDFE', # blue
        "legend": "0005_005_train_3_step_2_layer",
        "scaling": 1,
    },
    {
        "score_path": "experiments/001_01_train_3_step_2_layer/scores.json",
        "smoothing": 100,
        "color": '#47DBCD', # green
        "legend": "001_01_train_3_step_2_layer",
        "scaling": 1,
    },
    {
        "score_path": "experiments/001_01_train_1_step_2_layer/scores.json",
        "smoothing": 100,
        "color": '#F3A0F2', # pink
        "legend": "001_01_train_1_step_2_layer",
        "scaling": 3,
    },
]



title = "training_scores"
# configs = [
#     {
#         "score_path": "experiments/0_0_test_0005_005_train_3_step_2_layer/scores_test.json",
#         "smoothing": 1,
#         "color": '#2CBDFE', # blue
#         "legend": "0005_005_train_3_step_2_layer",
#         "scaling": 1,
#     },
#     {
#         "score_path": "experiments/0_0_test_001_01_train_3_step_2_layer/scores_test.json",
#         "smoothing": 1,
#         "color": '#47DBCD', # green
#         "legend": "001_01_train_3_step_2_layer",
#         "scaling": 1,
#     },
#     {
#         "score_path": "experiments/0_0_test_001_01_train_1_step_2_layer/scores_test.json",
#         "smoothing": 1,
#         "color": '#F3A0F2', # pink
#         "legend": "001_01_train_1_step_2_layer",
#         "scaling": 1,
#     },
#     {
#         "score_path": "experiments/test_random/scores_test.json",
#         "smoothing": 1,
#         "color": '#BDBDBD', # grey
#         "legend": "random_actor",
#         "scaling": 1,
#     },
# ]

# configs = [
#     {
#         "score_path": "experiments/broken_1_test_0005_005_train_3_step_2_layer/scores_test.json",
#         "smoothing": 1,
#         "color": '#2CBDFE', # blue
#         "legend": "0005_005_train_3_step_2_layer",
#         "scaling": 1,
#     },
#     {
#         "score_path": "experiments/broken_1_test_001_01_train_3_step_2_layer/scores_test.json",
#         "smoothing": 1,
#         "color": '#47DBCD', # green
#         "legend": "001_01_train_3_step_2_layer",
#         "scaling": 1,
#     },
#     {
#         "score_path": "experiments/broken_1_test_001_01_train_1_step_2_layer/scores_test.json",
#         "smoothing": 1,
#         "color": '#F3A0F2', # pink
#         "legend": "001_01_train_1_step_2_layer",
#         "scaling": 1,
#     },
#     {
#         "score_path": "experiments/broken_1_test_random/scores_test.json",
#         "smoothing": 1,
#         "color": '#BDBDBD', # grey
#         "legend": "random_actor",
#         "scaling": 1,
#     },
# ]

plt.figure(figsize=(40, 20))
plt.rcParams.update({'font.size': 30})
ax = plt.gca()
ax.grid(True)
plt.subplot(111)
plt.title(title)

for config in configs:
    scores = json.load(open(config['score_path'], 'r')) 
    scores = np.array(scores) * config['scaling']
    plt.plot(np.convolve(scores, np.ones(config['smoothing'])/config['smoothing'], mode='valid'), color=config['color'], linewidth=3, label=config['legend'] + "_smoothing=" + config['smoothing'].__str__())
    plt.plot(np.ones(len(scores)) * np.mean(scores), "--", color=config['color'], linewidth=3)
plt.legend()
plt.savefig(title+".png")