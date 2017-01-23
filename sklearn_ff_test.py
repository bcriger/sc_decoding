import cPickle as pkl
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import stabiliser_sampling as ss

d = 5

attr_dict = {
    "hidden_layer_sizes" : (60,60,60),
    "activation" : "logistic",
    "batch_size" : 10000,
    "max_iter" : 2000,
    "random_state" : 1
}

classifier = MLPClassifier(**attr_dict)

with open('train_dict_d5_p10.pkl', 'r') as phil:
    train_dict = pkl.load(phil)

X, y = (train_dict[ki] for ki in ('input', 'labels')) 

classifier.fit(X, y)
# score = classifier.score(X_test, y_test)

ps = np.linspace(0.001, 0.2, 20)
log_ps = np.zeros_like(ps)
n_trials = 20000
for pdx, p in enumerate(ps):
    with open('test_dict_d5_p' + str(pdx) + '.pkl', 'r') as phil:
        test_dict = pkl.load(phil)
    temp_X, temp_y = (test_dict[ki] for ki in ('input', 'labels'))
    log_ps[pdx] = 1. - classifier.score(temp_X, temp_y) #p_error