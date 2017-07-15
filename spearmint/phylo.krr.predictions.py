"""
Predictions for a kernel ridge regression that uses phylogenetic regularization.
Spearmint is used to select the model hyperparameters.

"""
import cPickle as c
import json
import numpy as np
import os

from collections import defaultdict
from phyloreg.classifiers import RidgeRegression
from phyloreg.species import ExponentialAdjacencyMatrixBuilder
from simple_spearmint import SimpleSpearmint
from sklearn.metrics import roc_auc_score
from time import time


def cross_validation(phylo_tree, train_data, folds, params):
    # Load orthologous sequences
    orthologs = defaultdict(lambda: {"X": [], "species": []})
    orthologs.update(train_data["ortho_info"])

    # Create the species adjacency matrix
    species, adjacency = \
        ExponentialAdjacencyMatrixBuilder(sigma=params["sigma"])(phylo_tree)

    example_ids = np.array(train_data["labels"].keys(), dtype=np.uint)
    fold_aucs = []
    for fold in np.unique(folds):
        print "... Fold", fold
        train_ids = example_ids[folds != fold]
        test_ids = example_ids[folds == fold]

        # Prepare the testing data
        X_train = np.vstack((train_data["labelled_examples"][i] for i in train_ids))
        y_train = np.array([train_data["labels"][i] for i in train_ids], dtype=np.uint8)

        # Prepare the testing data
        X_test = np.vstack((train_data["labelled_examples"][i] for i in test_ids))
        y_test = np.array([train_data["labels"][i] for i in test_ids], dtype=np.uint8)

        # Fit the classifier
        clf = RidgeRegression(alpha=params["alpha"],
                              beta=params["beta"],
                              normalize_laplacian=params["normalize_laplacian"],
                              fit_intercept=True)
        clf.fit(X=X_train,
                X_species=["hg38"] * X_train.shape[0],
                y=y_train,
                orthologs=orthologs,
                species_graph_adjacency=adjacency,
                species_graph_names=species)

        # Compute the AUC on the testing set and add it to the fold AUC list
        fold_aucs.append(roc_auc_score(y_true=y_test, y_score=clf.predict(X_test)))

    # Compute the mean AUC and return its value
    return np.mean(fold_aucs), clf


if __name__ == "__main__":
    training_data_file = "../data/270.pkl"
    testing_data_file = "../data/269.pkl"
    phylo_tree_file = "../data/phylogenetic_tree.json"
    n_cv_folds = 3
    random_state = np.random.RandomState(42)
    n_parameter_combinations = 100
    n_random_combinations = 5

    parameter_space = {'sigma': {'type': 'float', 'min': 1e-5, 'max': 1e0},
                       'alpha': {'type': 'float', 'min': 1e-8, 'max': 1e8},
                       'beta': {'type': 'float', 'min': 1e-8, 'max': 1e8},
                       'normalize_laplacian': {'type': 'enum', 'options': [True, False]}}

    # Load the training data
    train_data = c.load(open(training_data_file, "r"))
    phylo_tree = json.load(open(phylo_tree_file, "r"))

    # Make cross-validation folds
    folds = np.arange(len(train_data["labels"])) % n_cv_folds
    random_state.shuffle(folds)

    # Hyperparameter selection
    ss = SimpleSpearmint(parameter_space, minimize=False)
    best_model = None
    best_model_cv_score = -np.infty
    for i in xrange(n_parameter_combinations):
        print "Hyperparameter combination #", i + 1, "/", n_parameter_combinations
        if i < n_random_combinations:
            params = ss.suggest_random()
        else:
            params = ss.suggest()
        print "The suggestion is:", params

        t = time()
        cv_score, model = cross_validation(phylo_tree, train_data, folds, params=params)
        print "CV score is", cv_score, "-- Took", time() - t, "seconds.\n\n"

        if cv_score > best_model_cv_score:
            best_model = model
            best_model_cv_score = cv_score
            print "Found a better model!\n\n"

        ss.update(params, cv_score)

    # Predictions on testing set
    test_data = c.load(open(testing_data_file, "r"))
    test_predictions = best_model.predict(test_data["labelled_examples"])
    print "Test AUC is", roc_auc_score(y_true=test_data["labels"], y_score=test_predictions)
    open(os.path.join("predictions", "phylo.krr.{0!s}".format(os.path.basename(testing_data_file).replace(".pkl", ""))), "w").write("\n".join(str(x) for x in test_predictions))
