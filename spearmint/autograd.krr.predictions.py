"""
Predictions for a kernel ridge regression that uses phylogenetic regularization.
Spearmint is used to select the model hyperparameters.

"""
import cPickle as c
import json
import numpy as np
import os

from phyloreg.autograd_classifiers import AutogradRidgeRegression
from collections import defaultdict
from phyloreg.species import ExponentialAdjacencyMatrixBuilder
from simple_spearmint import SimpleSpearmint
from sklearn.metrics import roc_auc_score
from time import time


def cross_validation(phylo_tree, train_data, folds, params):
    sgd_shuffler = np.random.RandomState(42)

    # Create the species adjacency matrix (and disable it)
    species, adjacency = \
        ExponentialAdjacencyMatrixBuilder(sigma=1e-100)(phylo_tree)
    adjacency = np.eye(adjacency.shape[0])  # Only self connections

    example_ids = np.array(train_data["labels"].keys(), dtype=np.uint)
    fold_aucs = []
    for fold in np.unique(folds):
        print "... Fold", fold
        train_ids = np.array(example_ids[folds != fold])
        sgd_shuffler.shuffle(train_ids)
        test_ids = example_ids[folds == fold]

        # Prepare the training data
        X_train = np.vstack((train_data["labelled_examples"][i] for i in train_ids))
        y_train = np.array([train_data["labels"][i] for i in train_ids], dtype=np.uint8)

        # Build the orthologs dictionnary
        orthologs = defaultdict(lambda: {"X": [], "species": []})

        # Prepare the testing data
        X_test = np.vstack((train_data["labelled_examples"][i] for i in test_ids))
        y_test = np.array([train_data["labels"][i] for i in test_ids], dtype=np.uint8)

        # Fit the classifier
        clf = AutogradRidgeRegression(alpha=params["alpha"],
                              beta=0.,
                              fit_intercept=True,
                              opti_lr=params["opti_lr"],
                              opti_tol=1e-5,
                              opti_max_epochs=300,
                              opti_patience=5,
                              opti_batch_size=20,
                              opti_clip_norm=params["opti_clip_norm"])
        clf.fit(X=X_train,
                X_species=["hg38"] * X_train.shape[0],
                y=y_train,
                orthologs=orthologs,
                species_graph_adjacency=adjacency,
                species_graph_names=species)

        # Compute the AUC on the testing set and add it to the fold AUC list
        fold_aucs.append(roc_auc_score(y_true=y_test, y_score=clf.predict(X_test)))

    # Compute the mean AUC and return its value
    return np.mean(fold_aucs)


def train_test_with_fixed_params(train_data, test_data, phylo_tree, params):
    sgd_shuffler = np.random.RandomState(42)

    # Create the species adjacency matrix (and disable it)
    species, adjacency = \
        ExponentialAdjacencyMatrixBuilder(sigma=1-100)(phylo_tree)
    adjacency = np.eye(adjacency.shape[0])  # Only self connections

    # Prepare the training data
    train_ids = np.array(train_data["labels"].keys())  # Use the entire training set
    sgd_shuffler.shuffle(train_ids)
    X_train = np.vstack((train_data["labelled_examples"][i] for i in train_ids))
    y_train = np.array([train_data["labels"][i] for i in train_ids], dtype=np.uint8)

    # Build the orthologs dictionnary
    orthologs = defaultdict(lambda: {"X": [], "species": []})

    # Prepare the testing data
    test_ids = test_data["labels"].keys()  # Use the entire testing set
    X_test = np.vstack((test_data["labelled_examples"][i] for i in test_ids))
    y_test = np.array([test_data["labels"][i] for i in test_ids], dtype=np.uint8)

    # Fit the classifier
    clf = AutogradRidgeRegression(alpha=params["alpha"],
                          beta=0.,
                          fit_intercept=True,
                          opti_lr=params["opti_lr"],
                          opti_tol=1e-5,
                          opti_max_epochs=300,
                          opti_patience=5,
                          opti_batch_size=20,
                          opti_clip_norm=params["opti_clip_norm"])
    clf.fit(X=X_train,
            X_species=["hg38"] * X_train.shape[0],
            y=y_train,
            orthologs=orthologs,
            species_graph_adjacency=adjacency,
            species_graph_names=species)

    test_predictions = clf.predict(X_test)
    auc = roc_auc_score(y_true=y_test, y_score=test_predictions)
    return test_predictions, auc, clf


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.DEBUG,
                            format="%(asctime)s.%(msecs)d %(levelname)s %(module)s - %(funcName)s: %(message)s")

    bootstrap_file = None  #"predictions/autograd.krr.269.spearmint"
    training_data_file = "../data/270.pkl"
    testing_data_file = "../data/269.pkl"
    phylo_tree_file = "../data/phylogenetic_tree.json"
    n_cv_folds = 3
    random_state = np.random.RandomState(42)
    n_parameter_combinations = 310
    n_random_combinations = 10
    output_path = os.path.join("predictions", "autograd.krr.{0!s}".format(os.path.basename(testing_data_file).replace(".pkl", "")))

    parameter_space = {'alpha': {'type': 'float', 'min': 1e-8, 'max': 1e4},
                       'opti_lr': {'type': 'float', 'min': 1e-5, 'max': 1e-1},
                       'opti_clip_norm': {'type': 'float', 'min': 1e0, 'max': 1e3}}

    # Load the training data
    train_data = c.load(open(training_data_file, "r"))
    phylo_tree = json.load(open(phylo_tree_file, "r"))

    # Make cross-validation folds
    folds = np.arange(len(train_data["labels"])) % n_cv_folds
    random_state.shuffle(folds)

    # Create a Spearmint object
    ss = SimpleSpearmint(parameter_space, minimize=False)

    # Bootstrap with previous spearmint run
    if bootstrap_file is not None:
        print "ATTENTION: Bootstrapping with previous spearmint run! Number of random suggestions will be reset to 0."
        n_random_combinations = 0
        bootstrap = json.load(open(bootstrap_file, "r"))
        for run in bootstrap:
            ss.update(run["params"], run["cv_score"] * -1)  # Scores are - because we maximize...
        print "Bootstrapped with", len(bootstrap), "parameter combinations"

    # Hyperparameter selection
    np.random.seed(42)  # ss doesnt take a random state so we set the global numpy seed
    best_model_cv_score = -np.infty if bootstrap_file is None else ss.get_best_parameters()[1]
    best_params = None if bootstrap_file is None else ss.get_best_parameters()[0]
    print "Best score is", best_model_cv_score
    for i in xrange(n_parameter_combinations):
        # Get a suggestion from Spearmint
        print "Hyperparameter combination #", i + 1, "/", n_parameter_combinations
        if i < n_random_combinations:
            params = ss.suggest_random()
        else:
            params = ss.suggest()
        print "The suggestion is:", params

        # Try this suggestion using cross-validation
        t = time()
        cv_score = cross_validation(phylo_tree, train_data, folds, params=params)
        print "CV score is", cv_score, "-- Took", time() - t, "seconds.\n\n"

        # Determine the optimality of the suggestion
        if cv_score > best_model_cv_score:
            best_model_cv_score = cv_score
            best_params = params
            print "Found a better model!\n\n"
        elif np.isclose(cv_score, best_model_cv_score):
            # Tiebreaker: pick the model with the least amount of regularization,
            # starting with manifold regularization. This way, we will only see
            # a large beta if it is really helping.
            if params["beta"] < best_params["beta"] or \
               (np.isclose(params["beta"], best_params["beta"]) and params["alpha"] < best_params["alpha"]):
                best_model_cv_score = cv_score
                best_params = params
                print "Found a better model!\n\n"

        # Feed the result back to Spearmint
        ss.update(params, cv_score)

        # Checkpoint: save the hyperparameter combinations and their objective values so we can relaunch if needed
        json.dump([dict(params=p, cv_score=v) for p, v in zip(ss.parameter_values, ss.objective_values)], open(output_path + ".spearmint", "w"))

    # Predictions on testing set
    print "\n\nRe-training with best parameters and predicting on testing data"
    print "The best parameters are:", best_params
    test_data = c.load(open(testing_data_file, "r"))
    test_predictions, test_auc, model = \
        train_test_with_fixed_params(train_data, test_data, phylo_tree, best_params)
    print "Test AUC is", test_auc

    # Save test predictions and best parameters
    open(output_path, "w").write("\n".join(str(x) for x in test_predictions))
    json.dump(best_params, open(output_path + ".parameters", "w"))
