"""
Predictions for a logistic regression that uses phylogenetic regularization.
Spearmint is used to select the model hyperparameters.

"""
import cPickle as c
import json
import numpy as np
import os

from collections import defaultdict
from phyloreg.classifiers import LogisticRegression
from phyloreg.species import ExponentialAdjacencyMatrixBuilder
from simple_spearmint import SimpleSpearmint
from sklearn.metrics import roc_auc_score
from time import time


def cross_validation(phylo_tree, train_data, folds, params):

    # Create the species adjacency matrix
    species, adjacency = \
        ExponentialAdjacencyMatrixBuilder(sigma=params["sigma"])(phylo_tree)

    example_ids = np.array(train_data["labels"].keys(), dtype=np.uint)
    fold_aucs = []
    for fold in np.unique(folds):
        print "... Fold", fold
        train_ids = example_ids[folds != fold]
        test_ids = example_ids[folds == fold]

        # Prepare the training data
        X_train = np.vstack((train_data["labelled_examples"][i] for i in train_ids))
        y_train = np.array([train_data["labels"][i] for i in train_ids], dtype=np.uint8)

        # Build the orthologs dictionnary
        orthologs = defaultdict(lambda: {"X": [], "species": []})
        for i, id in enumerate(train_ids):
            # The ortholog dictionnary key must be the index of the example in the
            # feature matrix. The orthologs need to be in the same order as the
            # feature vectors and labels.
            if train_data["ortho_info"].has_key(id):
                orthologs[i] = train_data["ortho_info"][id]

        # Prepare the testing data
        X_test = np.vstack((train_data["labelled_examples"][i] for i in test_ids))
        y_test = np.array([train_data["labels"][i] for i in test_ids], dtype=np.uint8)

        # Fit the classifier
        clf = LogisticRegression(alpha=params["alpha"],
                                 beta=params["beta"],
                                 normalize_laplacian=params["normalize_laplacian"],
                                 fit_intercept=True,
                                 opti_max_iter=1e6,
                                 opti_lookahead_steps=X_train.shape[0],  # 1 epoch
                                 opti_tol=1e-4,
                                 opti_learning_rate=params["lr"],
                                 opti_learning_rate_decrease=params["lr_decrease"],
                                 stop_on_nan_inf=True)
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

    # Create the species adjacency matrix
    species, adjacency = \
        ExponentialAdjacencyMatrixBuilder(sigma=params["sigma"])(phylo_tree)

    # Prepare the training data
    train_ids = train_data["labels"].keys()  # Use the entire training set
    X_train = np.vstack((train_data["labelled_examples"][i] for i in train_ids))
    y_train = np.array([train_data["labels"][i] for i in train_ids], dtype=np.uint8)

    # Build the orthologs dictionnary
    orthologs = defaultdict(lambda: {"X": [], "species": []})
    for i, id in enumerate(train_ids):
        # The ortholog dictionnary key must be the index of the example in the
        # feature matrix. The orthologs need to be in the same order as the
        # feature vectors and labels.
        if train_data["ortho_info"].has_key(id):
            orthologs[i] = train_data["ortho_info"][id]

    # Prepare the testing data
    test_ids = test_data["labels"].keys()  # Use the entire testing set
    X_test = np.vstack((test_data["labelled_examples"][i] for i in test_ids))
    y_test = np.array([test_data["labels"][i] for i in test_ids], dtype=np.uint8)

    # Fit the classifier
    clf = LogisticRegression(alpha=params["alpha"],
                             beta=params["beta"],
                             normalize_laplacian=params["normalize_laplacian"],
                             fit_intercept=True,
                             opti_max_iter=1e6,
                             opti_lookahead_steps=X_train.shape[0],  # 1 epoch
                             opti_tol=1e-4,
                             opti_learning_rate=params["lr"],
                             opti_learning_rate_decrease=params["lr_decrease"],
                             stop_on_nan_inf=True)
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

    training_data_file = "../data/270.pkl"
    testing_data_file = "../data/269.pkl"
    phylo_tree_file = "../data/phylogenetic_tree.json"
    n_cv_folds = 3
    random_state = np.random.RandomState(42)
    n_parameter_combinations = 300
    n_random_combinations = 20

    parameter_space = {'sigma': {'type': 'float', 'min': 1e-5, 'max': 1e0},
                       'alpha': {'type': 'float', 'min': 1e-8, 'max': 1e8},
                       'beta': {'type': 'float', 'min': 1e-8, 'max': 1e8},
                       'normalize_laplacian': {'type': 'enum', 'options': [True, False]},
                       'lr': {'type': 'float', 'min': 1e-6, 'max': 1e-1},
                       'lr_decrease': {'type': 'float', 'min': 1e-6, 'max': 1e-1}}

    # Load the training data
    train_data = c.load(open(training_data_file, "r"))
    phylo_tree = json.load(open(phylo_tree_file, "r"))

    # Make cross-validation folds
    folds = np.arange(len(train_data["labels"])) % n_cv_folds
    random_state.shuffle(folds)

    # Hyperparameter selection
    ss = SimpleSpearmint(parameter_space, minimize=False)
    np.random.seed(42)  # ss doesnt take a random state so we set the global numpy seed
    best_model_cv_score = -np.infty
    best_params = None
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

    # Predictions on testing set
    print "\n\nRe-training with best parameters and predicting on testing data"
    print "The best parameters are:", best_params
    test_data = c.load(open(testing_data_file, "r"))
    test_predictions, test_auc, model = \
        train_test_with_fixed_params(train_data, test_data, phylo_tree, best_params)
    print "Test AUC is", test_auc
    output_path = os.path.join("predictions", "phylo.logistic.{0!s}".format(os.path.basename(testing_data_file).replace(".pkl", "")))
    open(output_path, "w").write("\n".join(str(x) for x in test_predictions))
    json.dump(best_params, open(output_path + ".parameters", "w"))
