# Sub-projects

We want to show that using phylogenetic regularization increases the prediction accuracy of predictors (classifiers and possibly regressors). To achieve this, we will compare the following type of algorithms:

1. Phylo [algorithm]: the algorithm to which the phyloreg term was added to include orthologous sequences
2. Pooled [algorithm]: the algorithm is trained using the ortholog sequences and assigning them the label of the labelled example.
3. We also compare to the vanilla version of each algorithm

## Logistic Regression vs Phylo Logistic Regression

Result: 
1.  ___ is better in terms of AUC on test set. Hyperparameters were learned using grid search over
range of parameters.
2. _____ is better in terms of AUC on test set. Hyperparameters were learned using simple-spearmint tool.

## Ridge Regression vs Phylo Ridge Regression
Result:
1. __ is better in terms of AUC on test set. Hyperparameters were selected as described in 1.a
2. ____ is better in terms of AUC on test set. Hyperparameters were selected as described in 1.b

## Pooled Logistic Regression vs Phylo Logistic Regression
Result:
1. PLR is better in terms of AUC on test set. PLR's Hyperparameters were learned 
using grid search on train set with 10-fold cross validation. While, PMLR's hyper-parameters
were selected by training on train set with range of values and by AUC score on test set.
2. ______ is better in terms of AUC on test set. Hyperparameters were learned as described in 1.b

## Pooled Ridge Regression vs Phylo Ridge Regression
Result:
1. PRR is better in terms of AUC on test set. Hyperparameters were learned as described in 3.a
2. ____ is better in terms of AUC on test set. Hyperparameters were learned as described in 1.b

## di-mismatch string kernel SVM >> Pooled Logistic Regression >= Weighted Pooled Logistic Regression >> Phylo Logistic Regression
[ Hyperparameters of first was learned using 10-fold CV, 
of second and third were learned using grid search,
while for PMLR as described in 1.b ]


## Phylo CNN vs CNN

## Phylo RNN vs RNN
