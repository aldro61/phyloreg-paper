# Phylogenetic regularization

## Status updates

### July 16 2017

* Results for training on #270 and testing on #269:
  * PhyloRR: AUC of 0.7113 on test
  * RR: AUC of 0.7690
* Modified the logistic regression code to stop if a nan or inf value is encountered in the gradient. New class parameter: self.stop\_on\_nan\_inf.
* Fixed major memory leaks in Python bindings. We now keep copies of the feature, labels, adjacency matrices and tell Python that it can delete its own copy. This also avoids having a pointer to a corrupted matrix for which the memory was freed by the Python garbage collector.

### July 14-15 2017

* Wrote code to select the hyperparameters of ridge regression with and without phylo regularization.
* Launched computations on a single data set (270-269 data files).
* Results to come shortly



## Tools that could be useful

* [Autograd](https://github.com/HIPS/autograd) [[paper](https://indico.lal.in2p3.fr/event/2914/session/1/contribution/6/3/material/paper/0.pdf)]
* [Tensorflow](https://www.tensorflow.org/)
* [Keras](https://keras.io/)
