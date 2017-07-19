# Phylogenetic regularization

## Status updates

### July 19 2017

* Found a problem with our manifold terms. We were using 0-valued vectors for missing orthologs, but this doesn't make sense since it leads to a cost! Working on a fix.
* For the autograd classifiers, it will be **very important that the data is shuffled** prior to training. I did not implement this in the learner to avoid creating a copy of the data.
* The ortholog loss function bug has been fixed. Will launch a spearmint experiment on the CHUL clusters later today.

### July 18 2017

* Created a framework to implement learners that use phylogenetic regularization with autograd.
* Implemented all our learning algorithms in this framework
* Experimented with some tricks to help the optimization.
  * Gradient clipping seems to help avoiding an explosion of the gradient/objective due to the manifold term
  * I saw that when using gradient clipping, we were able to fit the data very well, even with larger beta values, without catching nan or inf values
* Will now launch Spearmint experiments on the 269/270 dataset

### July 17 2017

* Started working on autograd implementations of the algorithms as a sanity check for our other implementations
* If this yields implementations that are fast enough, we will be able to do prototyping very fast using autograd + spearmint
* Still have some work to do before comitting and running experiments

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
