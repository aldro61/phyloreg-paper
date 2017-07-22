# Phylogenetic regularization

## Status updates

### July 22 2017

* New results for the autograd implementations show that phylogenetic regularization does help achieve better testing set AUCs (train on file #270, test on file #269):
  * RR: (running)
  * Logistic: 0.759
  * PhyloRR: 0.802
  * PhyloLogistic: 0.807
* Protocol: Spearmint was used to choose the hyperparameters within the following ranges:
{'sigma': {'type': 'float', 'min': 1e-5, 'max': 1e0}, 'alpha': {'type': 'float', 'min': 1e-8, 'max': 1e4}, 'beta': {'type': 'float', 'min': 1e-8, 'max': 1e4}, 'opti\_lr': {'type': 'float', 'min': 1e-5, 'max': 1e-1}, 'opti\_clip\_norm': {'type': 'float', 'min': 1e0, 'max': 1e3}}. The hyperparameter search was limited to 310 combinations for each algorithm.
* Note: I believe that the hyperparameter ranges tried are way too large and once we have run experiments on many datasets we will have a better idea of the range of values of interest. This will speed up HP search.
                       

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
