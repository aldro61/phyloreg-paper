# Autograd

Some experiments using autograd to train the algorithms. We are interested in seeing:

1. Is it fast enough to do gradient descent in python/numpy with autograd

Yes, definitely.

2. Does autograd respect memory optimizations (e.g. if the objective is computed block-wise, is the gradient also computed like that?)

Probably, but it doesn't matter for now. We just load all the orthologs into memory.

3. Does an autograd implementation find the same solution as our current implementations? (If no, then there might be a bug in our stuff)

Working on that.


## Code

We have a framework to design learners based on autograd. This shoudl greatly speed up our development process. The code is available at [https://github.com/aldro61/phyloreg/blob/master/phyloreg/autograd_classifiers.py](https://github.com/aldro61/phyloreg/blob/master/phyloreg/autograd_classifiers.py)
