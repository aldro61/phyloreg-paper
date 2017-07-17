# Autograd

Some experiments using autograd to train the algorithms. We are interested in seeing:

1. Is it fast enough to do gradient descent in python/numpy with autograd
2. Does autograd respect memory optimizations (e.g. if the objective is computed block-wise, is the gradient also computed like that?)
3. Does an autograd implementation find the same solution as our current implementations? (If no, then there might be a bug in our stuff)
