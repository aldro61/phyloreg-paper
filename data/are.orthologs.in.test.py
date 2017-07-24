"""
Ortholog pooling works impressively well. This is really surprising. Can this
be because for some reason some of the orthologous sequences are sequences in
the set?

"""
import cPickle as c
import numpy as np

train_file = "270.pkl"
test_file = "269.pkl"

train_data = c.load(open(train_file, "r"))
test_data = c.load(open(test_file, "r"))

X_test = np.vstack(test_data["labelled_examples"].values())
y_test = np.hstack(test_data["labels"].values())
del test_data

# Hash testing set
X_test_hashed = {}  # Create a lookup table
for i, x in enumerate(X_test):
    x.flags['WRITEABLE'] = False
    X_test_hashed[hash(x.data)] = i

count = 0
for orthologs in train_data["ortho_info"].itervalues():
    for x in orthologs["X"]:
        assert len(x) == X_test.shape[1]
        count += 1
        x.flags["WRITEABLE"] = False
        if X_test_hashed.has_key(hash(x.data)):
            print "FOUND A MATCH! THATS BAD!"
print "Looked at {} orthologous sequences".format(count)
